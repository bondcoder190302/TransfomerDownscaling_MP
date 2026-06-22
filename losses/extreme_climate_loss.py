import torch
import torch.nn.functional as F
from torch import nn
from basicsr.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class MaskedExtremeWeightedCharbonnierLoss(nn.Module):
    """
    Charbonnier Loss with two-tail extreme-event weighting and asymmetric
    under-prediction penalty for precipitation downscaling.

    Inspired by Chandel et al. (2025, JGR Atmospheres).

    Operates in log1p + Z-score normalised space.  The Land-Sea Mask (LSM) is
    applied by the model class BEFORE calling this loss, so ocean pixels arrive
    as pred=0.0 and target=0.0.  For those pixels the Charbonnier gradient is
    exactly d/d(pred)[sqrt((0-0)^2 + eps^2)] = 0/eps = 0.0, so no gradient
    signal leaks from ocean pixels — no explicit mask check is needed here.

    ── Weight map (applied to Charbonnier base loss) ────────────────────────
    Step 1 – Upper tail ramp (P90):
        target ≤ extreme_threshold          → weight = 1.0
        extreme_threshold < target
            < 2 * extreme_threshold         → weight ramps linearly from 1.0
                                               up to extreme_weight
        target ≥ 2 * extreme_threshold      → weight = extreme_weight (saturates)

    Step 2 – Asymmetric under-prediction penalty:
        pred < target (under-predicting)    → weight *= under_weight
        pred ≥ target (over-predicting)     → weight unchanged
        This directly penalises the dry bias observed in diagnostics.

    Step 3 – Lower tail (optional, disabled when low_weight=1.0):
        target < low_threshold              → weight += (low_weight - 1.0)
        Currently disabled (low_weight=1.0) to avoid distracting from the
        extreme under-prediction problem.

    ── Optional BCE occurrence term ─────────────────────────────────────────
    When wet_weight > 0, adds a soft Binary Cross-Entropy term that penalises
    wet/dry spatial mismatches independently of magnitude errors.

    ── Calibrated thresholds (India, CHIRPS log1p z-score space) ───────────
    CHIRPS log1p stats:  mean=0.4180, std=0.9470, max=6.5211
    extreme_threshold  = 3.1916  →  wet-day P90 ≈ 30.2 mm/day
    wet_threshold      = -0.4414 →  0 mm/day (wet/dry boundary)
    low_threshold      = -0.4414 →  disabled (low_weight=1.0)
    """

    def __init__(
        self,
        loss_weight=1.0,
        reduction='sum',
        eps=1e-3,

        # ── Upper tail (P90 of wet days) ──────────────────────────────────
        extreme_threshold=3.1916,  # z-score of wet-day P90 ≈ 30.2 mm/day
        extreme_weight=1.5,        # Chandel r2; weight saturates at this value

        # ── Asymmetric under-prediction penalty ───────────────────────────
        # Multiplies the Charbonnier weight when pred < target.
        # 1.0 = symmetric (disabled).  Try 1.5–2.0 to counter dry bias.
        # Raise carefully: if validation bias flips to positive (wet), reduce.
        under_weight=1.0,

        # ── Lower tail (P10) ─────────────────────────────────────────────
        # Set low_weight > 1.0 AND a physically meaningful low_threshold to
        # upweight light-rain pixels (e.g. threshold=0.88 = z-score of 2.5mm).
        # Currently disabled (low_weight=1.0) — set both to enable.
        # IMPORTANT: keep low_threshold > 0.0 so ocean pixels (forced to
        # z=0.0 by LSM) are NOT caught by this branch.
        low_threshold=None,
        low_weight=1.0,

        # ── Wet-day BCE occurrence term ───────────────────────────────────
        # Adds a soft BCE that penalises wet/dry spatial mismatches.
        # wet_threshold = -0.4414 = z-score of log1p(0mm) = (0 - 0.4180)/0.9470
        # wet_scale: sigmoid temperature; smaller = sharper wet/dry boundary.
        # Ocean pixels (pred=target=0.0 in z-space) produce identical logits
        # → BCE gradient = sigmoid(logit) - gt_wet = 0, so no ocean leakage.
        wet_weight=0.0,
        wet_threshold=-0.4414,
        wet_scale=0.3,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

        self.extreme_threshold = extreme_threshold
        self.extreme_weight = extreme_weight

        self.under_weight = under_weight

        self.low_threshold = low_threshold
        self.low_weight = low_weight

        self.wet_weight = wet_weight
        self.wet_threshold = wet_threshold
        self.wet_scale = wet_scale

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred   (Tensor): (B, C, H, W) – network prediction in log-z space.
            target (Tensor): (B, 1, H, W) – CHIRPS ground truth in log-z space.
                             Broadcasts to (B, C, H, W) for multiscale outputs.
        Returns:
            Scalar loss (reduction='sum'/'mean') or per-element map.
        """
        # ── 1. Base Charbonnier Loss ──────────────────────────────────────
        # sqrt((pred - target)^2 + eps^2)
        # At pred == target == 0 (LSM-masked ocean pixels), gradient = 0.
        diff = pred - target
        base_loss = torch.sqrt(diff * diff + self.eps * self.eps)

        # ── 2. Upper-tail ramp weight ─────────────────────────────────────
        # Linearly ramps from 1.0 at extreme_threshold to extreme_weight
        # at 2 * extreme_threshold, then saturates (clamps) at extreme_weight.
        # Saturation prevents gradient explosion from rare data-noise spikes.
        #
        #   target = extreme_threshold       → weight = 1.0  (ramp starts)
        #   target = 2 * extreme_threshold   → weight = extreme_weight (cap)
        #   target > 2 * extreme_threshold   → weight = extreme_weight (flat)
        excess_upper = torch.relu(target - self.extreme_threshold)
        upper_ramp   = torch.clamp(excess_upper / self.extreme_threshold, max=1.0)
        weight_map   = 1.0 + (self.extreme_weight - 1.0) * upper_ramp

        # ── 3. Asymmetric under-prediction penalty ────────────────────────
        # Multiplies the weight map when pred < target so that the model
        # pays a heavier penalty for predicting too low than too high.
        # This directly addresses the dry bias confirmed in diagnostics
        # (log-space bias -0.18, heavy-day bias -20 mm/day).
        #
        # Effect by pixel type:
        #   Extreme under-predicted  (target>P90, pred<target): weight * extreme_weight * under_weight
        #   Normal under-predicted   (pred<target):              weight * under_weight
        #   Over-predicted           (pred≥target):              weight unchanged
        if self.under_weight != 1.0:
            under_mask = (pred < target).float()
            weight_map = weight_map * (1.0 + (self.under_weight - 1.0) * under_mask)

        # ── 4. Lower-tail weight (disabled when low_weight=1.0) ──────────
        # Step-function upweight for pixels below low_threshold.
        # Currently disabled: low_weight=1.0 means this branch is skipped.
        if self.low_threshold is not None and self.low_weight != 1.0:
            low_mask   = (target < self.low_threshold).float()
            weight_map = weight_map + (self.low_weight - 1.0) * low_mask

        weighted_loss = base_loss * weight_map

        # ── 5. Optional soft wet-day BCE occurrence term ──────────────────
        # Adds a soft BCE on a sigmoid-transformed wet-probability label.
        # Uses binary_cross_entropy_with_logits (log-sum-exp trick) for
        # numerical stability — avoids log(0) NaN from a hard sigmoid.
        #
        # wet_scale=0.3 creates a ±0.3 z-score transition band around
        # wet_threshold (-0.4414 = z-score of 0mm), giving a sharp but
        # smooth wet/dry boundary.
        if self.wet_weight > 0.0:
            pred_logits = (pred   - self.wet_threshold) / self.wet_scale
            tgt_logits  = (target - self.wet_threshold) / self.wet_scale
            # expand_as broadcasts (B,1,H,W) → (B,C,H,W) so BCE shape matches pred
            gt_wet      = torch.sigmoid(tgt_logits).expand_as(pred_logits).detach()
            occ_loss    = F.binary_cross_entropy_with_logits(
                pred_logits, gt_wet, reduction='none'
            )
            total_loss = weighted_loss + self.wet_weight * occ_loss
        else:
            total_loss = weighted_loss

        # ── 6. Reduction ──────────────────────────────────────────────────
        # For reduction='sum', per-land-pixel normalisation is done in
        # optimize_parameters via norm_factor = sum(lsm_hr).
        if self.reduction == 'sum':
            return self.loss_weight * total_loss.sum()
        elif self.reduction == 'mean':
            return self.loss_weight * total_loss.mean()
        else:
            return self.loss_weight * total_loss