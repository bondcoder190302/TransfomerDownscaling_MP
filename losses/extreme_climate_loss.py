import torch
import torch.nn.functional as F
from torch import nn
from basicsr.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class MaskedExtremeWeightedCharbonnierLoss(nn.Module):
    """
    Charbonnier Loss with two-tail extreme-event weighting for precipitation
    downscaling, inspired by Chandel et al. (2025, JGR Atmospheres).

    Operates in log1p + Z-score normalized space.  The Land-Sea Mask (LSM) is
    applied by the model class BEFORE calling this loss, so ocean pixels arrive
    as pred=0.0 and target=0.0.  For those pixels the Charbonnier gradient is
    exactly d/d(pred)[sqrt((0-0)^2 + eps^2)] = 0/eps = 0.0, so no gradient
    signal leaks from ocean pixels — no explicit mask check is needed here.

    Weight map (Chandel et al. eq. 2-3 adapted for continuous ramp):
        target < low_threshold  → weight = low_weight    (lower-tail upweight)
        low_threshold ≤ target ≤ extreme_threshold → weight = 1.0
        target > extreme_threshold → weight ramps linearly from 1.0 at
            extreme_threshold up to extreme_weight at 2*extreme_threshold,
            then SATURATES at extreme_weight.  The saturation cap prevents
            gradient explosion for rare outlier targets beyond 2*threshold.

    Chandel's best values (from grid-search over {1, 1.5, 2, 3, 4}):
        low_weight   = 2.0  (r1, for below P10)
        extreme_weight = 1.5  (r2, for above P90)
    We use higher extreme_weight here because our Charbonnier base loss is
    smaller in magnitude than MSE, requiring a stronger push on extremes.
    """

    def __init__(
        self,
        loss_weight=1.0,
        reduction='sum',
        eps=1e-3,
        # ── Upper tail (P90) ──────────────────────────────────────────────
        extreme_threshold=2.773,   # z-score of ~20 mm/day in CHIRPS log-z space
        extreme_weight=5.0,        # weight SATURATES at this value at 2×threshold
        # ── Lower tail (P10) ─────────────────────────────────────────────
        # Set low_weight=1.0 (default) to DISABLE lower-tail upweighting.
        # Set low_threshold to a physically meaningful value, e.g. the z-score
        # of the wet-day boundary (log1p(2.5mm) z-scored ≈ +0.88) to upweight
        # pixels that are transitional rain (light → moderate).
        # For the dry-day sentinel (ocean/masked pixels at z=0.0) you do NOT
        # want to include those in the lower-tail weight, so set low_threshold
        # to a value clearly above 0.0 (e.g. 0.5 or 0.88).
        low_threshold=None,        # z-score below which lower-tail weight applies
        low_weight=1.0,            # multiplier for lower-tail pixels (Chandel r1=2.0)
        # ── Wet-day BCE occurrence term ───────────────────────────────────
        wet_weight=0.0,            # 0.0 = disabled; try 0.2–0.5 to enable
        # wet_threshold: z-score corresponding to the wet/dry boundary.
        # CORRECT value = (log1p(0.0) - mu) / sigma = (0.0 - 0.4180) / 0.9470 = -0.4414
        # The YAML previously used -0.9, which is physically incorrect (it
        # implies a negative precipitation value of -0.35 mm/day).
        wet_threshold=-0.4414,
        wet_scale=0.3,             # sigmoid temperature; smaller = harder boundary
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

        self.extreme_threshold = extreme_threshold
        self.extreme_weight = extreme_weight

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
            Scalar loss (if reduction='sum'/'mean') or per-element map.
        """
        # ── 1. Base Charbonnier Loss ──────────────────────────────────────
        # Standard Charbonnier: sqrt((pred - target)^2 + eps^2).
        # At pred==target==0 (masked ocean pixels) gradient is exactly 0/eps=0,
        # confirming no gradient contamination from masked pixels.
        diff = pred - target
        base_loss = torch.sqrt(diff * diff + self.eps * self.eps)

        # ── 2. Two-Tail Extreme-Event Weight Map ─────────────────────────
        # UPPER TAIL: linearly ramp from 1.0 at extreme_threshold up to
        # extreme_weight at 2*extreme_threshold, then saturate (clamp).
        # This is the "smooth step" equivalent of Chandel's discrete r2 weight.
        #
        # Why saturation matters:
        #   Without clamp, a target of 7.0 z (≈1100 mm/day, unrealistic but
        #   possible from data noise) would reach weight=7.1x, potentially
        #   destabilising training.  The saturation cap ensures the maximum
        #   gradient amplification is controlled and matches the intended
        #   extreme_weight parameter.
        #
        # Scale interpretation:
        #   At target = extreme_threshold:     weight = 1.0   (ramp starts)
        #   At target = 2*extreme_threshold:   weight = extreme_weight (cap)
        #   At target > 2*extreme_threshold:   weight = extreme_weight (flat)
        excess_upper = torch.relu(target - self.extreme_threshold)
        upper_ramp = torch.clamp(excess_upper / self.extreme_threshold, max=1.0)
        weight_map = 1.0 + (self.extreme_weight - 1.0) * upper_ramp

        # LOWER TAIL: step-function upweight for light-rain/near-dry pixels
        # (inspired by Chandel r1=2.0 for target < P10).
        # NOTE: masked ocean pixels sit at target=0.0 in z-space, which is
        # ABOVE the physical dry-day z-score of -0.4414.  If low_threshold
        # is set below 0.0, ocean pixels would get incorrectly upweighted.
        # Keep low_threshold > 0.0 (e.g. 0.88, the wet-day boundary) to
        # avoid this.  When low_threshold is None or low_weight==1.0, this
        # branch is a no-op.
        if self.low_threshold is not None and self.low_weight != 1.0:
            low_mask = (target < self.low_threshold).float()
            weight_map = weight_map + (self.low_weight - 1.0) * low_mask

        weighted_loss = base_loss * weight_map

        # ── 3. Optional Soft Wet-Day BCE Occurrence Term ──────────────────
        # When enabled (wet_weight > 0), adds a soft Binary Cross-Entropy term
        # that penalises the model for predicting rain where the target is dry
        # and vice-versa.  This is complementary to the regression loss and
        # helps sharpen the wet/dry boundary without requiring a hard threshold.
        #
        # NUMERICAL STABILITY: we use binary_cross_entropy_with_logits (log-sum-
        # exp trick) rather than BCE(sigmoid(x), ...) which computes log(0)
        # when sigmoid saturates, producing -inf and NaN gradients.
        if self.wet_weight > 0.0:
            pred_logits = (pred   - self.wet_threshold) / self.wet_scale
            tgt_logits  = (target - self.wet_threshold) / self.wet_scale
            # soft labels: no gradient through target
            gt_wet = torch.sigmoid(tgt_logits).detach()
            occ_loss = F.binary_cross_entropy_with_logits(
                pred_logits, gt_wet, reduction='none'
            )
            total_loss = weighted_loss + self.wet_weight * occ_loss
        else:
            total_loss = weighted_loss

        # ── 4. Reduction ──────────────────────────────────────────────────
        # For reduction='sum', the per-land-pixel normalisation is applied
        # in optimize_parameters (divides by norm_factor = sum(lsm_hr)).
        # Ocean pixels contribute sqrt(eps^2) = eps to the sum, but their
        # gradient is 0, so training is unaffected; only logged loss values
        # are slightly inflated.  This is a cosmetic issue, not a training bug.
        if self.reduction == 'sum':
            return self.loss_weight * total_loss.sum()
        elif self.reduction == 'mean':
            return self.loss_weight * total_loss.mean()
        else:
            return self.loss_weight * total_loss
