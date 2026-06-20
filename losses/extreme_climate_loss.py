import torch
import torch.nn.functional as F
from torch import nn
from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class MaskedExtremeWeightedCharbonnierLoss(nn.Module):
    """
    Masked Charbonnier Loss with higher penalties for extreme precipitation events.
    
    Operates on log1p + z-score normalised CHIRPS targets.

    Three components
    ----------------
    1. Masked Charbonnier  – standard smooth-L1 restricted to land pixels.
    2. Normalised extreme ramp – weight = 1 + (extreme_weight - 1) * excess / extreme_threshold
       This is bounded: at excess == extreme_threshold the multiplier reaches exactly
       extreme_weight, then continues to grow linearly beyond that.
       (The old formula `1 + extreme_weight * excess` reached extreme_weight × threshold
       at the same point, making it ~2.8× more aggressive than intended.)
    3. Soft wet-day BCE (optional, disabled by default) – adds a binary cross-entropy
       term that penalises disagreement on rain occurrence, addressing the dry-day
       imbalance independently of intensity errors.

    Ocean-pixel epsilon fix
    -----------------------
    When the land-sea mask is applied *externally* in optimize_parameters (both
    self.output and gt_lsm are multiplied by lsm_hr before reaching this loss),
    ocean pixels arrive here with pred == 0 and target == 0 exactly.  Charbonnier
    with eps > 0 gives sqrt(eps) ≈ 0.032 even for a perfect 0-0 prediction, which
    leaks systematically into the sum loss.  We detect those pixels via
    `(target != 0.0)` and zero-out their contribution.
    Land dry-pixels are safe: their z-score is (log1p(0) – mean)/std ≈ -0.44, never
    exactly 0.

    YAML example
    ------------
    pixel_opt:
      type: MaskedExtremeWeightedCharbonnierLoss
      loss_weight: 10
      reduction: sum          # norm_factor division is applied in optimize_parameters
      eps: !!float 1e-3
      extreme_threshold: 2.773   # >20 mm/day in CHIRPS log-z-score space
      extreme_weight: 5.0        # at exactly the threshold, weight = extreme_weight
      wet_weight: 0.0            # set > 0 (e.g. 0.5) to enable occurrence BCE term
      wet_threshold: -0.9        # z-score of the dry/wet boundary (compute from stats pkl)
      wet_scale: 0.3             # sigmoid temperature for soft wet-day boundary
    """

    def __init__(
        self,
        loss_weight=1.0,
        reduction='mean',
        eps=1e-3,
        extreme_threshold=2.773,
        extreme_weight=5.0,
        wet_weight=0.0,
        wet_threshold=-0.9,
        wet_scale=0.3,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps
        self.extreme_threshold = extreme_threshold
        self.extreme_weight = extreme_weight
        self.wet_weight = wet_weight
        self.wet_threshold = wet_threshold
        self.wet_scale = wet_scale

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reduce(self, loss, valid_mask):
        """Reduce a pixel-wise loss tensor using the given valid-pixel mask."""
        if valid_mask is not None:
            loss = loss * valid_mask
            if self.reduction == 'mean':
                n_valid = valid_mask.sum().clamp_min(1.0)
                return loss.sum() / n_valid
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss
        else:
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, pred, target, mask=None, **kwargs):
        """
        Args:
            pred   (Tensor): model output  (B, C, H, W), log-z-score space
            target (Tensor): ground truth  (B, C, H, W), log-z-score space
            mask   (Tensor, optional): explicit land-sea mask (B, 1, H, W).
                   1 = land, 0 = ocean.  If None, ocean pixels are auto-detected
                   from target == 0.0 (valid when lsm masking is applied upstream).
        """
        # ── 1. Build valid-pixel mask ──────────────────────────────────────────
        if mask is not None:
            # Explicit mask provided (e.g., called directly with lsm_hr)
            valid_mask = mask.float()
            if valid_mask.shape != target.shape:
                valid_mask = valid_mask.expand_as(target)
        else:
            # Auto-detect ocean pixels: lsm was applied externally so ocean = 0.0 exactly.
            # Land dry-pixels have z-score ≈ -0.44, never exactly 0.
            valid_mask = (target != 0.0).float()
            # Safety: if everything is non-zero (no external masking), use all pixels
            if valid_mask.all():
                valid_mask = None

        # ── 2. Charbonnier base loss ───────────────────────────────────────────
        base_loss = torch.sqrt((pred - target) ** 2 + self.eps)

        # ── 3. Normalised extreme-event weighting ramp ─────────────────────────
        # At excess == 0          → weight = 1.0  (no change for moderate rainfall)
        # At excess == threshold  → weight = extreme_weight  (exactly)
        # Beyond threshold        → weight grows linearly (unbounded but gradual)
        excess = torch.relu(target - self.extreme_threshold)
        weight_map = 1.0 + (self.extreme_weight - 1.0) * excess / self.extreme_threshold

        weighted_loss = base_loss * weight_map

        # ── 4. Reduce Charbonnier component ───────────────────────────────────
        final_loss = self._reduce(weighted_loss, valid_mask)

        # ── 5. Optional soft wet-day BCE ──────────────────────────────────────
        if self.wet_weight > 0.0:
            gt_wet = torch.sigmoid((target - self.wet_threshold) / self.wet_scale)
            pr_wet = torch.sigmoid((pred   - self.wet_threshold) / self.wet_scale)
            # Use gt_wet.detach() so the occurrence term only trains the prediction side
            occ_loss = F.binary_cross_entropy(
                pr_wet, gt_wet.detach(), reduction='none'
            )
            occ_reduced = self._reduce(occ_loss, valid_mask)
            final_loss = final_loss + self.wet_weight * occ_reduced

        return final_loss * self.loss_weight
