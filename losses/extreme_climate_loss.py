import torch
import torch.nn.functional as F
from torch import nn
from basicsr.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class MaskedExtremeWeightedCharbonnierLoss(nn.Module):
    """
    Charbonnier loss with two-tail extreme weighting, an asymmetric
    under-prediction penalty, and an optional wet-day BCE term, for
    precipitation downscaling. Inspired by Chandel et al. (2025, JGR Atmos).

    Operates in log1p + z-score space. The LSM is applied by the model class
    BEFORE this loss, so ocean pixels arrive as pred=0, target=0 (zero gradient).

    -- PER-PIXEL vs SCALAR thresholds ---------------------------------------
    forward() accepts optional p90 / p10 tensors (B,1,H,W), already cropped to
    match pred/target by the dataset (they ride the same crop as lsm_hr).

      * If p90 is given  -> PER-PIXEL DISCRETE weighting (faithful to Chandel
        eq. 3): weight = extreme_weight where target > p90(grid), else 1.0;
        plus low_weight where target < p10(grid). extreme_weight = Chandel r2,
        low_weight = Chandel r1.

      * If p90 is None   -> SCALAR fallback: continuous ramp from
        extreme_threshold to 2*extreme_threshold (previous behaviour).

    -- Asymmetric under-prediction penalty ----------------------------------
    Always applied on top: where pred < target, weight *= under_weight.
    Counters the dry bias (log-space bias -0.18, heavy-day bias -20 mm). An
    under-predicted extreme grid therefore gets extreme_weight * under_weight.

    Recommended per-pixel config:
        extreme_weight = 1.5 (r2)   under_weight = 1.5
        low_weight     = 1.0 (off)  wet_weight   = 0.5
    """

    def __init__(
        self,
        loss_weight=1.0,
        reduction='sum',
        eps=1e-3,
        # Upper tail (SCALAR fallback only, when p90 not passed)
        extreme_threshold=3.1916,
        extreme_weight=1.5,        # Chandel r2
        # Asymmetric under-prediction penalty (always on)
        under_weight=1.0,
        # Lower tail (low_weight=1.0 disables)
        low_threshold=None,
        low_weight=1.0,            # Chandel r1
        # Wet-day BCE occurrence term
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

    def forward(self, pred, target, p90=None, p10=None, **kwargs):
        # 1. Base Charbonnier
        diff = pred - target
        base_loss = torch.sqrt(diff * diff + self.eps * self.eps)

        # 2. Extreme-event weight map
        if p90 is not None:
            # PER-PIXEL discrete (Chandel eq. 3): step at the local P90
            upper_mask = (target > p90).float()
            weight_map = 1.0 + (self.extreme_weight - 1.0) * upper_mask
            if p10 is not None and self.low_weight != 1.0:
                lower_mask = (target < p10).float()
                weight_map = weight_map + (self.low_weight - 1.0) * lower_mask
        else:
            # SCALAR fallback: continuous ramp threshold -> 2*threshold
            excess_upper = torch.relu(target - self.extreme_threshold)
            upper_ramp   = torch.clamp(excess_upper / self.extreme_threshold, max=1.0)
            weight_map   = 1.0 + (self.extreme_weight - 1.0) * upper_ramp
            if self.low_threshold is not None and self.low_weight != 1.0:
                lower_mask = (target < self.low_threshold).float()
                weight_map = weight_map + (self.low_weight - 1.0) * lower_mask

        # 3. Asymmetric under-prediction penalty (always)
        if self.under_weight != 1.0:
            under_mask = (pred < target).float()
            weight_map = weight_map * (1.0 + (self.under_weight - 1.0) * under_mask)

        weighted_loss = base_loss * weight_map

        # 4. Optional soft wet-day BCE occurrence term
        if self.wet_weight > 0.0:
            pred_logits = (pred   - self.wet_threshold) / self.wet_scale
            tgt_logits  = (target - self.wet_threshold) / self.wet_scale
            gt_wet = torch.sigmoid(tgt_logits).expand_as(pred_logits).detach()
            occ_loss = F.binary_cross_entropy_with_logits(
                pred_logits, gt_wet, reduction='none'
            )
            total_loss = weighted_loss + self.wet_weight * occ_loss
        else:
            total_loss = weighted_loss

        # 5. Reduction (sum is normalised by norm_factor in optimize_parameters)
        if self.reduction == 'sum':
            return self.loss_weight * total_loss.sum()
        elif self.reduction == 'mean':
            return self.loss_weight * total_loss.mean()
        else:
            return self.loss_weight * total_loss