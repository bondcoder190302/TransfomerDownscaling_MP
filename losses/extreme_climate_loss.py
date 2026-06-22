import torch
import torch.nn.functional as F
from torch import nn
from basicsr.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class MaskedExtremeWeightedCharbonnierLoss(nn.Module):
    """
    Masked Charbonnier Loss with dynamic penalties for extreme precipitation events
    and an asymmetric penalty for under-prediction to fix negative dry bias.
    """
    def __init__(self, loss_weight=1.0, reduction='sum', eps=1e-3, 
                 extreme_threshold=3.1916, extreme_weight=1.5,
                 under_weight=1.5,
                 wet_weight=0.5, wet_threshold=-0.4414, wet_scale=0.3):
        super(MaskedExtremeWeightedCharbonnierLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps
        
        self.extreme_threshold = extreme_threshold
        self.extreme_weight = extreme_weight
        self.under_weight = under_weight
        
        self.wet_weight = wet_weight
        self.wet_threshold = wet_threshold
        self.wet_scale = wet_scale

    def forward(self, pred, target, **kwargs):
        # 1. Base Charbonnier Loss (eps squared for 0.0 ocean safety)
        base_loss = torch.sqrt((pred - target)**2 + self.eps**2)
        
        # 2. Extreme Weighting Ramp (with saturation cap to prevent explosion)
        weight_map = torch.ones_like(target)
        extreme_mask = target > self.extreme_threshold
        if extreme_mask.any():
            excess = target[extreme_mask] - self.extreme_threshold
            ramp = 1.0 + (self.extreme_weight - 1.0) * torch.clamp(excess / self.extreme_threshold, max=1.0)
            weight_map[extreme_mask] = ramp
        
        # 3. Asymmetric Under-Prediction Penalty
        # Punishes the model an extra 'under_weight' times if it under-predicts the ground truth
        if self.under_weight != 1.0:
            under_mask = (pred < target).float()
            weight_map = weight_map * (1.0 + (self.under_weight - 1.0) * under_mask)
        
        weighted_loss = base_loss * weight_map
        
        # 4. Soft Wet-Day BCE (Stable version with logits)
        if self.wet_weight > 0.0:
            pred_logits = (pred - self.wet_threshold) / self.wet_scale
            tgt_logits  = (target - self.wet_threshold) / self.wet_scale
            gt_wet = torch.sigmoid(tgt_logits).detach()
            occ_loss = F.binary_cross_entropy_with_logits(pred_logits, gt_wet, reduction='none')
            total_loss = weighted_loss + (self.wet_weight * occ_loss)
        else:
            total_loss = weighted_loss

        # 5. Built-in Reduction
        if self.reduction == 'sum':
            return self.loss_weight * total_loss.sum()
        elif self.reduction == 'mean':
            return self.loss_weight * total_loss.mean()
        else:
            return self.loss_weight * total_loss