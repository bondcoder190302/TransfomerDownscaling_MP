import torch
import torch.nn.functional as F
from torch import nn
from basicsr.utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class MaskedExtremeWeightedCharbonnierLoss(nn.Module):
    """
    Masked Charbonnier Loss with dynamic penalties for extreme precipitation events.
    
    This loss function is designed specifically for precipitation downscaling where 
    the target variables (e.g., CHIRPS) have been log1p transformed and Z-score 
    normalized. 
    
    Ocean pixels are handled naturally without explicit masking inside this class: 
    because the Land-Sea Mask (LSM) is multiplied to both the prediction and target 
    in the main model class before loss calculation, ocean pixels arrive as exactly 0.0. 
    The Charbonnier epsilon is squared mathematically to prevent NaN gradients on 
    these zeroed-out ocean pixels during backpropagation.
    """
    def __init__(self, loss_weight=1.0, reduction='sum', eps=1e-3, 
                 extreme_threshold=2.89, extreme_weight=3.0,
                 wet_weight=0.0, wet_threshold=-0.8, wet_scale=0.5):
        """
        Args:
            loss_weight (float): Global multiplier applied to the final reduced loss.
            reduction (str): Specifies the reduction to apply to the output: 
                             'sum' | 'mean' | 'none'. Default: 'sum'.
            eps (float): Epsilon value to prevent division by zero in Charbonnier 
                         loss derivative. Default: 1e-3.
            extreme_threshold (float): The log-z-score value above which the extreme 
                                       penalty starts ramping up. (e.g., 2.89 represents 
                                       the 90th percentile of CHIRPS precipitation).
            extreme_weight (float): The penalty multiplier applied to the loss when 
                                    rainfall reaches or exceeds the threshold.
            wet_weight (float): Weight for the optional wet-day occurrence BCE loss. 
                                Set to 0.0 to disable. Default: 0.0.
            wet_threshold (float): Z-score boundary separating dry days from wet days.
            wet_scale (float): Temperature scaling factor for the BCE sigmoid function.
        """
        super(MaskedExtremeWeightedCharbonnierLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps
        
        self.extreme_threshold = extreme_threshold
        self.extreme_weight = extreme_weight
        
        self.wet_weight = wet_weight
        self.wet_threshold = wet_threshold
        self.wet_scale = wet_scale

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): Predicted precipitation tensor of shape (B, C, H, W).
            target (Tensor): Ground truth precipitation tensor of shape (B, C, H, W).
            
        Returns:
            Tensor: The calculated loss scalar (if reduction is sum/mean) or map.
        """
        # 1. Base Charbonnier Loss
        # We square the epsilon (self.eps**2) inside the square root. For ocean pixels 
        # where pred = 0.0 and target = 0.0, this prevents a 0/0 gradient singularity.
        base_loss = torch.sqrt((pred - target)**2 + self.eps**2)
        
        # 2. Normalised Extreme-Event Weighting Ramp
        # Evaluates the ground truth target. If target > extreme_threshold, the penalty 
        # multiplier scales smoothly instead of applying a harsh step function.
        # This normalisation prevents gradient explosion at high extremes.
        excess = torch.relu(target - self.extreme_threshold)
        weight_map = 1.0 + (self.extreme_weight - 1.0) * (excess / self.extreme_threshold)
        
        # Apply the penalty multiplier to the base Charbonnier loss map
        weighted_loss = base_loss * weight_map
        
        # 3. Optional Soft Wet-Day BCE (Disabled if wet_weight == 0.0)
        # Penalizes the model for missing the occurrence of rain entirely (dry vs wet).
        if self.wet_weight > 0.0:
            gt_wet = torch.sigmoid((target - self.wet_threshold) / self.wet_scale)
            pr_wet = torch.sigmoid((pred - self.wet_threshold) / self.wet_scale)
            
            # Detach the target soft-labels so gradients only flow through the prediction
            occ_loss = F.binary_cross_entropy(pr_wet, gt_wet.detach(), reduction='none')
            total_loss = weighted_loss + (self.wet_weight * occ_loss)
        else:
            total_loss = weighted_loss

        # 4. Built-in Reduction
        # We use standard PyTorch reductions since the model architecture handles 
        # spatial ocean masking independently prior to the loss call.
        if self.reduction == 'sum':
            return self.loss_weight * total_loss.sum()
        elif self.reduction == 'mean':
            return self.loss_weight * total_loss.mean()
        else:
            return self.loss_weight * total_loss