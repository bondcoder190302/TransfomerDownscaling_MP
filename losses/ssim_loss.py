import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY


def _gaussian(kernel_size, sigma, dtype: torch.dtype, device: torch.device):
    dist = torch.arange(
        start=(1 - kernel_size) / 2,
        end=(1 + kernel_size) / 2,
        step=1,
        dtype=dtype,
        device=device)
    gauss = torch.exp(-torch.pow(dist / sigma, 2) / 2)
    return (gauss / gauss.sum()).unsqueeze(dim=0)


def _gaussian_kernel(channel, kernel_size, sigma, dtype: torch.dtype, device: torch.device):
    gaussian_kernel_x = _gaussian(kernel_size[0], sigma[0], dtype, device)
    gaussian_kernel_y = _gaussian(kernel_size[1], sigma[1], dtype, device)
    kernel = torch.matmul(gaussian_kernel_x.t(), gaussian_kernel_y)
    return kernel.expand(channel, 1, kernel_size[0], kernel_size[1])


@LOSS_REGISTRY.register()
class ClimateSSIMLoss(nn.Module):
    def __init__(
            self,
            loss_weight=1.0,
            data_range=None,
            kernel_size=(11, 11),
            sigma=(1.5, 1.5),
            k1=0.01,
            k2=0.03,
            reduction='mean',
            range_eps=1e-8):
        super().__init__()
        self.loss_weight = loss_weight
        self.data_range = data_range
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.k1 = k1
        self.k2 = k2
        self.reduction = reduction
        self.range_eps = range_eps

        if reduction != 'mean':
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported mode: mean.')

    def forward(self, pred, target):
        if pred.shape != target.shape:
            raise ValueError(f'Image shapes are different: {pred.shape}, {target.shape}.')

        channel = pred.size(1)
        dtype = pred.dtype
        device = pred.device
        kernel = _gaussian_kernel(channel, self.kernel_size, self.sigma, dtype, device)

        pad_h = (self.kernel_size[0] - 1) // 2
        pad_w = (self.kernel_size[1] - 1) // 2

        pred = F.pad(pred, (pad_w, pad_w, pad_h, pad_h), mode='reflect')
        target = F.pad(target, (pad_w, pad_w, pad_h, pad_h), mode='reflect')

        input_list = torch.cat((pred, target, pred * pred, target * target, pred * target))
        outputs = F.conv2d(input_list, kernel, groups=channel)
        output_list = outputs.split(pred.shape[0])

        mu_pred_sq = output_list[0].pow(2)
        mu_target_sq = output_list[1].pow(2)
        mu_pred_target = output_list[0] * output_list[1]
        sigma_pred_sq = output_list[2] - mu_pred_sq
        sigma_target_sq = output_list[3] - mu_target_sq
        sigma_pred_target = output_list[4] - mu_pred_target

        if self.data_range is None:
            dynamic_range = (target.detach().amax(dim=(-2, -1), keepdim=True) -
                             target.detach().amin(dim=(-2, -1), keepdim=True))
            zero_range_mask = dynamic_range <= self.range_eps
            dynamic_range = dynamic_range.clamp_min(self.range_eps)
        else:
            dynamic_range = pred.new_tensor(float(self.data_range))
            zero_range_mask = None

        c1 = (self.k1 * dynamic_range)**2
        c2 = (self.k2 * dynamic_range)**2

        ssim_idx = ((2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)) / (
            (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2))
        h_slice = slice(pad_h, -pad_h if pad_h > 0 else None)
        w_slice = slice(pad_w, -pad_w if pad_w > 0 else None)
        ssim_idx = ssim_idx[..., h_slice, w_slice]
        ssim_values = ssim_idx.mean(dim=(-2, -1))

        if zero_range_mask is not None:
            zero_range_mask = zero_range_mask.squeeze(-1).squeeze(-1)
            ssim_values = torch.where(zero_range_mask, torch.ones_like(ssim_values), ssim_values)

        loss = 1.0 - ssim_values.mean()
        return self.loss_weight * loss
