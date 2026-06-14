import numpy as np
import torch
import torch.nn.functional as F
from basicsr.utils.registry import METRIC_REGISTRY
import os

_LSM_HR_CACHE = None

def _get_lsm_mask(device, shape):
    global _LSM_HR_CACHE
    if _LSM_HR_CACHE is None:
        lsm_path = './DownScale_Paper/LSM_fix/india_lsm_HR.npy'
        if os.path.exists(lsm_path):
            lsm = np.load(lsm_path)
            _LSM_HR_CACHE = torch.from_numpy(lsm).float().to(device)
        else:
            _LSM_HR_CACHE = torch.ones(shape, device=device) # Dummy fallback
    return _LSM_HR_CACHE > 0.5

@METRIC_REGISTRY.register()
def calculate_climate_mae(img, img2, crop_border, **kwargs):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img = img[..., crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]

    mask = _get_lsm_mask(img.device, img.shape[-2:])
    # broadcast mask to batch and channel dims if necessary
    # mask is expected to be (H, W). Add dims: (1, 1, H, W)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    
    mask = mask.expand_as(img)

    maes = []
    channels = img.shape[1]
    for c in range(channels):
        input = img[:, [c]][mask[:, [c]]]
        target = img2[:, [c]][mask[:, [c]]]
        if input.numel() > 0:
            mae = F.l1_loss(input, target)
        else:
            mae = torch.tensor(0.0, device=img.device)
        maes.append(mae)

    return maes

@METRIC_REGISTRY.register()
def calculate_climate_mse(img, img2, crop_border, **kwargs):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img = img[..., crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]

    mask = _get_lsm_mask(img.device, img.shape[-2:])
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    mask = mask.expand_as(img)

    mses = []
    channels = img.shape[1]
    for c in range(channels):
        input = img[:, [c]][mask[:, [c]]]
        target = img2[:, [c]][mask[:, [c]]]
        if input.numel() > 0:
            mse = F.mse_loss(input, target)
        else:
            mse = torch.tensor(0.0, device=img.device)
        mses.append(mse)

    return mses

@METRIC_REGISTRY.register()
def calculate_climate_rmse(img, img2, crop_border, **kwargs):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img = img[..., crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]

    mask = _get_lsm_mask(img.device, img.shape[-2:])
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    mask = mask.expand_as(img)

    mses = []
    if img.shape[1] == 0:
        return [torch.tensor(0, dtype=img.dtype, device=img.device)]
    channels = img.shape[1]
    for c in range(channels):
        input = img[:, [c]][mask[:, [c]]]
        target = img2[:, [c]][mask[:, [c]]]
        if input.numel() > 0:
            mse = torch.sqrt(F.mse_loss(input, target))
        else:
            mse = torch.tensor(0.0, device=img.device)
        mses.append(mse)
    return mses