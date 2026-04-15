import torch
import numpy as np
import os
import math
from tqdm import tqdm
from collections import OrderedDict
import torch.nn.functional as F
from .climatesr_model import ClimateSRModel, ClimateSRAddHGTModel
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils import get_root_logger
from basicsr.metrics import calculate_metric
from Plot.pcolor_map_one import pcolor_map_one_python as pcolor_map_one
from basicsr.utils.dist_util import get_dist_info
from torch import distributed as dist

def expand2square(timg, factor=16.0):
    h, w = timg.shape[-2:]
    rsp = timg.shape[:-2]

    X = int(math.ceil(max(h,w)/float(factor))*factor)

    img = torch.zeros(*rsp,X,X).type_as(timg) # 3, h,w
    mask = torch.zeros(rsp[0],1,X,X).type_as(timg)

    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[..., ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)] = timg
    mask[..., ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)].fill_(1.0)

    return img, mask

def expand2square2(timg, factor=16.0):
    h, w = timg.shape[-2:]
    rsp = timg.shape[:-2]

    X = int(math.ceil(max(h,w)/float(factor))*factor)
    mod_pad_w = X - w
    mod_pad_h = X - h
    if timg.ndim == 5:
        img = F.pad(timg, (0, mod_pad_w, 0, mod_pad_h, 0, 0), 'reflect')
    elif timg.ndim == 4:
        img = F.pad(timg, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    mask = torch.zeros(rsp[0],1,X,X).type_as(timg)
    mask[..., :h,:w].fill_(1.0)

    return img, mask

@MODEL_REGISTRY.register()
class ClimateUformerModel(ClimateSRAddHGTModel):
    def test(self):
        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        scale = self.opt.get('scale')
        h, w = self.lq['lq'].shape[-2:]
        self.lq['lq'], mask = expand2square2(self.lq['lq'], factor=window_size * 4)
        if h == w and h % (window_size * 4) == 0:
            mask = None
        if self.lq['hgt'].shape[-1]:
            self.lq['hgt'], _ = expand2square2(self.lq['hgt'], factor=window_size * 4 * scale)
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq, mask)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq, None)
            self.net_g.train()

        self.output = self.output[..., :h*scale, :w*scale]

@MODEL_REGISTRY.register()
class ClimateUformerMultiscaleFuseModel(ClimateSRAddHGTModel):
    def __init__(self, opt):
        super().__init__(opt)
        self._freeze_layernorm_affine()

    def _freeze_layernorm_affine(self):
        for name, param in self.net_g.named_parameters():
            if '.norm' in name and (name.endswith('.weight') or name.endswith('.bias')):
                param.requires_grad = False

    def test(self):        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        scale = self.opt.get('scale')
        h, w = self.lq['lq'].shape[-2:]
        self.lq['lq'], mask = expand2square2(self.lq['lq'], factor=window_size * 4)
        if h == w and h % (window_size * 4) == 0:
            mask = None
        if self.lq['hgt'].shape[-1]:
            self.lq['hgt'], _ = expand2square2(self.lq['hgt'], factor=window_size * 4 * scale)
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq, mask)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq, mask)
            self.net_g.train()

        self.output = self.output[0][..., :h*scale, :w*scale]

    def optimize_parameters(self, current_iter):
        logger = get_root_logger()

        def _set_skip_log():
            self.log_dict = OrderedDict()
            self.log_dict['l_pix'] = torch.tensor(0.0, device=self.device)

        self.optimizer_g.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        # NaN/Inf guard on network parameters before forward pass
        for n, p in self.net_g.named_parameters():
            if p is not None and not torch.isfinite(p).all():
                raise RuntimeError(f'Non-finite param before forward: {n}')
        with torch.cuda.amp.autocast(enabled=self.amp_training):
            self.output = self.net_g(self.lq)
        self.output = [v.float() for v in self.output]

        # NaN/Inf guard on network outputs
        for idx, out_tensor in enumerate(self.output):
            if not torch.isfinite(out_tensor).all():
                logger.warning(
                    f'NaN/Inf in network output[{idx}] at iter {current_iter}, skipping update'
                )
                _set_skip_log()
                return

        # Clamp outputs to z-score range; precipitation z-scores rarely exceed ±4σ
        self.output = [v.clamp(-4.0, 4.0) for v in self.output]

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output[0], self.gt)
            if not torch.isfinite(l_pix):
                logger.warning(
                    f'NaN/Inf in l_pix at iter {current_iter}, skipping update'
                )
                _set_skip_log()
                return
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        if self.cri_pix_multiscale:
            # ClimateUformerMultiScaleHGTMultiScaleOut returns 4 outputs:
            # (out, out0, out1, out2). out0/out1/out2 are already upsampled to
            # the final spatial size before fusing, so compare all against self.gt.
            l_pix_0 = self.cri_pix_multiscale(self.output[1], self.gt)
            l_pix_1 = self.cri_pix_multiscale(self.output[2], self.gt)
            l_pix_2 = self.cri_pix_multiscale(self.output[3], self.gt)
            l_pix_multi = l_pix_0 + l_pix_1 + l_pix_2
            if not torch.isfinite(l_pix_multi):
                logger.warning(
                    f'NaN/Inf in l_pix_multi at iter {current_iter}, skipping update'
                )
                _set_skip_log()
                return
            l_total += l_pix_multi
            loss_dict['l_pix_multi'] = l_pix_multi

        if not torch.isfinite(l_total):
            logger.warning(
                f'NaN/Inf in total loss at iter {current_iter}, skipping update'
            )
            _set_skip_log()
            return

        # self.scaler.scale(l_total).backward()
        # self.scaler.unscale_(self.optimizer_g)
        # # Clip gradients to prevent explosion and NaN loss
        # torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), max_norm=0.1) # Adjust max_norm as needed based on observed gradient magnitudes. earlier "1.0"
        # self.scaler.step(self.optimizer_g)
        # self.scaler.update()
        # NaN/Inf guard on gradients before optimizer step
        self.scaler.scale(l_total).backward()
        self.scaler.unscale_(self.optimizer_g)

        # 1) Check grads are finite before clipping/step
        has_bad_grad = False
        for n, p in self.net_g.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                logger.warning(f'Non-finite grad in {n} at iter {current_iter}, skipping step')
                has_bad_grad = True
                break
        if has_bad_grad:
            self.optimizer_g.zero_grad(set_to_none=True)
            _set_skip_log()
            return

        # 2) Clip grads
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), max_norm=0.1)

        # 3) Optimizer step
        self.scaler.step(self.optimizer_g)
        self.scaler.update()

        # 4) Check params stayed finite after step
        for n, p in self.net_g.named_parameters():
            if p is not None and not torch.isfinite(p).all():
                logger.warning(f'Non-finite param after step in {n} at iter {current_iter}, skipping update')
                self.optimizer_g.zero_grad(set_to_none=True)
                _set_skip_log()
                return

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
