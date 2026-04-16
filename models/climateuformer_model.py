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
        # Propagate top-level finite_debug flag into the network arch (default off)
        finite_debug = bool(opt.get('finite_debug', False))
        if hasattr(self.net_g, 'debug_finite_checks'):
            self.net_g.debug_finite_checks = finite_debug
        # Also propagate to all Mlp/LeFF submodules so per-module debug checks work.
        for m in self.net_g.modules():
            if hasattr(m, 'debug_finite_checks'):
                m.debug_finite_checks = finite_debug

        if opt.get('freeze_layernorm', False):
            self._freeze_layernorm_affine()

        # Optional LayerNorm affine warmup freeze: freeze for first N iters, then unfreeze.
        # Controlled by ln_freeze_iters (default 0 = disabled).
        self._ln_freeze_iters = int(opt.get('ln_freeze_iters', 0))
        self._ln_frozen = False
        if self._ln_freeze_iters > 0:
            self._freeze_layernorm_affine()
            self._ln_frozen = True

        # Scheduler coupling: track whether the optimizer actually stepped this iteration.
        # Used by update_learning_rate to avoid advancing LR on skipped updates.
        # Initialize to True so the very first update_learning_rate call (iter 0 / warmup)
        # is not blocked before any optimize_parameters has run.
        self._optimizer_stepped = True

        # Skip-step observability counters.
        self._total_iters = 0
        self._skipped_iters = 0

    def _freeze_layernorm_affine(self):
        logger = get_root_logger()
        logger.info('Freezing LayerNorm affine parameters')
        for name, param in self.net_g.named_parameters():
            if '.norm' in name and (name.endswith('.weight') or name.endswith('.bias')):
                param.requires_grad = False

    def _unfreeze_layernorm_affine(self):
        logger = get_root_logger()
        logger.info('Unfreezing LayerNorm affine parameters after warmup')
        for name, param in self.net_g.named_parameters():
            if '.norm' in name and (name.endswith('.weight') or name.endswith('.bias')):
                param.requires_grad = True

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """Override to couple scheduler stepping to successful optimizer updates only."""
        if self._optimizer_stepped:
            super().update_learning_rate(current_iter, warmup_iter=warmup_iter)
        # Reset flag each time update_learning_rate is called (once per training iter).
        self._optimizer_stepped = False

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
        self._total_iters += 1

        # Auto-unfreeze LayerNorm affine params after warmup period.
        if self._ln_frozen and current_iter > self._ln_freeze_iters:
            self._unfreeze_layernorm_affine()
            self._ln_frozen = False

        def _set_skip_log():
            self.log_dict = OrderedDict()
            self.log_dict['l_pix'] = torch.tensor(0.0, device=self.device)

        def _skip(reason):
            """Record a skipped step and emit periodic skip-rate summary."""
            self._skipped_iters += 1
            skip_rate = self._skipped_iters / self._total_iters
            logger.warning(
                f'Skipping optimizer step at iter {current_iter} ({reason}). '
                f'Cumulative skip rate: {self._skipped_iters}/{self._total_iters} '
                f'({skip_rate:.1%})'
            )
            _set_skip_log()

        self.optimizer_g.zero_grad()

        # Anomaly detection is expensive; only enable when explicitly requested
        if self.opt.get('anomaly_detection', False):
            torch.autograd.set_detect_anomaly(True)

        # NaN/Inf guard on network parameters before forward pass (warn and skip, don't crash)
        for n, p in self.net_g.named_parameters():
            if p is not None and not torch.isfinite(p).all():
                _skip(f'non-finite param before forward: {n}')
                return

        # Use AMP only when fp16 is requested AND CUDA is available
        use_amp = bool(self.opt.get('fp16', False)) and torch.cuda.is_available()
        with torch.cuda.amp.autocast(enabled=use_amp):
            self.output = self.net_g(self.lq)
        self.output = [v.float() for v in self.output]

        # NaN/Inf guard on network outputs
        for idx, out_tensor in enumerate(self.output):
            if not torch.isfinite(out_tensor).all():
                _skip(f'non-finite output[{idx}]')
                return

        # Clamp outputs to z-score range; precipitation z-scores rarely exceed ±4σ
        self.output = [v.clamp(-4.0, 4.0) for v in self.output]

        l_total = torch.tensor(0.0, device=self.device)
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output[0], self.gt)
            if not torch.isfinite(l_pix):
                _skip('non-finite l_pix')
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
                _skip('non-finite l_pix_multi')
                return
            l_total += l_pix_multi
            loss_dict['l_pix_multi'] = l_pix_multi

        if not torch.isfinite(l_total):
            _skip('non-finite total loss')
            return

        # NaN/Inf guard on gradients before optimizer step
        self.scaler.scale(l_total).backward()
        self.scaler.unscale_(self.optimizer_g)

        # 1) Check grads are finite before clipping/step
        has_bad_grad = False
        bad_grad_name = None
        for n, p in self.net_g.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                has_bad_grad = True
                bad_grad_name = n
                break
        if has_bad_grad:
            self.optimizer_g.zero_grad(set_to_none=True)
            _skip(f'non-finite grad in {bad_grad_name}')
            return

        # 2) Clip grads
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), max_norm=0.1)

        # 3) Optimizer step
        self.scaler.step(self.optimizer_g)
        self.scaler.update()

        # 4) Check params stayed finite after step
        for n, p in self.net_g.named_parameters():
            if p is not None and not torch.isfinite(p).all():
                self.optimizer_g.zero_grad(set_to_none=True)
                _skip(f'non-finite param after step: {n}')
                return

        # Mark that the optimizer actually stepped so scheduler advances.
        self._optimizer_stepped = True

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
