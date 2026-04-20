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

        # Pre-cache LayerNorm affine parameters (weight + bias from every nn.LayerNorm
        # module in the network).  Used by the per-LN gradient clip in optimize_parameters
        # so we avoid re-iterating all named_parameters() on every training step and
        # avoid relying on name-string matching.
        self._ln_affine_params = [
            p
            for m in self.net_g.modules()
            if isinstance(m, torch.nn.LayerNorm)
            for p in (m.weight, m.bias)
            if p is not None
        ]

        # Optional extended freeze for encoderlayer_0 LN params only.
        # encoderlayer_0 (the first encoder block) produces the most unstable gradients
        # immediately after the global LN unfreeze because its features are closest to the
        # raw input.  encoder0_ln_freeze_iters keeps these specific params frozen beyond
        # ln_freeze_iters; set to 0 (default) to use the same duration as ln_freeze_iters.
        self._encoder0_ln_freeze_iters = int(opt.get('encoder0_ln_freeze_iters', 0))
        self._encoder0_ln_frozen = False
        # Pre-cache encoderlayer_0 LN params for targeted freeze/unfreeze.
        if hasattr(self.net_g, 'encoderlayer_0'):
            self._encoder0_ln_affine_params = [
                p
                for m in self.net_g.encoderlayer_0.modules()
                if isinstance(m, torch.nn.LayerNorm)
                for p in (m.weight, m.bias)
                if p is not None
            ]
        else:
            self._encoder0_ln_affine_params = []
        # If encoder0_ln_freeze_iters extends beyond ln_freeze_iters, ensure encoder0 is
        # frozen from the start (the global freeze may or may not be active).
        if (self._encoder0_ln_affine_params
                and self._encoder0_ln_freeze_iters > self._ln_freeze_iters):
            for p in self._encoder0_ln_affine_params:
                p.requires_grad = False
            self._encoder0_ln_frozen = True

        # Scheduler coupling: track whether the optimizer actually stepped this iteration.
        # Used by update_learning_rate to avoid advancing LR on skipped updates.
        # Initialize to True so the very first update_learning_rate call (iter 0 / warmup)
        # is not blocked before any optimize_parameters has run.
        self._optimizer_stepped = True

        # Skip-step and grad-zeroing observability counters.
        self._total_iters = 0
        self._skipped_iters = 0
        # Counts iterations where some parameter gradients were NaN/Inf but were
        # zeroed out individually rather than causing a full step skip.
        self._zeroed_grad_iters = 0

    def _freeze_layernorm_affine(self):
        logger = get_root_logger()
        logger.info('Freezing LayerNorm affine parameters')
        for m in self.net_g.modules():
            if isinstance(m, torch.nn.LayerNorm):
                for param in (m.weight, m.bias):
                    if param is not None:
                        param.requires_grad = False

    def _unfreeze_layernorm_affine(self):
        logger = get_root_logger()
        logger.info('Unfreezing LayerNorm affine parameters after warmup')
        for m in self.net_g.modules():
            if isinstance(m, torch.nn.LayerNorm):
                for param in (m.weight, m.bias):
                    if param is not None:
                        param.requires_grad = True

    def _unfreeze_non_encoder0_layernorm_affine(self):
        """Unfreeze all LN affine params except those in encoderlayer_0 (extended freeze)."""
        logger = get_root_logger()
        enc0_ids = set(id(p) for p in self._encoder0_ln_affine_params)
        n_unfreeze = 0
        for m in self.net_g.modules():
            if isinstance(m, torch.nn.LayerNorm):
                for param in (m.weight, m.bias):
                    if param is not None and id(param) not in enc0_ids:
                        param.requires_grad = True
                        n_unfreeze += 1
        logger.info(
            f'Unfreezing {n_unfreeze} non-encoder0 LayerNorm affine parameters; '
            f'encoderlayer_0 LN stays frozen until iter {self._encoder0_ln_freeze_iters}'
        )

    def _unfreeze_encoder0_layernorm_affine(self):
        """Unfreeze encoderlayer_0 LN affine params after extended freeze."""
        logger = get_root_logger()
        for p in self._encoder0_ln_affine_params:
            p.requires_grad = True
        logger.info(
            f'Unfreezing {len(self._encoder0_ln_affine_params)} encoderlayer_0 '
            f'LayerNorm affine parameters (extended freeze ended)'
        )

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

        # Clamp outputs to a conservative z-score range to prevent loss explosion on
        # outlier batches.  ±6σ covers >99.99 % of a Gaussian while preserving gradient
        # flow for extreme precipitation events that can reach 5–6σ after log1p + z-score
        # normalisation.  The previous ±4σ bound cut off heavy-rain extremes and blocked
        # gradient signal for those samples.
        self.output = [v.clamp(-6.0, 6.0) for v in self.output]

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

        # 1) Handle non-finite gradients: zero out isolated NaN/Inf grads rather than
        #    discarding the full batch.  LN affine params (especially norm1.weight in
        #    encoderlayer_0) can produce NaN/Inf gradients in the first few iterations
        #    after warmup-unfreeze because the rest of the network has adapted to frozen
        #    LN statistics and the first post-unfreeze gradients are disproportionately
        #    large.  Zeroing an individual parameter's gradient is equivalent to keeping
        #    it frozen for one step; other parameters still receive valid updates.
        #    If too many parameters have bad gradients (> grad_fix_max_bad_params), the
        #    entire batch is treated as corrupt and the step is skipped as before.
        _max_bad = int(self.opt.get('grad_fix_max_bad_params', 3))
        _bad_grad_names = []
        _bad_grad_params_list = []
        for n, p in self.net_g.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                _bad_grad_names.append(n)
                _bad_grad_params_list.append(p)

        if len(_bad_grad_names) > _max_bad:
            # Too many bad parameters → full batch is corrupt; discard the step.
            self.optimizer_g.zero_grad(set_to_none=True)
            _skip(
                f'{len(_bad_grad_names)} non-finite grad params '
                f'(>{_max_bad} threshold), e.g. {_bad_grad_names[0]}'
            )
            return

        # Zero only the bad grads (below threshold); all other grads remain valid.
        for p in _bad_grad_params_list:
            p.grad.zero_()

        if _bad_grad_names:
            # Isolated NaN/Inf (typically LN affine post-unfreeze) → zero them and
            # proceed.  Record in the zero-rate counter (separate from skip-rate).
            self._zeroed_grad_iters += 1
            zero_rate = self._zeroed_grad_iters / self._total_iters
            param_summary = ", ".join(_bad_grad_names[:2])
            if len(_bad_grad_names) > 2:
                param_summary += "..."
            logger.warning(
                f'Zeroed {len(_bad_grad_names)} non-finite grad(s) at iter {current_iter} '
                f'({param_summary}). '
                f'Cumulative zero rate: {self._zeroed_grad_iters}/{self._total_iters} '
                f'({zero_rate:.1%})'
            )

        # 2) Clip grads — max_norm=1.0 is aggressive enough to prevent run-away updates
        # while allowing meaningful weight updates on successful (non-skipped) steps.
        # max_norm=0.1 was too restrictive and caused the validation metrics to remain flat.
        torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), max_norm=1.0)

        # 2b) Tighter per-group clip for LayerNorm affine params.
        # After the LN warmup-freeze unfreeze, norm1/norm2 weight gradients can spike
        # (the main network weights have adapted to the frozen LN statistics, so the
        # first few post-unfreeze gradients for LN params are disproportionately large).
        # A secondary cap of max_norm=0.1 on all LN affine params combined keeps those
        # updates small and prevents the non-finite grad recurrences seen at
        # encoderlayer_0.blocks.0.norm1.weight.  This is a no-op during the warmup
        # freeze period because requires_grad=False → p.grad is None.
        _ln_grad_params = [
            p for p in self._ln_affine_params
            if p.grad is not None and p.requires_grad
        ]
        if _ln_grad_params:
            torch.nn.utils.clip_grad_norm_(_ln_grad_params, max_norm=0.1)

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
