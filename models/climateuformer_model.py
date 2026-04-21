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

        # Pre-cache encoder0 LayerNorm parameter names for granular freeze/unfreeze.
        # Must be populated before any encoder0-specific freeze calls below.
        self._enc0_ln_param_names = {
            n
            for n, p in self.net_g.named_parameters()
            if n.startswith('encoderlayer_0.')
            and ('.norm' in n)
            and (n.endswith('.weight') or n.endswith('.bias'))
        }

        # Optional LayerNorm affine warmup freeze: freeze for first N iters, then unfreeze.
        # Controlled by ln_freeze_iters (default 0 = disabled).
        self._ln_freeze_iters = int(opt.get('ln_freeze_iters', 0))
        self._ln_frozen = False
        if self._ln_freeze_iters > 0:
            self._freeze_layernorm_affine()
            self._ln_frozen = True

        # Optional per-encoder0 LN freeze: keep encoderlayer_0 LayerNorm affine params
        # frozen longer than the rest of the network.  Default 0 = same as ln_freeze_iters.
        self._enc0_ln_freeze_iters = int(
            opt.get('freeze_encoder0_layernorm_warmup_iters', 0)
        )
        self._enc0_ln_frozen = False
        if self._enc0_ln_freeze_iters > 0:
            self._freeze_encoder0_layernorm_affine()
            self._enc0_ln_frozen = True

        # Optional freeze of attention relative-position bias tables during early training.
        # Relative position bias is initialised with trunc_normal(std=0.02) which is safe,
        # but its gradient can blow up in the very first few iterations before the rest of
        # the network has settled.  Freezing for N iters eliminates that hotspot.
        self._attn_bias_freeze_iters = int(
            opt.get('freeze_attention_bias_warmup_iters', 0)
        )
        self._attn_bias_frozen = False
        if self._attn_bias_freeze_iters > 0:
            self._freeze_attention_bias_tables()
            self._attn_bias_frozen = True

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

        # Pre-cache relative_position_bias_table parameters for targeted grad clipping
        # and debug logging.  Collecting by attribute name avoids false matches.
        self._rel_pos_bias_params = [
            p
            for n, p in self.net_g.named_parameters()
            if 'relative_position_bias_table' in n
        ]

        # Non-finite gradient policy.
        # 'skip'               – skip the whole step if >grad_fix_max_bad_params bad grads.
        # 'sanitize_then_step' – zero all non-finite grads and proceed if the fraction of
        #                        bad-grad params ≤ nonfinite_param_ratio_threshold; skip otherwise.
        self._nonfinite_policy = opt.get('nonfinite_policy', 'skip')
        self._nonfinite_ratio_threshold = float(
            opt.get('nonfinite_param_ratio_threshold', 0.05)
        )

        # Optional gradient clip for relative_position_bias_table params.
        # None = disabled (use global clip only); a float enables a tighter per-group clip.
        _rpb_clip = opt.get('clip_grad_relative_position_bias', None)
        self._rpb_grad_clip = float(_rpb_clip) if _rpb_clip is not None else None

        # Scheduler coupling: track whether the optimizer actually stepped this iteration.
        # Used by update_learning_rate to avoid advancing LR on skipped updates.
        # Initialize to True so the very first update_learning_rate call (iter 0 / warmup)
        # is not blocked before any optimize_parameters has run.
        self._optimizer_stepped = True

        # Track successful (non-skipped) iterations separately so the warmup LR ramp
        # is based on actual learning steps rather than total elapsed iters.  Without
        # this guard, a high skip-rate during early training causes the LR to jump to
        # base_lr at warmup_iter even when the optimiser has barely trained.
        self._successful_iters = 0

        # Skip-step and grad-zeroing observability counters.
        self._total_iters = 0
        self._skipped_iters = 0
        # Counts iterations where some parameter gradients were NaN/Inf but were
        # zeroed out individually rather than causing a full step skip.
        self._zeroed_grad_iters = 0

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

    def _freeze_encoder0_layernorm_affine(self):
        logger = get_root_logger()
        logger.info('Freezing encoderlayer_0 LayerNorm affine parameters')
        for name, param in self.net_g.named_parameters():
            if name in self._enc0_ln_param_names:
                param.requires_grad = False

    def _unfreeze_encoder0_layernorm_affine(self):
        logger = get_root_logger()
        logger.info('Unfreezing encoderlayer_0 LayerNorm affine parameters after warmup')
        for name, param in self.net_g.named_parameters():
            if name in self._enc0_ln_param_names:
                param.requires_grad = True

    def _freeze_attention_bias_tables(self):
        logger = get_root_logger()
        logger.info('Freezing relative_position_bias_table parameters')
        for name, param in self.net_g.named_parameters():
            if 'relative_position_bias_table' in name:
                param.requires_grad = False

    def _unfreeze_attention_bias_tables(self):
        logger = get_root_logger()
        logger.info('Unfreezing relative_position_bias_table parameters after warmup')
        for name, param in self.net_g.named_parameters():
            if 'relative_position_bias_table' in name:
                param.requires_grad = True

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """Override to couple scheduler stepping to successful optimizer updates only.

        Warmup guard: the warmup LR ramp is based on the number of *successful*
        (non-skipped) optimizer steps rather than on total elapsed iterations.
        This prevents the LR from jumping aggressively to base_lr after a run of
        skipped steps when current_iter reaches warmup_iter.
        """
        if self._optimizer_stepped:
            # Use effective (successful) iteration count for warmup so a high early
            # skip-rate does not cause an abrupt LR jump.  Only apply the guard while
            # still within the warmup window; once warmup is over, use current_iter to
            # allow the milestone-based scheduler to operate normally.
            effective_iter = (
                self._successful_iters
                if warmup_iter > 0 and current_iter <= warmup_iter
                else current_iter
            )
            super().update_learning_rate(effective_iter, warmup_iter=warmup_iter)
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

        # Auto-unfreeze encoderlayer_0 LN affine params (separate, longer freeze).
        if self._enc0_ln_frozen and current_iter > self._enc0_ln_freeze_iters:
            self._unfreeze_encoder0_layernorm_affine()
            self._enc0_ln_frozen = False

        # Auto-unfreeze relative_position_bias_table params after warmup.
        if self._attn_bias_frozen and current_iter > self._attn_bias_freeze_iters:
            self._unfreeze_attention_bias_tables()
            self._attn_bias_frozen = False

        def _set_skip_log():
            self.log_dict = OrderedDict()
            self.log_dict['l_pix'] = torch.tensor(0.0, device=self.device)
            self.log_dict['skipped'] = torch.tensor(1.0, device=self.device)
            self.log_dict['skip_total'] = torch.tensor(
                float(self._skipped_iters), device=self.device
            )

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

        # 1) Handle non-finite gradients.
        #
        # Two policies are supported (controlled by nonfinite_policy in config):
        #
        # 'skip' (default):
        #   If >grad_fix_max_bad_params parameters have non-finite gradients, skip the
        #   whole step.  Otherwise zero the bad grads and proceed.
        #
        # 'sanitize_then_step':
        #   Count the fraction of parameters with non-finite gradients.  If the fraction
        #   is ≤ nonfinite_param_ratio_threshold (default 5%), zero the bad grads and
        #   proceed.  If the fraction exceeds the threshold, skip the whole step.
        #   This is more permissive than 'skip' at low bad-grad counts and more
        #   principled about defining "too many bad grads" in terms of a ratio rather
        #   than a fixed head-count.
        #
        # In either case, when relative_position_bias_table gradients are non-finite,
        # a diagnostic log entry is emitted showing the table statistics.

        _bad_grad_names = []
        _bad_grad_params_list = []
        _total_grad_params = 0
        for n, p in self.net_g.named_parameters():
            if p.grad is not None:
                _total_grad_params += 1
                if not torch.isfinite(p.grad).all():
                    _bad_grad_names.append(n)
                    _bad_grad_params_list.append(p)

        # Debug logging: log relative_position_bias_table stats when non-finite grads found.
        if _bad_grad_names:
            for n, p in self.net_g.named_parameters():
                if 'relative_position_bias_table' in n and p.grad is not None:
                    tbl = p.data
                    g = p.grad
                    logger.warning(
                        f'[iter {current_iter}] relative_position_bias_table "{n}": '
                        f'param max={tbl.max().item():.4f} min={tbl.min().item():.4f} '
                        f'mean={tbl.mean().item():.4f}; '
                        f'grad finite={torch.isfinite(g).all().item()} '
                        f'max={g[torch.isfinite(g)].max().item() if torch.isfinite(g).any() else float("nan"):.4f}'
                    )

        # Decide whether to skip or sanitize based on the configured policy.
        _should_skip = False
        if _bad_grad_names:
            if self._nonfinite_policy == 'sanitize_then_step':
                bad_ratio = len(_bad_grad_names) / max(_total_grad_params, 1)
                if bad_ratio > self._nonfinite_ratio_threshold:
                    self.optimizer_g.zero_grad(set_to_none=True)
                    _skip(
                        f'{len(_bad_grad_names)}/{_total_grad_params} non-finite grad params '
                        f'({bad_ratio:.1%} > ratio threshold {self._nonfinite_ratio_threshold:.1%}), '
                        f'e.g. {_bad_grad_names[0]}'
                    )
                    _should_skip = True
            else:
                # Default 'skip' policy: use fixed count threshold.
                _max_bad = int(self.opt.get('grad_fix_max_bad_params', 3))
                if len(_bad_grad_names) > _max_bad:
                    self.optimizer_g.zero_grad(set_to_none=True)
                    _skip(
                        f'{len(_bad_grad_names)} non-finite grad params '
                        f'(>{_max_bad} threshold), e.g. {_bad_grad_names[0]}'
                    )
                    _should_skip = True

        if _should_skip:
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

        # 2c) Optional targeted clip for relative_position_bias_table params.
        # These are the first hotspot at training startup (iter 1–15 in logs showing
        # 73–150+ non-finite grad params).  A tighter clip on just these small tables
        # (typically 225 × nH floats) stops early gradient explosions without capping
        # the rest of the network.  Disabled by default (clip_grad_relative_position_bias
        # = null in config); enable by setting a positive float value.
        if self._rpb_grad_clip is not None:
            _rpb_grad_active = [
                p for p in self._rel_pos_bias_params
                if p.grad is not None and p.requires_grad
            ]
            if _rpb_grad_active:
                torch.nn.utils.clip_grad_norm_(
                    _rpb_grad_active, max_norm=self._rpb_grad_clip
                )

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
        self._successful_iters += 1

        self.log_dict = self.reduce_loss_dict(loss_dict)
        self.log_dict['skipped'] = torch.tensor(0.0, device=self.device)
        self.log_dict['skip_total'] = torch.tensor(
            float(self._skipped_iters), device=self.device
        )

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
