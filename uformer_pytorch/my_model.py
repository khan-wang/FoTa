"""FoTa-Net generator and boundary-conditioned spatial--spectral fusion."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Optional

from .TaylorFormer_Block import MultiBranchTransformerBlock
from .FNO_Block import FNOBlock
from utils.amp import autocast
from utils.debug import assert_finite_maybe, sanitize_maybe

def icnr_(weight, scale=2, init=nn.init.kaiming_normal_):
    """
    ICNR (Initialization by Nearest-Neighbor Replication) a weight tensor.
    This is used to initialize a Conv2d layer before a PixelShuffle layer
    to avoid checkerboard artifacts.
    """
    out_channels, in_channels, kernel_height, kernel_width = weight.shape
    target_out_channels = out_channels // (scale ** 2)


    subkernel = torch.empty(target_out_channels, in_channels, kernel_height, kernel_width, device=weight.device)

    init(subkernel)


    # (repeat a,b,c,d) -> (a*r, b*r, c*r, d*r)
    # (repeat_interleave a, dim=x) -> only dim x is repeated
    repeated_kernel = subkernel.repeat_interleave(scale ** 2, dim=0)

    with torch.no_grad():
        weight.copy_(repeated_kernel)
# ==============================================================================
#           DSDCN Module (Re-integrated into this file)
# ==============================================================================
try:
    from mmcv.ops import DeformConv2d as MMCV_DeformConv2d

    _HAS_MMCV = True
except ImportError:
    MMCV_DeformConv2d = None
    _HAS_MMCV = False

try:
    from torchvision.ops import DeformConv2d as TV_DeformConv2d

    _HAS_TV_MODULE = True
except ImportError:
    TV_DeformConv2d = None
    _HAS_TV_MODULE = False

try:
    from torchvision.ops import deform_conv2d as tv_deform_conv2d
except ImportError:
    tv_deform_conv2d = None


class DSDCN(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, padding=None,
                 backend='auto', mode='compat', clamp_off=1.0, bias=True):
        super().__init__()
        assert k in (3, 5, 7), "DSDCN currently supports k in {3,5,7}"
        self.in_ch, self.out_ch = in_ch, out_ch
        self.k = k
        self.stride = stride
        self.pad = k // 2 if padding is None else padding
        self.clamp_off = float(clamp_off)

        if backend == 'auto':
            self.backend = 'mmcv' if _HAS_MMCV else 'torchvision'
        else:
            self.backend = backend
        if not _HAS_MMCV and self.backend == 'mmcv':
            print(f" DSDCN: MMCV backend requested but not found. Falling back to torchvision.")
            self.backend = 'torchvision'
        self.mode = mode if self.backend == 'mmcv' else 'compat'

        if self.mode == 'channelwise':
            self.offset_ch = 2 * k * k * in_ch
            self.deform_groups = in_ch
        else:
            self.offset_ch = 2 * k * k
            self.deform_groups = 1

        self.off_dw = nn.Conv2d(in_ch, in_ch, k, stride, self.pad, groups=in_ch, bias=True)
        self.off_pw = nn.Conv2d(in_ch, self.offset_ch, 1, 1, 0, bias=True)
        nn.init.zeros_(self.off_dw.weight)
        nn.init.zeros_(self.off_dw.bias)
        nn.init.zeros_(self.off_pw.weight)
        nn.init.zeros_(self.off_pw.bias)

        if self.backend == 'mmcv':
            self.dcn_dw = MMCV_DeformConv2d(in_ch, in_ch, k, stride=stride, padding=self.pad, dilation=1, groups=in_ch,
                                            deform_groups=self.deform_groups, bias=bias)
            self._backend_name = f"mmcv/{self.mode}"
        else:
            if not _HAS_TV_MODULE and tv_deform_conv2d is None:
                raise ImportError(
                    "Neither DeformConv2d module nor deform_conv2d functional found in torchvision. Please update torchvision.")
            if _HAS_TV_MODULE:
                self.dcn_dw = TV_DeformConv2d(in_ch, in_ch, k, stride=stride, padding=self.pad, dilation=1,
                                              groups=in_ch, bias=bias)
                self._backend_name = "torchvision/module"
            else:
                self.dw_weight = nn.Parameter(torch.randn(in_ch, 1, k, k) * (2.0 / (in_ch * k * k)) ** 0.5)
                self.dw_bias = nn.Parameter(torch.zeros(in_ch)) if bias else None
                self.dcn_dw = None
                self._backend_name = "torchvision/functional"

        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=bias)

    @autocast(device_type='cuda', enabled=False)
    def forward(self, x):
        x_float = x.float()
        off = torch.clamp(self.off_pw(self.off_dw(x_float)), -self.clamp_off, self.clamp_off)
        if self.backend == 'mmcv':
            y = self.dcn_dw(x_float, off)
        else:
            if _HAS_TV_MODULE:
                y = self.dcn_dw(x_float, off)
            else:
                y = tv_deform_conv2d(input=x_float, weight=self.dw_weight, offset=off, bias=self.dw_bias,
                                     stride=(self.stride, self.stride), padding=(self.pad, self.pad), dilation=(1, 1),
                                     mask=None)
        return self.pw(y)

    def extra_repr(self):
        return (f"in={self.in_ch}, out={self.out_ch}, k={self.k}, stride={self.stride}, "
                f"backend={self._backend_name}, deform_groups={self.deform_groups}")


# ==============================================================================
#           Model Blocks
# ==============================================================================

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if m.bias is not None: nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
        if m.weight is not None: nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None: nn.init.constant_(m.bias.data, 0.0)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        y = self.depthwise(x)
        # Avoid output aliasing under compiled execution.
        try:
            import torch._dynamo as _dynamo
            if _dynamo.is_compiling():
                y = y.clone()
        except Exception:
            pass
        return self.pointwise(y)


def _odd_kernel(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def _normalize_spatial_map(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    flat = x.flatten(2)
    lo = flat.amin(dim=-1, keepdim=True).unsqueeze(-1)
    hi = flat.amax(dim=-1, keepdim=True).unsqueeze(-1)
    return (x - lo) / (hi - lo + eps)


class GatedFusion(nn.Module):
    _ALLOWED_CUES = {
        "mask",
        "mask_distance",
        "texture",
        "branch_disagreement",
        "branch_norm_ratio",
        "fno_highfreq",
    }

    def __init__(
            self, in_channels, cue_names=("mask",), texture_kernel: int = 3, distance_kernel: int = 15,
            cue_calibration: bool = False, gate_zero_init: bool = False,
            gate_logit_scale: float = 1.0, gate_logit_clamp: float = 0.0,
            competitive_softmax: bool = False,
            fno_main_correction: bool = False,
            fno_main_max_taylor_weight: float = 1.0,
            fno_main_init_taylor_weight: float = 0.5):
        super().__init__()
        cue_names = tuple(cue_names or ("mask",))
        unknown = sorted(set(cue_names) - self._ALLOWED_CUES)
        if unknown:
            raise ValueError(f"Unknown reliability fusion cues: {unknown}")
        if "mask" not in cue_names:
            raise ValueError("GatedFusion requires the 'mask' cue.")

        self.cue_names = cue_names
        self.texture_kernel = _odd_kernel(texture_kernel)
        self.distance_kernel = _odd_kernel(distance_kernel)
        self.gate_logit_scale = float(gate_logit_scale)
        self.gate_logit_clamp = float(gate_logit_clamp)
        self.competitive_softmax = bool(competitive_softmax)
        self.fno_main_correction = bool(fno_main_correction)
        self.fno_main_max_taylor_weight = float(fno_main_max_taylor_weight)
        self.fno_main_init_taylor_weight = float(fno_main_init_taylor_weight)
        if self.fno_main_correction:
            if self.competitive_softmax:
                raise ValueError("fno_main_correction cannot be combined with competitive_softmax.")
            if not (0.0 < self.fno_main_max_taylor_weight <= 1.0):
                raise ValueError("fno_main_max_taylor_weight must be in (0, 1].")
            if not (0.0 < self.fno_main_init_taylor_weight < self.fno_main_max_taylor_weight):
                raise ValueError("fno_main_init_taylor_weight must be in (0, max_taylor_weight).")

        # --- [Gate Stabilization] Branch-wise normalization for gate prediction ---
        # Only used on the gate-input path (does not change the fused feature magnitudes).
        self.norm_taylor = nn.GroupNorm(8, in_channels)
        self.norm_fno = nn.GroupNorm(8, in_channels)

        cue_channels = len(self.cue_names)
        self.cue_calibration = None
        if cue_calibration:
            self.cue_calibration = nn.Conv2d(cue_channels, cue_channels, kernel_size=1, groups=cue_channels)
            nn.init.ones_(self.cue_calibration.weight)
            nn.init.zeros_(self.cue_calibration.bias)

        # Gate head consumes branch features plus lightweight reliability cues.
        final_out_channels = in_channels * 2 if self.competitive_softmax else in_channels
        final_gate_conv = nn.Conv2d(in_channels, final_out_channels, kernel_size=3, padding=1, bias=True)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2 + cue_channels, in_channels, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(8, in_channels),
            nn.GELU(),
            final_gate_conv,
        )
        if self.fno_main_correction:
            nn.init.zeros_(final_gate_conv.weight)
            init_ratio = self.fno_main_init_taylor_weight / self.fno_main_max_taylor_weight
            init_ratio = min(1.0 - 1e-4, max(1e-4, init_ratio))
            nn.init.constant_(final_gate_conv.bias, math.log(init_ratio / (1.0 - init_ratio)))
        elif gate_zero_init:
            nn.init.zeros_(final_gate_conv.weight)
            nn.init.zeros_(final_gate_conv.bias)

    def _local_texture(self, feat: torch.Tensor) -> torch.Tensor:
        k = self.texture_kernel
        smooth = F.avg_pool2d(feat.float(), kernel_size=k, stride=1, padding=k // 2)
        texture = (feat.float() - smooth).abs().mean(dim=1, keepdim=True)
        return _normalize_spatial_map(texture).to(dtype=feat.dtype)

    def _signed_mask_distance_proxy(self, mask: torch.Tensor) -> torch.Tensor:
        k = self.distance_kernel
        mask_f = mask.float()
        hole_density = F.avg_pool2d(mask_f, kernel_size=k, stride=1, padding=k // 2)
        outside_density = F.avg_pool2d(1.0 - mask_f, kernel_size=k, stride=1, padding=k // 2)
        return (hole_density - outside_density).to(dtype=mask.dtype)

    def _build_cues(self, taylor_norm, fno_norm, mask, source_feat=None):
        cue_maps = []
        for name in self.cue_names:
            if name == "mask":
                cue = mask
            elif name == "mask_distance":
                cue = self._signed_mask_distance_proxy(mask)
            elif name == "texture":
                cue = self._local_texture(source_feat if source_feat is not None else 0.5 * (taylor_norm + fno_norm))
            elif name == "branch_disagreement":
                cue = (taylor_norm - fno_norm).abs().mean(dim=1, keepdim=True)
                cue = _normalize_spatial_map(cue)
            elif name == "branch_norm_ratio":
                t_norm = torch.linalg.vector_norm(taylor_norm.float(), dim=1, keepdim=True)
                f_norm = torch.linalg.vector_norm(fno_norm.float(), dim=1, keepdim=True)
                cue = ((t_norm - f_norm) / (t_norm + f_norm + 1e-6)).clamp(-1.0, 1.0)
            elif name == "fno_highfreq":
                cue = self._local_texture(fno_norm)
            else:
                raise RuntimeError(f"Unhandled reliability fusion cue: {name}")
            cue_maps.append(cue.to(dtype=taylor_norm.dtype))
        return torch.cat(cue_maps, dim=1)

    def forward(self, taylor_feat, fno_feat, mask, source_feat=None, temp: float = 1.0):
        # Normalize only for gate prediction (scale alignment between branches).
        taylor_norm = self.norm_taylor(taylor_feat)
        fno_norm = self.norm_fno(fno_feat)
        cue_maps = self._build_cues(taylor_norm, fno_norm, mask, source_feat=source_feat)
        if self.cue_calibration is not None:
            cue_maps = self.cue_calibration(cue_maps.float()).to(dtype=taylor_norm.dtype)

        gate_logits = self.gate_conv(torch.cat([taylor_norm, fno_norm, cue_maps], dim=1))
        if self.gate_logit_scale != 1.0:
            gate_logits = gate_logits * self.gate_logit_scale
        if self.gate_logit_clamp > 0.0:
            gate_logits = gate_logits.clamp(-self.gate_logit_clamp, self.gate_logit_clamp)
        if self.competitive_softmax:
            b, _, h, w = gate_logits.shape
            branch_logits = gate_logits.view(b, 2, -1, h, w) / temp
            branch_weights = torch.softmax(branch_logits.float(), dim=1).to(dtype=taylor_feat.dtype)
            gate_map = branch_weights[:, 0]
            fno_weight = branch_weights[:, 1]
            return gate_map * taylor_feat + fno_weight * fno_feat, gate_map

        gate_map = torch.sigmoid(gate_logits / temp)
        if self.fno_main_correction:
            gate_map = gate_map * self.fno_main_max_taylor_weight
            return fno_feat + gate_map * (taylor_feat - fno_feat), gate_map

        # Fuse original features (keep representational scale).
        return gate_map * taylor_feat + (1 - gate_map) * fno_feat, gate_map



class DualBranchModule(nn.Module):
    def __init__(self, inp_channels, num_blocks, heads, fno_modes, focusing_factor, use_cpe=True,
                 use_cmt=True, cmt_stages=(1, 2, 3), cmt_alpha_max=0.2, cmt_warmup_steps=2000,
                 cmt_shifted=True, stage_idx=0, use_fno: bool = True,
                 taylor_num_paths: int = 2,
                 fno_bottleneck: float = 0.5, use_checkpoint: bool = False,
                 reliability_fusion_cues=None, reliability_texture_kernel: int = 3,
                 reliability_distance_kernel: int = 15, reliability_gate_cue_calibration: bool = False,
                 reliability_gate_zero_init: bool = False, reliability_gate_logit_scale: float = 1.0,
                 reliability_gate_logit_clamp: float = 0.0,
                 reliability_gate_competitive_softmax: bool = False,
                 fno_main_correction: bool = False,
                 fno_main_max_taylor_weight: float = 1.0,
                 fno_main_init_taylor_weight: float = 0.5):
        super().__init__()
        self.use_fno = use_fno
        self.reliability_fusion_cues = tuple(reliability_fusion_cues or ("mask",))

        self.taylor_branch = MultiBranchTransformerBlock(
            dim=inp_channels, num_blocks=num_blocks, num_heads=heads, num_path=int(taylor_num_paths),
            focusing_factor=focusing_factor, use_cpe=use_cpe,
            use_cmt=use_cmt, cmt_stages=cmt_stages, cmt_alpha_max=cmt_alpha_max,
            cmt_warmup_steps=cmt_warmup_steps, cmt_shifted=cmt_shifted, stage_idx=stage_idx,
            use_checkpoint=use_checkpoint
        )

        if self.use_fno:
            if not (0.0 < float(fno_bottleneck) <= 1.0):
                raise ValueError(
                    f"fno_bottleneck must be in the range (0, 1], got {fno_bottleneck!r}"
                )

            bottleneck_dim = max(1, math.ceil(inp_channels * float(fno_bottleneck)))
            self.fno_pre_proj = nn.Conv2d(inp_channels, bottleneck_dim, 1) if fno_bottleneck < 1.0 else nn.Identity()
            self.fno_branch = FNOBlock(in_channels=bottleneck_dim, out_channels=bottleneck_dim, modes_height=fno_modes,
                                       modes_width=fno_modes)
            self.fno_post_proj = nn.Conv2d(bottleneck_dim, inp_channels, 1) if fno_bottleneck < 1.0 else nn.Identity()
            self.fusion = GatedFusion(
                inp_channels,
                cue_names=self.reliability_fusion_cues,
                texture_kernel=reliability_texture_kernel,
                distance_kernel=reliability_distance_kernel,
                cue_calibration=reliability_gate_cue_calibration,
                gate_zero_init=reliability_gate_zero_init,
                gate_logit_scale=reliability_gate_logit_scale,
                gate_logit_clamp=reliability_gate_logit_clamp,
                competitive_softmax=reliability_gate_competitive_softmax,
                fno_main_correction=fno_main_correction,
                fno_main_max_taylor_weight=fno_main_max_taylor_weight,
                fno_main_init_taylor_weight=fno_main_init_taylor_weight,
            )
        else:
            self.fno_branch = None
            self.fusion = None

    def _create_boundary_aware_mask(self, mask_01):
        """Create a soft mask whose transition band conditions gated fusion."""

        k_size = 7
        pad = k_size // 2

        outer = torch.clamp(F.max_pool2d(mask_01, k_size, 1, pad) - mask_01, 0, 1)

        inner = torch.clamp(mask_01 - (1.0 - F.max_pool2d(1.0 - mask_01, k_size, 1, pad)), 0, 1)


        edge = torch.clamp(inner + outer, 0, 1)
        edge_blurred = F.avg_pool2d(edge, kernel_size=3, stride=1, padding=1)


        soft_mask = torch.clamp(mask_01 + 0.5 * edge_blurred, 0, 1)
        return soft_mask

    def forward(self, fea, mask_downsampled, config=None, spectral_dropout_rate=0.0, global_step=0, gate_temp=1.0):
        # Taylor branch
        taylor_out = self.taylor_branch(fea, stage_mask=mask_downsampled, global_step=global_step, config=config)

        # FNO branch (may be disabled by FNO_STAGES == [])
        if self.use_fno and self.fno_branch is not None:
            fno_fea = self.fno_pre_proj(fea)
            fno_out_bottleneck = self.fno_branch(fno_fea, spectral_dropout_rate, config=config, global_step=global_step)
            fno_out = self.fno_post_proj(fno_out_bottleneck)

            # ===========================
            # [Ablation] Fusion Mode Switch
            # ===========================
            fusion_mode = getattr(config, "FUSION_MODE", "gated") if config is not None else "gated"
            use_boundary_prior = bool(
                getattr(config, "USE_BOUNDARY_PRIOR_FOR_FUSION", True)) if config is not None else True

            gate_map = None
            force_branch_mode = getattr(config, "FORCE_BRANCH_MODE", "fused") if config is not None else "fused"
            if force_branch_mode in {"taylor", "fno", "avg"}:
                if force_branch_mode == "taylor":
                    fused = taylor_out
                elif force_branch_mode == "fno":
                    fused = fno_out.to(taylor_out.dtype)
                else:
                    fused = 0.5 * (taylor_out + fno_out.to(taylor_out.dtype))
                return fea + fused, None
            if force_branch_mode != "fused":
                raise ValueError(f"Unknown FORCE_BRANCH_MODE: {force_branch_mode}")

            if fusion_mode == "gated":
                # Legacy gated fusion path, kept for checkpoint/protocol compatibility.
                if use_boundary_prior:
                    soft_mask_for_fusion = self._create_boundary_aware_mask(mask_downsampled)
                else:
                    # Ablation: Naive mask without boundary emphasis
                    soft_mask_for_fusion = F.interpolate(mask_downsampled, size=taylor_out.shape[2:], mode='nearest')

                m = soft_mask_for_fusion.to(fea.dtype)
                fused, gate_map = self.fusion(taylor_out, fno_out.to(taylor_out.dtype), m, temp=gate_temp)

            elif fusion_mode == "reliability_gated":
                # Data-driven reliability cues replace the hard boundary-prior target.
                m = F.interpolate(mask_downsampled, size=taylor_out.shape[2:], mode='nearest').to(fea.dtype)
                fused, gate_map = self.fusion(
                    taylor_out,
                    fno_out.to(taylor_out.dtype),
                    m,
                    source_feat=fea,
                    temp=gate_temp,
                )

            elif fusion_mode == "avg":
                # Average-fusion ablation.
                fused = 0.5 * (taylor_out + fno_out.to(taylor_out.dtype))
                gate_map = None

            elif fusion_mode == "add":
                # Ablation: Additive Fusion
                fused = taylor_out + fno_out.to(taylor_out.dtype)
                gate_map = None
            else:
                raise ValueError(f"Unknown FUSION_MODE: {fusion_mode}")

            return fea + fused, gate_map

        else:
            # No FNO branch
            return fea + taylor_out, None

class DualBranchUformer(nn.Module):
    def __init__(
            self, img_channel=3, out_channel=3, embed_dim=(32, 64, 128, 256),
            num_blocks=(2, 2, 2, 2),
            encoder_blocks=None,
            heads=(1, 2, 4, 8),
            taylor_num_paths_per_stage=(2, 2, 2, 2),
            fno_stages=(3,),
            focusing_factor=6, use_cpe=True,
            fno_modes_per_stage=(12, 12, 12, 12),
            fno_channel_bottleneck: float = 0.5,
            reliability_fusion_cues=None,
            reliability_texture_kernel: int = 3,
            reliability_distance_kernel: int = 15,
            reliability_gate_cue_calibration: bool = False,
            reliability_gate_zero_init: bool = False,
            reliability_gate_logit_scale: float = 1.0,
            reliability_gate_logit_clamp: float = 0.0,
            reliability_gate_competitive_softmax: bool = False,
            fno_main_correction: bool = False,
            fno_main_max_taylor_weight: float = 1.0,
            fno_main_init_taylor_weight: float = 0.5,
            use_dsdcn=False, dsdcn_backend='auto', dsdcn_mode='compat', dsdcn_clamp=1.0,
            use_cmt: bool = True, cmt_stages: Iterable[int] = (1, 2, 3), cmt_alpha_max: float = 0.2,
            cmt_warmup_steps: int = 2000, cmt_shifted: bool = True,
            gradient_checkpointing: bool = False,
            **kwargs
    ):
        super().__init__()
        self.img_channel = img_channel
        self.out_channel = out_channel
        self.embed_dim = embed_dim
        self.depth = len(embed_dim)

        if encoder_blocks is None:
            self.encoder_blocks = tuple([1] * self.depth)
        else:
            self.encoder_blocks = tuple(encoder_blocks)

        self.fno_stages = set(fno_stages)
        self.use_dsdcn = bool(use_dsdcn)
        self.dsdcn_backend = dsdcn_backend
        self.dsdcn_mode = dsdcn_mode
        self.dsdcn_clamp = dsdcn_clamp

        # Encoder
        self.encoders = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_dim = self.img_channel + 1
        for i in range(self.depth):
            current_out_dim = self.embed_dim[i]
            encoder_stage_blocks = []
            if i == 0:
                first_op = DSDCN(in_dim, current_out_dim, k=3, stride=1, backend=self.dsdcn_backend,
                                 mode=self.dsdcn_mode,
                                 clamp_off=self.dsdcn_clamp) if self.use_dsdcn else DepthwiseSeparableConv(in_dim,
                                                                                                           current_out_dim,
                                                                                                           kernel_size=3,
                                                                                                           padding=1)
                encoder_stage_blocks.extend([first_op, nn.GroupNorm(8, current_out_dim), nn.GELU()])

            for _ in range(self.encoder_blocks[i]):
                encoder_stage_blocks.extend(
                    [DepthwiseSeparableConv(current_out_dim, current_out_dim, kernel_size=3, padding=1),
                     nn.GroupNorm(8, current_out_dim), nn.GELU()])
            self.encoders.append(nn.Sequential(*encoder_stage_blocks))
            if i < self.depth - 1:
                self.downsamples.append(nn.Sequential(
                    DepthwiseSeparableConv(self.embed_dim[i], self.embed_dim[i + 1], kernel_size=4, stride=2,
                                           padding=1), nn.GroupNorm(8, self.embed_dim[i + 1])))

        self.fno_stages = set(self.fno_stages)

        self.dual_branch_blocks = nn.ModuleDict()
        for stage_idx in range(self.depth):
            if num_blocks[stage_idx] > 0:
                c_dim, c_num_blocks, c_heads = embed_dim[stage_idx], num_blocks[stage_idx], heads[stage_idx]
                self.dual_branch_blocks[str(stage_idx)] = DualBranchModule(
                    inp_channels=c_dim, num_blocks=c_num_blocks, heads=c_heads,
                    fno_modes=fno_modes_per_stage[stage_idx], focusing_factor=focusing_factor, use_cpe=use_cpe,
                    use_cmt=use_cmt, cmt_stages=cmt_stages, cmt_alpha_max=cmt_alpha_max,
                    cmt_warmup_steps=cmt_warmup_steps, cmt_shifted=cmt_shifted, stage_idx=stage_idx,
                    use_fno=(stage_idx in self.fno_stages),
                    taylor_num_paths=int(taylor_num_paths_per_stage[stage_idx]),
                    fno_bottleneck=fno_channel_bottleneck,
                    use_checkpoint=gradient_checkpointing,
                    reliability_fusion_cues=reliability_fusion_cues,
                    reliability_texture_kernel=reliability_texture_kernel,
                    reliability_distance_kernel=reliability_distance_kernel,
                    reliability_gate_cue_calibration=reliability_gate_cue_calibration,
                    reliability_gate_zero_init=reliability_gate_zero_init,
                    reliability_gate_logit_scale=reliability_gate_logit_scale,
                    reliability_gate_logit_clamp=reliability_gate_logit_clamp,
                    reliability_gate_competitive_softmax=reliability_gate_competitive_softmax,
                    fno_main_correction=fno_main_correction,
                    fno_main_max_taylor_weight=fno_main_max_taylor_weight,
                    fno_main_init_taylor_weight=fno_main_init_taylor_weight,
                )
        print(f"Transformer blocks by stage: {list(num_blocks)}")
        print(f"FNO stages: {sorted(list(self.fno_stages))}")

        # Decoder
        # Decoder
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(self.depth - 1, 0, -1):
            if i == 1:
                self.upsamples.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(embed_dim[i], embed_dim[i - 1], kernel_size=3, padding=1, bias=False)
                ))
                print(f"Decoder stage {i}: bilinear upsampling followed by convolution")
            else:
                up_conv = nn.Conv2d(embed_dim[i], embed_dim[i - 1] * 4, kernel_size=3, padding=1, bias=False)

                self.upsamples.append(nn.Sequential(up_conv, nn.PixelShuffle(2)))

            decoder_block = [
                nn.Conv2d(embed_dim[i - 1] * 2, embed_dim[i - 1], kernel_size=3, padding=1, dilation=1),
                nn.GroupNorm(8, embed_dim[i - 1]), nn.GELU(),
                nn.Conv2d(embed_dim[i - 1], embed_dim[i - 1], kernel_size=3, padding=2, dilation=2),
                nn.GroupNorm(8, embed_dim[i - 1]), nn.GELU()
            ]
            self.decoders.append(nn.Sequential(*decoder_block))

        self.proj_out = nn.Sequential(DepthwiseSeparableConv(embed_dim[0], out_channel, kernel_size=3, padding=1),
                                      nn.Tanh())

        self.apply(weights_init)

        print("Applying ICNR initialization to PixelShuffle layers")

        for m in self.upsamples:
            layers = list(m.children())
            for i, layer in enumerate(layers):
                if isinstance(layer, nn.PixelShuffle):
                    scale = layer.upscale_factor
                    if i > 0 and isinstance(layers[i - 1], nn.Conv2d):
                        conv = layers[i - 1]
                        if conv.out_channels == conv.in_channels * (scale ** 2):
                            try:
                                icnr_(conv.weight, scale=scale)
                                # print(f"   ICNR applied to {conv} (scale={scale})")
                            except Exception as e:
                                print(f"   ICNR failed for {conv}: {e}")
                        elif conv.out_channels % (scale ** 2) == 0:
                            try:
                                icnr_(conv.weight, scale=scale)
                                # print(f"   ICNR (Soft Match) applied to {conv}")
                            except Exception as e:
                                print(f"   ICNR failed for {conv}: {e}")
    def forward(self, x, config=None, spectral_dropout_rate=0.0, global_step: int = 0, gate_temp: float = 1.0):
        masked_image = x[:, :self.img_channel]
        mask = x[:, self.img_channel:, :, :]
        skips = []
        gate_maps = {}

        mask_pyramid = [mask]
        for _ in range(self.depth - 1): mask_pyramid.append(F.max_pool2d(mask_pyramid[-1], kernel_size=2, stride=2))

        fea = x
        for i in range(self.depth):
            fea = self.encoders[i](fea)
            if str(i) in self.dual_branch_blocks:
                fea, gate_map = self.dual_branch_blocks[str(i)](
                    fea, mask_pyramid[i],
                    config=config,
                    spectral_dropout_rate=spectral_dropout_rate,
                    global_step=global_step,
                    gate_temp=gate_temp
                )
                if gate_map is not None:
                    gate_maps[f'stage_{i}'] = gate_map

            skips.append(fea)
            if i < self.depth - 1:
                fea = self.downsamples[i](fea)

        for i in range(self.depth - 1):
            fea = self.upsamples[i](fea)
            gated_skip = skips[self.depth - 2 - i] * (1.0 - mask_pyramid[self.depth - 2 - i])
            fea = torch.cat([fea, gated_skip], dim=1)
            fea = self.decoders[i](fea)

        final_residual = self.proj_out(fea)
        assert_finite_maybe(f"DualBranchUformer/final_output", final_residual, config, global_step)
        final_residual = sanitize_maybe(final_residual, config)
        return final_residual, gate_maps
