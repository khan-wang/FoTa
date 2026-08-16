"""Taylor-expanded spatial interaction blocks used by FoTa-Net."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple, Iterable, Set
import math
from utils.amp import autocast
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from utils.debug import assert_finite_maybe, sanitize_maybe


# ==============================================================================
#           CMT (Continuously Masked Transformer) Related Modules
# ==============================================================================

class SequentialWithArgs(nn.Sequential):
    """
    A special nn.Sequential that passes down extra kwargs to any submodule that accepts them.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self._forward_kwarg_names = tuple(
            frozenset(module.forward.__code__.co_varnames) for module in self
        )

    def forward(self, x, **kwargs):
        kwarg_names = frozenset(kwargs)
        for module, module_params in zip(self, self._forward_kwarg_names):
            if kwarg_names.issubset(module_params):
                x = module(x, **kwargs)
            else:
                applicable_kwargs = {k: v for k, v in kwargs.items() if k in module_params}
                x = module(x, **applicable_kwargs)
        return x


def _make_soft_mask(mask: torch.Tensor, size: Tuple[int, int], mode="gaussian", k=5) -> torch.Tensor:
    h, w = size
    m = F.interpolate(mask, size=(h, w), mode="nearest")
    if mode == "gaussian":
        with autocast(device_type='cuda', enabled=False):
            m = m.float()
            if k % 2 == 0: k += 1
            r = k // 2
            x = torch.arange(-r, r + 1, device=m.device, dtype=m.dtype).view(1, 1, -1, 1)
            g = torch.exp(-0.5 * (x / (0.3 * r + 1e-6)) ** 2)
            g = g / g.sum().clamp(min=1e-6)
            m = F.conv2d(m, g, padding=(r, 0), groups=1)
            m = F.conv2d(m, g.transpose(2, 3), padding=(0, r), groups=1)
            m = m.clamp_(0.0, 1.0)
            return m.to(mask.dtype)
    else:
        return F.avg_pool2d(m, kernel_size=3, stride=1, padding=1).clamp_(0, 1)


def _warmup_alpha(global_step: int, alpha_max: float, warmup: int) -> float:
    if warmup <= 0: return float(alpha_max)
    t = min(int(global_step), int(warmup))
    return float(alpha_max * (1.0 - math.exp(-5.0 * t / (warmup + 1e-8))))


# ==============================================================================
#           Core Modules
# ==============================================================================
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super().__init__()
        self.body = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        h, w = x.shape[-2:]
        with autocast(device_type='cuda', enabled=False):
            res = self.body(to_3d(x).float())
            return to_4d(res, h, w).to(x.dtype)


class GatedFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x);
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2;
        return self.project_out(x)


class ConvolutionalPositionalEncoding(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads;
        self.head_dim = dim // num_heads
        window = {3: 2, 5: 3, 7: 3} if num_heads >= 8 else {3: 2, 5: 2} if num_heads == 4 else {3: num_heads}
        total = sum(window.values());
        scale = num_heads / total if total > num_heads else 1
        for k in list(window.keys()): window[k] = max(1, int(round(window[k] * scale)))
        diff = num_heads - sum(window.values());
        window[list(window.keys())[0]] += diff
        self.splits = [];
        self.convs = nn.ModuleList()
        for ksz, hcount in window.items():
            self.splits.append(hcount);
            self.convs.append(
                nn.Conv2d(hcount * self.head_dim, hcount * self.head_dim, kernel_size=ksz, padding=ksz // 2,
                          groups=hcount * self.head_dim, bias=True))

    def forward(self, v_bchw):
        v_groups = torch.split(v_bchw, [h * self.head_dim for h in self.splits], dim=1)
        outs = [conv(group) for conv, group in zip(self.convs, v_groups)]
        refine = torch.cat(outs, dim=1)
        return rearrange(refine, 'b (h c) H W -> b h c (H W)', h=self.num_heads, c=self.head_dim)


class TaylorExpandedAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, bias: bool = False, focusing_factor: int = 8, use_cpe: bool = True,
                 use_cmt: bool = True, cmt_stages: Iterable[int] = (1, 2, 3), cmt_alpha_max: float = 0.2,
                 cmt_warmup_steps: int = 2000, cmt_shifted: bool = True, stage_idx: int = 0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads;
        self.head_dim = dim // num_heads;
        self.focusing_factor = focusing_factor;
        self.use_cpe = use_cpe
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        if use_cpe: self.cpe = ConvolutionalPositionalEncoding(dim=dim, num_heads=num_heads)
        self.scale = nn.Parameter(torch.zeros(1, num_heads, 1, 1));
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        self.use_cmt = bool(use_cmt);
        self.cmt_stages: Set[int] = set(int(i) for i in cmt_stages);
        self.cmt_alpha_max = float(cmt_alpha_max)
        self.cmt_warmup_steps = int(cmt_warmup_steps);
        self.cmt_shifted = bool(cmt_shifted);
        self.stage_idx = int(stage_idx)
        self._cmt_cache = None

    def forward(self, x: torch.Tensor, config, stage_mask: Optional[torch.Tensor] = None, global_step: int = 0):
        assert_finite_maybe(f"TaylorAttention(Stage {self.stage_idx})/input", x, config, global_step)

        B, C, H, W = x.shape;
        N = H * W;
        eps = 1e-6


        with autocast(device_type='cuda', enabled=False):

            qkv = self.qkv_dwconv(self.qkv(x.float()))
            q, k, v = qkv.chunk(3, dim=1)

            q_rearranged = rearrange(q, 'b (h c) H W -> b h (H W) c', h=self.num_heads)
            k_rearranged = rearrange(k, 'b (h c) H W -> b h c (H W)', h=self.num_heads)
            v_rearranged = rearrange(v, 'b (h c) H W -> b h (H W) c', h=self.num_heads)

            q1 = F.normalize(q_rearranged, p=2, dim=-1, eps=eps)
            k1 = F.normalize(k_rearranged, p=2, dim=-2, eps=eps)

            p = self.focusing_factor
            q2_base = F.relu(q_rearranged).pow(p)
            q2 = F.normalize(q2_base, p=2, dim=-1, eps=eps)
            k2_base = F.relu(k_rearranged).pow(p)
            k2 = F.normalize(k2_base, p=2, dim=-2, eps=eps)

            use_cmt_now = self.use_cmt and (self.stage_idx in self.cmt_stages) and (stage_mask is not None)
            if use_cmt_now:
                soft_mask = _make_soft_mask(stage_mask.float(), (H, W), mode="gaussian", k=5)
                alpha = _warmup_alpha(global_step, self.cmt_alpha_max, self.cmt_warmup_steps)

                w_raw = (1.0 + alpha * soft_mask)
                w = w_raw / (w_raw.mean(dim=(-1, -2), keepdim=True) + eps)

                w_k_shape = rearrange(w, 'b c h w -> b c (h w)').unsqueeze(1)
                w_v_shape = w_k_shape.transpose(-1, -2)

                k1_mod, k2_mod, v_mod = k1 * w_k_shape, k2 * w_k_shape, v_rearranged * w_v_shape

                if self.cmt_shifted:
                    def _shift_pad(x):
                        x_padded = F.pad(x, (1, 0, 1, 0), mode='constant', value=0)
                        return x_padded[:, :, :-1, :-1]

                    k_shifted = _shift_pad(k)
                    v_shifted = _shift_pad(v)
                    w_shifted = _shift_pad(w)

                    k_shifted_r = rearrange(k_shifted, 'b (h c) H W -> b h c (H W)', h=self.num_heads)
                    v_shifted_r = rearrange(v_shifted, 'b (h c) H W -> b h (H W) c', h=self.num_heads)
                    w_shifted_k = rearrange(w_shifted, 'b c h w -> b c (h w)').unsqueeze(1)
                    w_shifted_v = w_shifted_k.transpose(-1, -2)

                    k1_shifted = F.normalize(k_shifted_r, p=2, dim=-2, eps=eps)
                    k2_shifted_base = F.relu(k_shifted_r).pow(p)
                    k2_shifted = F.normalize(k2_shifted_base, p=2, dim=-2, eps=eps)

                    k1 = k1_mod + k1_shifted * w_shifted_k
                    k2 = k2_mod + k2_shifted * w_shifted_k
                    v_rearranged = v_mod + v_shifted_r * w_shifted_v
                else:
                    k1, k2, v_rearranged = k1_mod, k2_mod, v_mod

                if self.training:
                    with torch.no_grad():
                        self._cmt_cache = {"alpha": alpha, "w_in": (w * soft_mask).mean().item(),
                                           "w_out": (w * (1 - soft_mask)).mean().item()}
            else:
                self._cmt_cache = None

            scale = torch.sigmoid(self.scale)
            attn1_kv = k1 @ v_rearranged;
            attn2_kv = k2 @ v_rearranged
            num1 = q1 @ attn1_kv;
            num2 = q2 @ attn2_kv
            sum_v = v_rearranged.sum(dim=-2, keepdim=True);
            numerator = sum_v + num1 + num2 * scale

            sum_k1 = k1.sum(dim=-1, keepdim=True);
            sum_k2 = k2.sum(dim=-1, keepdim=True)
            den1 = q1 @ sum_k1;
            den2 = q2 @ (sum_k2 * scale.transpose(-1, -2))
            denominator = (N + eps) + den1 + den2

            out = numerator / (denominator + eps);
            out = out * self.temperature

            if self.use_cpe:
                refine = self.cpe(v)
                out = out + (torch.sigmoid(refine) * 0.1).permute(0, 1, 3, 2)

            out = rearrange(out, 'b h (H W) c -> b (h c) H W', H=H, W=W)


            final_out = self.project_out(out)
            assert_finite_maybe(f"TaylorAttention(Stage {self.stage_idx})/output", final_out, config, global_step)

            final_out = sanitize_maybe(final_out, config)

            return final_out.to(x.dtype)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias',
                 focusing_factor=8, use_cpe=True, use_cmt=True, cmt_stages=(1, 2, 3), cmt_alpha_max=0.2,
                 cmt_warmup_steps=2000, cmt_shifted=True, stage_idx=0, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = TaylorExpandedAttention(dim, num_heads, bias, focusing_factor=focusing_factor, use_cpe=use_cpe,
                                            use_cmt=use_cmt, cmt_stages=cmt_stages, cmt_alpha_max=cmt_alpha_max,
                                            cmt_warmup_steps=cmt_warmup_steps, cmt_shifted=cmt_shifted,
                                            stage_idx=stage_idx)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = GatedFeedForward(dim, ffn_expansion_factor, bias)

    def _forward_impl(self, x, config, stage_mask: Optional[torch.Tensor] = None, global_step: int = 0):
        x = x + self.attn(self.norm1(x), config=config, stage_mask=stage_mask, global_step=global_step)
        x = x + self.ffn(self.norm2(x))
        return x

    def forward(self, x, config, stage_mask: Optional[torch.Tensor] = None, global_step: int = 0):
        if self.use_checkpoint and self.training:
            # PyTorch's checkpoint function does not support all keyword arguments.
            # We can wrap the call in a lambda to handle this.
            return grad_checkpoint(
                lambda _x, _config, _stage_mask, _global_step: self._forward_impl(_x, _config, _stage_mask,
                                                                                  _global_step),
                x, config, stage_mask, global_step, use_reentrant=False
            )
        else:
            return self._forward_impl(x, config, stage_mask, global_step)


class SKFF(nn.Module):
    def __init__(self, in_channels, height=2, reduction=8, bias=False):
        super(SKFF, self).__init__();
        self.height = height
        d = max(int(in_channels / reduction), 4);
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())
        self.fcs = nn.ModuleList(
            [nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias) for _ in range(self.height)])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size, n_feats = inp_feats[0].shape[0], inp_feats[0].shape[1]
        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        feats_U = torch.sum(inp_feats, dim=1);
        feats_S = self.avg_pool(feats_U);
        feats_Z = self.conv_du(feats_S)
        attention_vectors = [fc(feats_Z) for fc in self.fcs];
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1);

        with autocast(device_type='cuda', enabled=False):
            attention_vectors = self.softmax(attention_vectors.float()).to(inp_feats.dtype)
        return torch.sum(inp_feats * attention_vectors, dim=1)


class MultiBranchTransformerBlock(nn.Module):
    def __init__(self, dim, num_blocks, num_heads, num_path=2, ffn_expansion_factor=2.66, bias=False,
                 LayerNorm_type='WithBias', focusing_factor=8, use_cpe=True, use_cmt=True, cmt_stages=(1, 2, 3),
                 cmt_alpha_max=0.2, cmt_warmup_steps=2000, cmt_shifted=True, stage_idx=0, use_checkpoint=False):
        super().__init__()
        self.num_path = num_path
        self.paths = nn.ModuleList()
        for _ in range(num_path):
            path_blocks = nn.ModuleList()
            for _ in range(num_blocks):
                path_blocks.append(
                    TransformerBlock(dim=dim, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                     LayerNorm_type=LayerNorm_type, focusing_factor=focusing_factor, use_cpe=use_cpe,
                                     use_cmt=use_cmt, cmt_stages=cmt_stages, cmt_alpha_max=cmt_alpha_max,
                                     cmt_warmup_steps=cmt_warmup_steps, cmt_shifted=cmt_shifted, stage_idx=stage_idx,
                                     use_checkpoint=use_checkpoint)
                )
            self.paths.append(SequentialWithArgs(*path_blocks))
        self.aggregate = SKFF(dim, height=num_path) if num_path > 1 else nn.Identity()

    def forward(self, x, config, stage_mask: Optional[torch.Tensor] = None, global_step: int = 0):
        path_outputs = []
        for path in self.paths:
            path_outputs.append(path(x, config=config, stage_mask=stage_mask, global_step=global_step))

        if self.num_path == 1:
            return path_outputs[0]
        return self.aggregate(path_outputs)
