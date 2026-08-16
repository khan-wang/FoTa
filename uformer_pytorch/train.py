"""FoTa-Net training entry point."""

# ==============================================================================
#           MOD-00: Global Imports & Flags
# ==============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import numpy as np
import sys
import math
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import warnings
from collections import deque
import csv
import copy
from typing import Dict, Any, List, Optional
import tempfile
import hashlib
import json
import argparse
import platform
from pathlib import Path
from skimage.metrics import structural_similarity
from PIL import Image  # Added for FixedEvalDataset

try:
    import torch._dynamo as dynamo  # type: ignore[attr-defined]

    dynamo.config.recompile_limit = 64
    dynamo.config.cache_size_limit = 2048
    dynamo.config.suppress_errors = True
except Exception:
    pass

try:
    from torch_fidelity import calculate_metrics
    _FID_AVAILABLE = True
except ImportError:
    calculate_metrics = None
    _FID_AVAILABLE = False

if hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "fp32_precision"):
    torch.backends.cuda.matmul.fp32_precision = "tf32"
if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
    torch.backends.cudnn.conv.fp32_precision = "tf32"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
# --- Optional perceptual metric ---
try:
    import lpips

    _LPIPS_AVAILABLE = True
except ImportError:
    lpips = None
    _LPIPS_AVAILABLE = False


def _cg_mark_if_compiled(cfg):
    if getattr(cfg, 'torch_compile', False):
        mark = getattr(torch.compiler, 'cudagraph_mark_step_begin', None)
        if callable(mark):
            mark()


# ==============================================================================
#           MOD-01: Project-Specific Imports
# ==============================================================================
import cv2
from .losses import HRFPerceptualLoss
from torch.utils.tensorboard import SummaryWriter

from .my_model import DualBranchUformer, DSDCN
from .TaylorFormer_Block import TaylorExpandedAttention, GatedFeedForward
from .FNO_Block import FNOBlock, SpectralConv2d
from .discriminator import PatchGANDiscriminator
from .utils_viz import (
    save_visualization_grid, calculate_masked_metrics, SeamConsistencyLoss, denorm_minus1_1_to_0_1,
    sobel_magnitude,
)
from .mask_generator import ZITSStyleMask

from config import load_config
from utils.random import set_all_seeds, seed_worker
from utils.amp import autocast, GradScaler
from .scheduler_manager import SchedulerManager
from .state_controller import GANStateController
from utils.ckpt_ops import override_lr_and_reset
from utils.checkpoint_safety import filter_matching_tensors, load_state_dict_report
from utils.debug import format_sanity_state
from utils.distributed import (
    any_true,
    barrier,
    broadcast_bool,
    broadcast_object,
    cleanup_distributed,
    configure_process_output,
    init_distributed,
    max_scalar,
    mean_scalars,
    unwrap_model,
)
from preflight import preflight_check


# ==============================================================================
#           MOD-02: Checkpointing & Resume Utilities
# ==============================================================================
def _atomic_torch_save(obj, path):
    """Write a checkpoint atomically to avoid partial files."""
    d = os.path.dirname(path);
    os.makedirs(d, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=d, delete=False) as f:
        tmp = f.name
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)  # atomic on same volume
    except Exception:
        try:
            os.remove(tmp)
        finally:
            raise


def _config_digest(cfg_obj):
    """Creates a short hash from the config object for identification."""
    try:
        d = {k: v for k, v in vars(cfg_obj).items() if isinstance(v, (int, float, str, bool, list, dict, type(None)))}
        d = {k: str(v) if isinstance(v, Path) else v for k, v in d.items()}
        s = json.dumps(d, sort_keys=True, default=str).encode('utf-8')
        return hashlib.sha1(s).hexdigest()[:10]
    except Exception as e:
        print(f"Warning: Could not create config digest. Error: {e}")
        return "NA"


def _cfg_value(cfg, name, default=None):
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _stable_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_stable_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _stable_value(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    return value


def _get_arch_signature(cfg):
    """Generates a signature string from key architecture and protocol configs."""
    try:
        keys = [
            "IMG_SIZE", "IMG_CHANNEL", "OUT_CHANNEL",
            "EMBED_DIM", "NUM_BLOCKS", "ENCODER_BLOCKS", "HEADS",
            "FOCUSING_FACTOR", "USE_CPE",
            "FNO_STAGES", "FNO_MODES_PER_STAGE", "FNO_CHANNEL_BOTTLENECK",
            "FUSION_MODE", "USE_BOUNDARY_PRIOR_FOR_FUSION",
            "USE_CMT", "CMT_STAGES", "CMT_SHIFTED", "CMT_ALPHA_MAX", "CMT_WARMUP_STEPS",
            "USE_DSDCN", "DSDCN_BACKEND", "DSDCN_MODE", "DSDCN_CLAMP",
            "SPECTRAL_DROPOUT_RATE", "SPECTRAL_DROPOUT_SCHEDULE", "SPECTRAL_DROPOUT_FORCE",
            "SPECTRAL_L1_WEIGHT", "SPECTRAL_L1_WEIGHT_SCHEDULE",
            "GATE_PRIOR_WEIGHT", "GATE_PRIOR_FORCE",
            "L1_HOLE_WEIGHT", "TV_INHOLE_WEIGHT", "SEAM_LOSS_WEIGHT", "EDGE_WEIGHT",
            "FEATURE_MATCH_WEIGHT", "PERC_WEIGHT", "LPIPS_WEIGHT",
        ]
        sig_dict = {key: _stable_value(_cfg_value(cfg, key, None)) for key in keys}
        sig_dict["FNO_DROPOUT_SEMANTICS"] = "fixed_top_energy_keep_fraction_v2"
        payload = json.dumps(sig_dict, sort_keys=True, default=str, separators=(",", ":"))
        return "sigv2_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    except Exception as exc:
        print(f"Warning: Could not create arch signature. Error: {exc}")
        return "unknown"


def _checkpoint_signature_payload(cfg):
    return {
        "arch_signature": _get_arch_signature(cfg),
        "fno_dropout_semantics": "fixed_top_energy_keep_fraction_v2",
        "legacy_note": "Checkpoints without sigv2 were trained with legacy platform behavior.",
    }


class _FreezeModuleParams:
    def __init__(self, module: nn.Module):
        self.module = module
        self._states = []

    def __enter__(self):
        self._states = [(p, p.requires_grad) for p in self.module.parameters()]
        for p, _ in self._states:
            p.requires_grad_(False)
        return self.module

    def __exit__(self, exc_type, exc, tb):
        for p, requires_grad in self._states:
            p.requires_grad_(requires_grad)
        self._states = []
        return False

def build_full_ckpt(epoch, global_step, best_psnr,
                    generator, generator_ema, discriminator,
                    optimizer_g, optimizer_d, scheduler_g, scheduler_d,
                    scaler, d_stalled_count, Config,
                    last_strong_d_step, last_probe):
    """Builds a comprehensive checkpoint dictionary."""
    config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(Config).items()}

    netG_sd = unwrap_model(generator).state_dict()
    netD_sd = unwrap_model(discriminator).state_dict()
    emaG_sd = unwrap_model(generator_ema).state_dict()
    gan_ctrl_state = {
        "last_strong_d_step": last_strong_d_step,
        "last_probe": last_probe,
    }

    ckpt = {
        "format": "full",
        "version": 1.1,
        "epoch": epoch,
        "global_step": global_step,
        "best_psnr": float(best_psnr),
        "netG": netG_sd,
        "emaG": emaG_sd,
        "netD": netD_sd,
        "optG": optimizer_g.state_dict(),
        "optD": optimizer_d.state_dict(),
        "schG": scheduler_g.state_dict() if scheduler_g is not None else None,
        "schD": scheduler_d.state_dict() if scheduler_d is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "d_stalled_count": int(d_stalled_count),
        "gan_ctrl": gan_ctrl_state,
        "rng": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        },
        "config": config_dict,
        "config_digest": _config_digest(Config),
        "arch_signature": _get_arch_signature(Config),
        "signature_payload": _checkpoint_signature_payload(Config),
        "torch_version": torch.__version__,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return ckpt


def save_full_checkpoint(checkpoint_dir, filename, **kwargs):
    """Saves a full checkpoint atomically."""
    path = os.path.join(checkpoint_dir, filename)
    _atomic_torch_save(build_full_ckpt(**kwargs), path)
    return path


def save_weights_only(checkpoint_dir, filename, state_dict):
    """Saves only the model weights atomically."""
    path = os.path.join(checkpoint_dir, filename)
    _atomic_torch_save(state_dict, path)
    return path


def smart_resume(path, device,
                 generator, generator_ema, discriminator,
                 optimizer_g, optimizer_d, scheduler_g, scheduler_d, scaler, current_config):
    """
    Intelligently resumes from a full checkpoint or a weights-only file.
    Returns: A tuple (global_step, best_psnr, start_epoch, loaded_gan_ctrl_state)
    """

    def _torch_load(p, weights_only=False):
        try:
            # PyTorch >= 1.10 supports weights_only
            return torch.load(p, map_location="cpu", weights_only=weights_only)
        except TypeError:  # Fallback for older PyTorch versions
            return torch.load(p, map_location="cpu")

    if not (path and os.path.exists(path)):
        raise FileNotFoundError(path)

    weights_only_flag = str(path).endswith(".weights.pth")
    ckpt = _torch_load(path, weights_only=weights_only_flag)


    loaded_sig = ckpt.get('arch_signature', None) if isinstance(ckpt, dict) else None
    current_sig = _get_arch_signature(current_config)
    if loaded_sig is None:
        print("\n\033[93m[resume] WARNING: legacy checkpoint without expanded sigv2 architecture signature.\033[0m")
        print("  - Treat prior results as legacy behavior; do not mix old and new training results.")
    elif loaded_sig and loaded_sig != current_sig:
        print("\n\033[93m[resume]  WARNING: Architecture mismatch detected!\033[0m")
        print(f"  - Checkpoint signature: {loaded_sig}")
        print(f"  - Current model signature: {current_sig}")
        print("  - Forcing WEIGHTS-ONLY resume. Optimizers, schedulers, and EMA will be RESET.")
        weights_only_flag = True



    reset_opt = getattr(current_config, 'RESET_OPT_AND_SCHED', False)
    if reset_opt:
        print(f"[resume] RESET_OPT_AND_SCHED=True. Optimizers/Schedulers/Scaler will NOT be loaded.")

    target_generator = unwrap_model(generator)
    target_discriminator = unwrap_model(discriminator)
    target_ema = unwrap_model(generator_ema)
    critical_g_prefixes = (
        "dual_branch_blocks", "encoders", "downsamples", "upsamples", "decoders", "proj_out"
    )
    fail_on_critical = bool(getattr(current_config, "FAIL_ON_CRITICAL_CKPT_MISMATCH", False))

    if isinstance(ckpt, dict) and (
            ("format" in ckpt and ckpt["format"] == "full") or
            all(k in ckpt for k in ["netG", "netD", "optG", "optD"])
    ):
        def pick(d, *keys):
            for k in keys:
                if k in d and d[k] is not None: return d[k]
            return None

        G_sd = pick(ckpt, "netG", "G", "state_dict")
        D_sd = pick(ckpt, "netD", "D")
        Gema_sd = pick(ckpt, "emaG", "G_EMA", "state_dict")
        optG = pick(ckpt, "optG")
        optD = pick(ckpt, "optD")
        schG = pick(ckpt, "schG", "schedG")
        schD = pick(ckpt, "schD", "schedD")
        sc_sd = pick(ckpt, "scaler")

        if G_sd:
            load_state_dict_report(
                target_generator, G_sd, "G",
                strict=False,
                critical_prefixes=critical_g_prefixes,
                fail_on_critical=fail_on_critical,
            )
        if Gema_sd:
            load_state_dict_report(
                target_ema, Gema_sd, "G_EMA",
                strict=False,
                critical_prefixes=critical_g_prefixes,
                fail_on_critical=fail_on_critical,
            )
            print("[resume]  Generator EMA weights loaded from checkpoint.")
        else:
            print("[resume]  WARNING: EMA weights not found in checkpoint! EMA will start from scratch.")

        loaded_optD = False
        cur = target_discriminator.state_dict()
        keep = {}
        if D_sd:
            keep, mismatched = filter_matching_tensors(D_sd, cur)
            if mismatched:
                print(f"\n[resume]  Discriminator arch mismatch on {len(mismatched)} keys. Partial load.")
                print(f"[resume] D mismatched keys sample: {mismatched[:20]}")
            load_state_dict_report(
                target_discriminator, keep, "D",
                strict=False,
                critical_prefixes=("layers", "final_conv"),
                fail_on_critical=fail_on_critical,
            )
        else:
            print("[resume] No D weights found; using fresh initialization.")


        if not weights_only_flag and not reset_opt:

            if optG:
                try:
                    optimizer_g.load_state_dict(optG)
                    print("[resume] Optimizer_G state loaded.")
                except Exception as e:
                    print(f"[resume] Warning: Failed to load Optimizer_G: {e}")


            if optD and len(keep) == len(cur):
                try:
                    optimizer_d.load_state_dict(optD)
                    print("[resume] Optimizer_D state loaded.")
                except Exception as e:
                    print(f"[resume] Warning: Failed to load Optimizer_D: {e}")
            else:
                print("[resume] Skipped Optimizer_D (mismatch or missing).")


            if scheduler_g is not None and schG: scheduler_g.load_state_dict(schG)
            if scheduler_d is not None and schD: scheduler_d.load_state_dict(schD)
            if scaler is not None and sc_sd: scaler.load_state_dict(sc_sd)
        else:
            print("[resume] Skipped loading Optimizers/Schedulers (weights_only or reset_opt).")


        rng = ckpt.get("rng")

        if isinstance(rng, dict) and not weights_only_flag and not reset_opt:
            try:
                if rng.get("torch") is not None: torch.set_rng_state(rng["torch"].cpu())
                if rng.get("numpy") is not None: np.random.set_state(rng["numpy"])
                if rng.get("cuda") is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(rng["cuda"])
                print("[resume] RNG states restored.")
            except Exception as e:
                print(f"[resume] WARNING: RNG restore skipped: {e}")

        gan_ctrl_state = ckpt.get("gan_ctrl", {})
        if gan_ctrl_state and not weights_only_flag and not reset_opt:
            print("[resume] GAN control states restored.")




        gs = int(ckpt.get("global_step", 0))
        best = float(ckpt.get("best_psnr", 0.0))
        epoch = int(ckpt.get("epoch", 0))
        print(f"[resume] mode=full, step={gs}, epoch={epoch}, best_psnr={best:.2f}dB from {path}")
        return gs, best, epoch, gan_ctrl_state

    # Fallback for raw state_dict files
    load_state_dict_report(
        target_generator, ckpt, "G_weights_only",
        strict=False,
        critical_prefixes=critical_g_prefixes,
        fail_on_critical=fail_on_critical,
    )
    load_state_dict_report(
        target_ema, ckpt, "G_EMA_weights_only",
        strict=False,
        critical_prefixes=critical_g_prefixes,
        fail_on_critical=fail_on_critical,
    )
    print(f"[resume] mode=weights-only (G & EMA loaded), optim/sched/rng reset. from {path}")
    return 0, 0.0, 0, {}



# ==============================================================================
#           MOD-03: Complexity Analysis Utilities
# ==============================================================================
def _num_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _count_conv2d(module: nn.Conv2d, x, y):
    x = x[0]
    macs = module.in_channels * module.out_channels * module.kernel_size[0] * module.kernel_size[1] * y.shape[-2] * \
           y.shape[-1]
    macs //= module.groups
    return macs


def _count_linear(module: nn.Linear, x, y):
    return module.in_features * module.out_features


def _count_dsdcn(module: DSDCN, x: Any, y: Any) -> int:
    x = x[0];
    h, w = y.shape[-2:]
    macs_off = module.in_ch * module.in_ch * module.k * module.k * h * w // module.in_ch
    macs_off += module.in_ch * module.offset_ch * 1 * 1 * h * w
    macs_dcn = module.in_ch * module.in_ch * module.k * module.k * h * w // module.in_ch
    macs_pw = module.in_ch * module.out_ch * 1 * 1 * h * w
    return macs_off + macs_dcn + macs_pw


def _count_spectral_conv2d(module: SpectralConv2d, x: Any, y: Any) -> int:
    x = x[0];
    b, c, h, w = x.shape;
    n = h * w
    fft_cost = 2.5 * c * n * math.log2(n) if n > 0 else 0
    comp_mul_macs = module.modes_height * module.modes_width * module.in_channels * module.out_channels * 2
    ifft_cost = 2.5 * module.out_channels * n * math.log2(n) if n > 0 else 0
    return int(fft_cost + comp_mul_macs + ifft_cost)


def _count_taylor_attention(module: TaylorExpandedAttention, x: Any, y: Any) -> int:
    x_in = x[0];
    b, c, h, w = x_in.shape;
    n = h * w
    num_heads = module.num_heads;
    head_dim = module.head_dim;
    macs_conv = 0
    qkv_out = module.qkv(x_in);
    macs_conv += _count_conv2d(module.qkv, x_in, qkv_out)
    qkv_dw_out = module.qkv_dwconv(qkv_out);
    macs_conv += _count_conv2d(module.qkv_dwconv, qkv_out, qkv_dw_out)
    macs_conv += _count_conv2d(module.project_out, y, y)
    if module.use_cpe:
        v = qkv_dw_out.chunk(3, dim=1)[2]
        for conv in module.cpe.convs: macs_conv += _count_conv2d(conv, v, v)
    macs_matmul = 2 * (b * num_heads * n * (head_dim ** 2))
    return macs_conv + macs_matmul


def _count_gated_ffn(module: GatedFeedForward, x: Any, y: Any) -> int:
    x = x[0];
    macs = 0
    proj_in_out = module.project_in(x);
    macs += _count_conv2d(module.project_in, x, proj_in_out)
    dw_out = module.dwconv(proj_in_out);
    macs += _count_conv2d(module.dwconv, proj_in_out, dw_out)
    proj_out_in = dw_out.chunk(2, dim=1)[0]
    macs += _count_conv2d(module.project_out, proj_out_in, y)
    return macs


def profile_and_report_complexity(model: nn.Module, config: object, input_shape=(1, 4, 256, 256),
                                  group_rules: Dict[str, Any] = None):
    target_model = unwrap_model(model)
    device = next(target_model.parameters()).device
    dummy_input = torch.randn(*input_shape, device=device)
    target_model.eval()

    CUSTOM_HANDLERS = {TaylorExpandedAttention: _count_taylor_attention, GatedFeedForward: _count_gated_ffn,
                       SpectralConv2d: _count_spectral_conv2d, DSDCN: _count_dsdcn, }
    per_module_stats = []
    handles = []

    def add_hooks(module: nn.Module):
        is_custom = type(module) in CUSTOM_HANDLERS
        children = list(module.children())
        has_children = len(children) > 0

        if is_custom:
            def custom_hook(m, x, y):
                macs = CUSTOM_HANDLERS[type(m)](m, x, y)
                per_module_stats.append({'module_type': type(m), 'params': _num_params(m), 'macs': macs})

            handles.append(module.register_forward_hook(custom_hook))
            return

        if not has_children and isinstance(module, (nn.Conv2d, nn.Linear)):
            def basic_hook(m, x, y):
                macs = 0
                if isinstance(m, nn.Conv2d):
                    macs = _count_conv2d(m, x, y)
                elif isinstance(m, nn.Linear):
                    macs = _count_linear(m, x, y)
                per_module_stats.append({'module_type': type(m), 'params': _num_params(m), 'macs': macs})

            handles.append(module.register_forward_hook(basic_hook))
            return

        for child in children:
            add_hooks(child)

    add_hooks(target_model)
    with torch.no_grad():
        target_model(dummy_input, config=config, global_step=10000)
    for h in handles: h.remove()

    total_params = sum(p.numel() for p in target_model.parameters() if p.requires_grad)
    total_macs = sum(item['macs'] for item in per_module_stats)
    grouped_stats = {}

    def match_group(module_type_name: str) -> str:
        for group, keys in (group_rules or {}).items():
            if module_type_name in keys: return group
        return 'Other'

    for item in per_module_stats:
        group_name = match_group(item['module_type'].__name__)
        if group_name not in grouped_stats:
            grouped_stats[group_name] = {'group': group_name, 'params': 0, 'macs': 0, 'count': 0}
        grouped_stats[group_name]['macs'] += item['macs']
        grouped_stats[group_name]['count'] += 1

    transformer_owner_prefixes = []
    fno_owner_prefixes = []
    deform_owner_prefixes = []
    for mod_name, mod in target_model.named_modules():
        if isinstance(mod, (TaylorExpandedAttention, GatedFeedForward)):
            transformer_owner_prefixes.append(mod_name + ".")
        elif mod.__class__.__name__ == "FNOBlock":
            fno_owner_prefixes.append(mod_name + ".")
        elif isinstance(mod, DSDCN):
            deform_owner_prefixes.append(mod_name + ".")

    for name, module in target_model.named_modules():
        is_leaf_or_custom = (len(list(module.children())) == 0) or (module.__class__ in CUSTOM_HANDLERS)
        if not is_leaf_or_custom:
            continue

        group_name = None
        for p in transformer_owner_prefixes:
            if name.startswith(p):
                group_name = "Transformer"
                break
        if group_name is None:
            for p in fno_owner_prefixes:
                if name.startswith(p):
                    group_name = "FNO"
                    break
        if group_name is None:
            for p in deform_owner_prefixes:
                if name.startswith(p):
                    group_name = "Deformable_Conv"
                    break

        if group_name is None:
            group_name = match_group(module.__class__.__name__)

        if group_name not in grouped_stats:
            grouped_stats[group_name] = {'group': group_name, 'params': 0, 'macs': 0, 'count': 0}

        params_in_module = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        grouped_stats[group_name]['params'] += params_in_module

    by_group = sorted(grouped_stats.values(), key=lambda x: x['macs'], reverse=True)

    print(f"{'Group':<20} | {'Params (M)':>12} | {'%':>6} | {'MACs (G)':>12} | {'%':>6} | {'Count':>7}")
    print("-" * 80)
    for g in by_group:
        params_m = g['params'] / 1e6
        macs_g = g['macs'] / 1e9
        params_pct = 100.0 * g['params'] / max(1, total_params) if total_params > 0 else 0
        macs_pct = 100.0 * g['macs'] / max(1, total_macs) if total_macs > 0 else 0
        print(
            f"{g['group']:<20} | {params_m:12.3f} | {params_pct:5.1f}% | {macs_g:12.3f} | {macs_pct:5.1f}% | {g['count']:7d}")
    print("-" * 80)
    print(
        f"{'Total':<20} | {total_params / 1e6:12.3f} | {100.0:5.1f}% | {total_macs / 1e9:12.3f} | {100.0:5.1f}% | {sum(g['count'] for g in grouped_stats.values()):7d}")
    target_model.train()


# ==============================================================================
#           MOD-04: Core Utilities & Loss Functions
# ==============================================================================

def get_gan_cap(gs: int, config, roi_scale: float):
    """Dynamically gets the GAN cap from config, with a safeguard during ROI warmup."""
    base = float(getattr(config, "GAN_CAP", 0.50))
    if roi_scale < 1.0:
        return min(base, 0.50)
    return base


def get_adv_scale(step, delay_steps, warmup_steps, cap=1.0):
    if step < delay_steps:
        return 0.0
    t = (step - delay_steps) / float(max(1, warmup_steps))
    frac = max(0.0, min(1.0, t))
    return frac * cap


def get_scheduled_value(step: int, schedule: List[Dict[str, Any]], default_value: float = 0.0) -> float:
    """
    Calculates a value based on a segmented linear schedule from the config.
    """
    if not schedule or not isinstance(schedule, list):
        return float(default_value)

    for segment in schedule:
        start_step = segment.get('start_step', 0)
        end_step = segment.get('end_step', float('inf'))

        if start_step <= step < end_step:
            span = max(1, end_step - start_step)
            progress = (step - start_step) / span
            start_val = float(segment.get('start_val', default_value))
            end_val = float(segment.get('end_val', default_value))
            return start_val + (end_val - start_val) * progress

    last_segment = schedule[-1]
    return float(last_segment.get('end_val', default_value))


def set_optimizer_lr_with_warmup(optimizer, base_lr, current_step, warmup_steps):
    """Sets optimizer learning rate with a linear warmup."""
    if current_step < warmup_steps:
        lr = base_lr * (current_step + 1) / warmup_steps
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def _make_perc_mask(mask, mode="band", band_px=7):
    """Creates a mask for perceptual loss based on the specified mode."""
    if mode == "full" or mask is None:
        return torch.ones_like(mask)
    if mode == "hole":
        return mask
    k_size = max(3, int(band_px))
    if k_size % 2 == 0: k_size += 1
    pad = k_size // 2
    dilated_mask = F.max_pool2d(mask, kernel_size=k_size, stride=1, padding=pad)
    return (mask + (dilated_mask - mask)).clamp(0, 1)


class Logger:
    def __init__(self, filename="training_log.txt"):
        self.terminal = sys.stdout
        self.log_file = open(filename, "a", encoding='utf-8', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        if hasattr(self, 'log_file') and not self.log_file.closed: self.log_file.flush()

    def close(self):
        if hasattr(self, 'log_file') and not self.log_file.closed: self.log_file.close()

    def isatty(self):
        return self.terminal.isatty()


class NullSummaryWriter:
    def add_text(self, *args, **kwargs):
        return None

    def add_histogram(self, *args, **kwargs):
        return None

    def add_scalar(self, *args, **kwargs):
        return None

    def flush(self):
        return None

    def close(self):
        return None


def format_time(seconds):
    if seconds is None or seconds < 0: return "N/A"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class TVLoss(nn.Module):
    def __init__(self, device):
        super(TVLoss, self).__init__()
        self.device = device

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        if h_x <= 1 or w_x <= 1:
            return x.new_tensor(0.0)
        count_h = (h_x - 1) * w_x
        count_w = h_x * (w_x - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size


def _bbox_from_mask_tensor(mask_01: torch.Tensor, pad: int = 12) -> List[tuple]:
    if mask_01.dim() == 4:
        B, _, H, W = mask_01.shape
        masks = mask_01[:, 0]
    elif mask_01.dim() == 3:
        B, H, W = mask_01.shape
        masks = mask_01
    else:
        raise ValueError(f"Unsupported mask dimension: {mask_01.dim()}")

    bboxes = []
    for b in range(B):
        nz = torch.nonzero(masks[b] > 0.5, as_tuple=False)
        if nz.numel() == 0:
            y0, y1, x0, x1 = 0, H, 0, W
        else:
            ys, xs = nz[:, 0], nz[:, 1]
            y0 = int(max(0, ys.min().item() - pad))
            y1 = int(min(H, ys.max().item() + 1 + pad))
            x0 = int(max(0, xs.min().item() - pad))
            x1 = int(min(W, xs.max().item() + 1 + pad))
        bboxes.append((y0, y1, x0, x1))
    return bboxes


def _expand_square_clamped(y0, y1, x0, x1, H, W, ratio: float = 1.2) -> tuple:
    h = y1 - y0
    w = x1 - x0
    size = int(max(h, w) * ratio)
    cy = (y0 + y1) // 2
    cx = (x0 + x1) // 2
    y0n = max(0, cy - size // 2)
    x0n = max(0, cx - size // 2)
    y1n = min(H, y0n + size)
    x1n = min(W, x0n + size)
    y0n = max(0, y1n - size)
    x0n = max(0, x1n - size)
    return y0n, y1n, x0n, x1n


def _crop_and_resize(img_bchw: torch.Tensor, bboxes: List[tuple], out_hw: int, mode='bilinear') -> torch.Tensor:
    B, C, H, W = img_bchw.shape
    outs = []
    for b in range(B):
        y0, y1, x0, x1 = bboxes[b]
        crop = img_bchw[b:b + 1, :, y0:y1, x0:x1]
        if crop.shape[-2] == 0 or crop.shape[-1] == 0:
            crop = img_bchw[b:b + 1]
        out = F.interpolate(
            crop, size=(out_hw, out_hw), mode=mode,
            align_corners=False if 'bilinear' in mode else None
        )
        outs.append(out)
    return torch.cat(outs, dim=0)


def _prepare_roi_bboxes(masks_b1hw: torch.Tensor, pad: int = 12, ratio: float = 1.2) -> List[tuple]:
    _, _, height, width = masks_b1hw.shape
    bboxes = _bbox_from_mask_tensor(masks_b1hw, pad=pad)
    return [_expand_square_clamped(*bbox, height, width, ratio=ratio) for bbox in bboxes]


def build_roi_batch(
        images_bchw: torch.Tensor,
        masks_b1hw: torch.Tensor,
        roi_size: int,
        bboxes: Optional[List[tuple]] = None,
) -> tuple:
    B, _, H, W = images_bchw.shape
    del B, H, W
    if bboxes is None:
        bboxes = _prepare_roi_bboxes(masks_b1hw)
    imgs_roi = _crop_and_resize(images_bchw, bboxes, roi_size, mode='bilinear')
    masks_roi = _crop_and_resize(masks_b1hw, bboxes, roi_size, mode='nearest')
    return imgs_roi, masks_roi


def build_roi_cache(
        images_bchw: torch.Tensor,
        completed_bchw: torch.Tensor,
        masks_b1hw: torch.Tensor,
        roi_size: int,
        bboxes: Optional[List[tuple]] = None,
) -> tuple:
    if bboxes is None:
        bboxes = _prepare_roi_bboxes(masks_b1hw)
    real_roi = _crop_and_resize(images_bchw, bboxes, roi_size, mode='bilinear')
    fake_roi = _crop_and_resize(completed_bchw, bboxes, roi_size, mode='bilinear')
    masks_roi = _crop_and_resize(masks_b1hw, bboxes, roi_size, mode='nearest')
    return real_roi, fake_roi, masks_roi


def mask_to_logit_grid(mask_b1hw, logits_b1hw):
    if mask_b1hw.shape[-2:] == logits_b1hw.shape[-2:]:
        return (mask_b1hw > 0).float()
    _, _, h, w = logits_b1hw.shape
    m = F.adaptive_max_pool2d(mask_b1hw, (h, w))
    return (m > 0).float()


def _d_inst_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return x
    return (x + sigma * torch.randn_like(x)).clamp(-1, 1)


# ==============================================================================
#           MOD-04.5: In-Training Advanced Evaluation Utilities & Datasets
# ==============================================================================

class FixedEvalDataset(Dataset):
    """
    A lightweight dataset for evaluation that loads images and masks from two separate flists.
    """

    def __init__(self, img_flist_path, mask_flist_path, img_size):
        try:
            with open(img_flist_path, 'r', encoding='utf-8-sig') as f:
                self.img_paths = [line.strip() for line in f if line.strip()]
            with open(mask_flist_path, 'r', encoding='utf-8-sig') as f:
                self.mask_paths = [line.strip() for line in f if line.strip()]
        except FileNotFoundError as e:
            print(f" CRITICAL ERROR: Could not open flist file: {e}")
            raise

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.img_paths[idx]).convert('RGB')
            mask_path = self.mask_paths[idx % len(self.mask_paths)]
            mask = Image.open(mask_path).convert('L')

            img_tensor = self.img_transform(img)
            mask_tensor = self.mask_transform(mask)

            return img_tensor, mask_tensor
        except Exception as e:
            print(
                f"Warning: Error loading data at index {idx} (img: {self.img_paths[idx]}). Returning dummy data. Error: {e}")
            return torch.zeros(3, 256, 256), torch.zeros(1, 256, 256)


def _inner_outer_bands(mask: torch.Tensor, k: int = 33):
    """
    Computes inner and outer bands of a mask.
    """
    k = int(k)
    if k % 2 == 0: k += 1
    pad = k // 2
    dil = F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)
    ero = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=k, stride=1, padding=pad)
    outer = torch.clamp(dil - mask, 0, 1)
    inner = torch.clamp(mask - ero, 0, 1)
    H, W = mask.shape[-2:]
    inner = inner[..., :H, :W]
    outer = outer[..., :H, :W]
    return inner, outer


def compute_gate_prior_loss(gate_maps: Dict[str, torch.Tensor], masks: torch.Tensor,
                            kernel: int = 13,
                            target_bound: float = 0.9,
                            target_inter: float = 0.1
                            ) -> torch.Tensor:
    """Compute region-normalized soft targets for interior and boundary gates."""

    if not gate_maps:
        return masks.new_tensor(0.0)

    total_loss = masks.new_tensor(0.0)
    counted = 0


    mask_full = (masks > 0.5).float()
    inner_band, outer_band = _inner_outer_bands(mask_full, k=kernel)

    # Boundary (Gate -> 0.9)
    boundary_full = torch.clamp(inner_band + outer_band, 0.0, 1.0)

    # Interior (Gate -> 0.1)
    interior_full = torch.clamp(mask_full - boundary_full, 0.0, 1.0)

    # Outside (Gate -> 0.0)
    outside_full = torch.clamp(1.0 - (interior_full + boundary_full), 0.0, 1.0)

    for gate_map in gate_maps.values():
        if gate_map is None:
            continue

        if gate_map.dim() == 4 and gate_map.size(1) > 1:
            gate_scalar = gate_map.mean(dim=1, keepdim=True)
        else:
            gate_scalar = gate_map

        target_size = gate_scalar.shape[-2:]


        boundary_down = F.interpolate(boundary_full, size=target_size, mode='bilinear', align_corners=False)
        interior_down = F.interpolate(interior_full, size=target_size, mode='bilinear', align_corners=False)
        outside_down = F.interpolate(outside_full, size=target_size, mode='bilinear', align_corners=False)

        eps = 1e-6



        # Edge Term: |Gate - 0.9|
        b_sum = boundary_down.sum()
        loss_edge = (torch.abs(gate_scalar - target_bound) * boundary_down).sum() / (b_sum + eps)

        # Interior Term: |Gate - 0.1|
        i_sum = interior_down.sum()
        loss_inter = (torch.abs(gate_scalar - target_inter) * interior_down).sum() / (i_sum + eps)

        # Outside Term: |Gate - 0.0|
        o_sum = outside_down.sum()
        loss_out = (gate_scalar * outside_down).sum() / (o_sum + eps)

        stage_loss = loss_edge + loss_inter + loss_out
        total_loss = total_loss + stage_loss
        counted += 1

    if counted == 0:
        return masks.new_tensor(0.0)
    return total_loss / counted

@torch.no_grad()
def calculate_advanced_eval_metrics(completed_m11, gt_m11, mask01, config, lpips_vgg=None, lpips_alex=None):
    """
    Calculates LPIPS and Border-SSIM during training.
    """
    B, _, H, W = completed_m11.shape
    out = {}

    def _masked_mean(lpips_map):
        lmap = F.interpolate(lpips_map, size=(H, W), mode='bilinear', align_corners=False)
        num = (lmap * mask01).sum()
        den = mask01.sum().clamp_min(1e-6)
        return (num / den).item()

    with autocast(enabled=False):
        comp_flt = completed_m11.float()
        gt_flt = gt_m11.float()
        if lpips_vgg is not None:
            try:
                out['lpips_mask_vgg'] = _masked_mean(lpips_vgg(comp_flt, gt_flt))
            except Exception as e:
                print(f"Warning: LPIPS(VGG) eval failed: {e}")
        if lpips_alex is not None:
            try:
                out['lpips_mask_alex'] = _masked_mean(lpips_alex(comp_flt, gt_flt))
            except Exception as e:
                print(f"Warning: LPIPS(Alex) eval failed: {e}")

    if getattr(config, 'EVAL_BORDER_SSIM', True):
        try:
            k_border = getattr(config, "SEAM_KERNEL_SIZE", 7)
            inner, outer = _inner_outer_bands(mask01, k=k_border)
            border = (inner + outer).clamp_max(1.0)

            vals = []
            comp_01 = denorm_minus1_1_to_0_1(completed_m11).cpu().numpy().transpose(0, 2, 3, 1)
            gt_01 = denorm_minus1_1_to_0_1(gt_m11).cpu().numpy().transpose(0, 2, 3, 1)
            bord_np = border.squeeze(1).cpu().numpy() > 0.5

            for b in range(B):
                ys, xs = np.where(bord_np[b])
                if ys.size == 0:
                    vals.append(1.0)
                    continue
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                h_box, w_box = y1 - y0, x1 - x0
                win_size = min(7, h_box, w_box)
                if win_size % 2 == 0: win_size -= 1

                if win_size >= 7:
                    val = structural_similarity(gt_01[b, y0:y1, x0:x1], comp_01[b, y0:y1, x0:x1],
                                                data_range=1.0, channel_axis=-1, win_size=win_size)
                    vals.append(val)
                else:
                    vals.append(float('nan'))

            out['ssim_border'] = float(np.nanmean(vals))
        except Exception as e:
            print(f"Warning: Border-SSIM eval failed: {e}")

    return out


def get_sobel_gradients_for_loss(img_bchw):
    """Compute float32 Sobel gradients for loss evaluation."""
    return sobel_magnitude(img_bchw.float())

# A new helper function to contain the D loss logic for clarity
def calculate_discriminator_loss(pred_real, pred_fake_d, completed_images, images, masks, config, roi_scale, sigma,
                                 roi_adv_w, discriminator, roi_bboxes=None):
    # Main adversarial loss
    if getattr(config, 'WEIGHT_D_BY_MASK_MAXPOOL', False):
        m_down = mask_to_logit_grid(masks, pred_real)
        w_fake, w_real, eps = m_down, 1.0 - m_down, 1e-6
        if getattr(config, 'GAN_LOSS', 'nsgan') == 'hinge':
            loss_d_real = (F.relu(1.0 - pred_real) * w_real).sum() / (w_real.sum() + eps)
            loss_d_fake = (F.relu(1.0 + pred_fake_d) * w_fake).sum() / (w_fake.sum() + eps)
        else:
            loss_d_real = (F.softplus(-pred_real) * w_real).sum() / (w_real.sum() + eps)
            loss_d_fake = (F.softplus(pred_fake_d) * w_fake).sum() / (w_fake.sum() + eps)
    else:
        if getattr(config, 'GAN_LOSS', 'nsgan') == 'hinge':
            loss_d_real, loss_d_fake = F.relu(1.0 - pred_real).mean(), F.relu(1.0 + pred_fake_d).mean()
        else:
            loss_d_real, loss_d_fake = F.softplus(-pred_real).mean(), F.softplus(pred_fake_d).mean()

    # ROI adversarial loss
    d_loss_roi = torch.tensor(0.0, device=images.device)
    if config.USE_ROI_GAN and roi_scale > 0:
        real_roi_imgs, fake_roi_imgs, roi_masks = build_roi_cache(
            images,
            completed_images.detach(),
            masks,
            config.D_ROI_SIZE,
            bboxes=roi_bboxes,
        )
        d_in_fake_roi = torch.cat([_d_inst_noise(fake_roi_imgs, sigma), roi_masks],
                                  dim=1) if config.USE_MASK_IN_D else _d_inst_noise(fake_roi_imgs, sigma)
        d_in_real_roi = torch.cat([_d_inst_noise(real_roi_imgs, sigma), roi_masks],
                                  dim=1) if config.USE_MASK_IN_D else _d_inst_noise(real_roi_imgs, sigma)

        _cg_mark_if_compiled(config)
        pred_fake_roi, _ = discriminator(d_in_fake_roi)
        if getattr(config, 'torch_compile', False):
            pred_fake_roi = pred_fake_roi.clone()

        _cg_mark_if_compiled(config)
        pred_real_roi, _ = discriminator(d_in_real_roi)
        if getattr(config, 'torch_compile', False):
            pred_real_roi = pred_real_roi.clone()

        if getattr(config, 'WEIGHT_D_BY_MASK_MAXPOOL', False):
            m_down_roi = mask_to_logit_grid(roi_masks, pred_real_roi)
            w_fake_roi, w_real_roi, eps = m_down_roi, 1.0 - m_down_roi, 1e-6
            if getattr(config, 'GAN_LOSS', 'nsgan') == 'hinge':
                loss_d_real_roi = (F.relu(1.0 - pred_real_roi) * w_real_roi).sum() / (w_real_roi.sum() + eps)
                loss_d_fake_roi = (F.relu(1.0 + pred_fake_roi) * w_fake_roi).sum() / (w_fake_roi.sum() + eps)
            else:
                loss_d_real_roi = (F.softplus(-pred_real_roi) * w_real_roi).sum() / (w_real_roi.sum() + eps)
                loss_d_fake_roi = (F.softplus(pred_fake_roi) * w_fake_roi).sum() / (w_fake_roi.sum() + eps)
        else:
            if getattr(config, 'GAN_LOSS', 'nsgan') == 'hinge':
                loss_d_real_roi, loss_d_fake_roi = F.relu(1.0 - pred_real_roi).mean(), F.relu(
                    1.0 + pred_fake_roi).mean()
            else:
                loss_d_real_roi, loss_d_fake_roi = F.softplus(-pred_real_roi).mean(), F.softplus(pred_fake_roi).mean()

        d_loss_roi = (loss_d_real_roi + loss_d_fake_roi) * roi_adv_w

    return loss_d_real, loss_d_fake, d_loss_roi


# ==============================================================================
#           MOD-05: Main Execution Block
# ==============================================================================
def main():
    # ==========================================================================
    # Configure the CUDA allocator before model construction.
    import os
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    parser = argparse.ArgumentParser(description="FoTa-Net image inpainting trainer")
    parser.add_argument('--run_mode', type=str, default='public',
                        help="Optional config/env preset (default: public)")
    parser.add_argument('--config', type=str, default=None,
                        help="Custom YAML path(s), comma-separated and applied left to right.")
    parser.add_argument('--mode', type=str, default=None,
                        choices=['hot_resume_latest', 'finetune_from_best', 'cold_from_ema_weights'],
                        help="Predefined run modes to avoid inconsistent boolean combinations.")
    parser.add_argument('--results_dir', type=str, default=None,
                        help="Override for the RESULTS_BASE_DIR specified in the config.")
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help="Explicit full checkpoint path for a hot resume.")
    parser.add_argument('--max_steps', type=int, default=None,
                        help="Override MAX_STEPS_PER_RUN for smoke and recovery runs.")
    parser.add_argument('--num_epochs', type=int, default=None,
                        help="Override NUM_EPOCHS for resume smoke tests.")
    parser.add_argument('--seed', type=int, default=None,
                        help="Override GLOBAL_SEED for repeated-seed experiments.")
    parser.add_argument('--save_latest_at_end', action='store_true',
                        help="Force an epoch-end latest.pth checkpoint.")
    args = parser.parse_args()

    dist_ctx = init_distributed()
    configure_process_output(dist_ctx.is_main)

    if _FID_AVAILABLE:
        print(" torch-fidelity library found. In-training FID calculation is available.")
    else:
        print(" WARNING: torch-fidelity library not found. In-training FID will be skipped.")
        print("           To enable it, run: pip install torch-fidelity")

    config = load_config(run_mode=args.run_mode, config_path=args.config)
    hrf_root = getattr(config, 'HRF_LOSS_WEIGHTS_PATH', None)
    hrf_file = (
        Path(hrf_root) / 'ade20k' / 'ade20k-resnet50dilated-ppm_deepsup' / 'encoder_epoch_20.pth'
        if hrf_root else None
    )
    if hrf_file is None or not hrf_file.is_file():
        raise FileNotFoundError(
            "Training requires the ADE20K ResNet-50 perceptual encoder. "
            "Set HRF_LOSS_WEIGHTS_PATH to the directory described in DATASETS.md."
        )
    config.DISTRIBUTED = bool(dist_ctx.launched)
    config.RANK = int(dist_ctx.rank)
    config.LOCAL_RANK = int(dist_ctx.local_rank)
    config.WORLD_SIZE = int(dist_ctx.world_size)
    config.DEVICE = str(dist_ctx.device)
    if args.resume_checkpoint:
        config.RESUME = True
        config.RESUME_AUTO = False
        config.RESUME_CHECKPOINT = Path(args.resume_checkpoint)
    if args.max_steps is not None:
        config.MAX_STEPS_PER_RUN = int(args.max_steps)
    if args.num_epochs is not None:
        config.NUM_EPOCHS = int(args.num_epochs)
    if args.seed is not None:
        config.GLOBAL_SEED = int(args.seed)
    if args.save_latest_at_end:
        config.SAVE_LATEST_AT_EPOCH_END = True
    if dist_ctx.launched and getattr(config, "torch_compile", False):
        print("[DDP] torch.compile is disabled for the first distributed protocol.")
        config.torch_compile = False
        config.compile_g = False
        config.compile_d = False

    grad_accum_requested = int(getattr(config, "GRADIENT_ACCUM_STEPS", 1))
    if dist_ctx.launched and grad_accum_requested != 1:
        raise ValueError("The first DDP protocol requires GRADIENT_ACCUM_STEPS=1.")
    if dist_ctx.launched and int(getattr(config, "FREEZE_ENCODER_STEPS", 0)) != 0:
        raise ValueError("The first DDP protocol requires FREEZE_ENCODER_STEPS=0.")

    exp_tag = getattr(config, "EXP_TAG", "")
    tag_suffix = f"__{exp_tag}" if exp_tag else ""

    force_spec_drop = getattr(config, "SPECTRAL_DROPOUT_FORCE", None)
    force_gate_prior = getattr(config, "GATE_PRIOR_FORCE", None)
    if getattr(config, 'torch_compile', False):
        print("[Info] torch.compile=True -> runtime NaN guard is disabled.")

    def _rewrite_to_sibling(path_like, basename: str) -> Path:
        p = Path(path_like) if path_like is not None else None
        if p is None:
            return Path(basename)
        p = Path(p)
        if p.is_dir():
            return p / basename
        return p if p.name == basename else p.with_name(basename)

    def _apply_mode(mode_name: str):
        if not mode_name:
            return
        mode_name = str(mode_name)
        config.MODE = mode_name
        ckpt_path = getattr(config, 'RESUME_CHECKPOINT', None)
        if mode_name == 'hot_resume_latest':
            config.RESUME = True
            config.FINETUNE_MODE = False
            config.SCHEDULE_ABSOLUTE = True
            config.RESET_OPT_AND_SCHED = False
            config.OVERRIDE_LR_ON_RESUME = True
            if ckpt_path:
                config.RESUME_CHECKPOINT = _rewrite_to_sibling(ckpt_path, 'latest.pth')
        elif mode_name == 'finetune_from_best':
            config.RESUME = True
            config.FINETUNE_MODE = True
            config.RESET_OPT_AND_SCHED = True
            config.OVERRIDE_LR_ON_RESUME = True
            if ckpt_path:
                cands = [
                    _rewrite_to_sibling(ckpt_path, 'best_generator_ema.pth'),
                    _rewrite_to_sibling(ckpt_path, 'best_ema.pth'),
                    _rewrite_to_sibling(ckpt_path, 'weights.pth'),
                ]
                for cand in cands:
                    if cand.exists():
                        config.RESUME_CHECKPOINT = cand
                        break
                else:
                    config.RESUME_CHECKPOINT = cands[0]
        elif mode_name == 'cold_from_ema_weights':
            config.RESUME = False
            config.FINETUNE_MODE = False
            config.RESET_OPT_AND_SCHED = True
        else:
            print(f"[Mode] Unknown mode '{mode_name}', skipping preset application.")
            return
        print(f"[Mode] {config.MODE} => RESUME={config.RESUME} FINETUNE_MODE={config.FINETUNE_MODE} RESET_OPT={config.RESET_OPT_AND_SCHED} ABS={getattr(config, 'SCHEDULE_ABSOLUTE', True)}")

    selected_mode = args.mode or getattr(config, 'MODE', None)
    _apply_mode(selected_mode)

    sched_mgr = SchedulerManager(config)
    fsm = GANStateController(config)

    def _apply_lr_from_config(optim, lr):
        if optim is None: return
        for pg in optim.param_groups:
            pg['lr'] = float(lr)
            if 'initial_lr' in pg:
                pg['initial_lr'] = float(lr)

    def _sync_scheduler_base_lrs(scheduler, optim):
        if scheduler is None: return
        try:
            # For CosineAnnealingLR and others that use base_lrs
            scheduler.base_lrs = [pg.get('initial_lr', pg['lr']) for pg in optim.param_groups]
        except Exception:
            pass  # Ignore if scheduler doesn't have base_lrs

    bank = getattr(config, "MASK_BANK", None)
    assert bank and all(k in bank for k in ["IRR_ROOT", "COCO_ROOT", "TRAIN_SOURCES", "BINS", "MIX_RATIO"]), \
        f"MASK_BANK incomplete: got {bank}"

    if args.results_dir:
        config.RESULTS_BASE_DIR = Path(args.results_dir)

    if platform.system() == "Windows" or config.DATALOADER_START_METHOD == 'spawn':
        if torch.multiprocessing.get_start_method(allow_none=True) != 'spawn':
            torch.multiprocessing.set_start_method("spawn", force=True)
            print("INFO: DataLoader multiprocessing start method set to 'spawn'.")

    original_stdout = sys.stdout
    logger = None
    csv_file = None
    try:
        print(f"\n======== INPAINTING TRAINING START (RUN MODE: {config.RUN_MODE}) ========")

        set_all_seeds(config.GLOBAL_SEED)

        cv2.setNumThreads(0)
        os.environ["OMP_NUM_THREADS"] = "1"
        torch.set_num_threads(1)

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

        if config.DETECT_ANOMALY:
            print("Anomalies detection enabled for debugging. This will slow down training.")
            torch.autograd.set_detect_anomaly(True)

        device = torch.device(config.DEVICE)

        lpips_vgg_eval, lpips_alex_eval = None, None
        if dist_ctx.is_main and _LPIPS_AVAILABLE:
            if getattr(config, 'EVAL_LPIPS_VGG', True):
                try:
                    lpips_vgg_eval = lpips.LPIPS(net='vgg', spatial=True).to(device).eval()
                    print(" Initialized LPIPS(VGG) for in-training evaluation.")
                except Exception as e:
                    print(f" Failed to initialize LPIPS(VGG): {e}")
            if getattr(config, 'EVAL_LPIPS_ALEX', True):
                try:
                    lpips_alex_eval = lpips.LPIPS(net='alex', spatial=True).to(device).eval()
                    print(" Initialized LPIPS(Alex) for in-training evaluation.")
                except Exception as e:
                    print(f" Failed to initialize LPIPS(Alex): {e}")

        from datetime import datetime
        run_stamp = None
        if dist_ctx.is_main:
            run_stamp = (
                time.strftime("%Y%m%d_%H%M") + tag_suffix,
                datetime.now().strftime("%Y%m%d-%H%M%S"),
            )
        run_name, start_time_str = broadcast_object(run_stamp, src=0, device=device)
        run_dir = config.RESULTS_BASE_DIR / run_name
        vis_dir, checkpoint_dir = run_dir / "visualizations", run_dir / "checkpoints"
        if dist_ctx.is_main:
            vis_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        barrier()


        if dist_ctx.is_main:
            logger = Logger(run_dir / f"training_log_{start_time_str}.txt")
            sys.stdout = logger
            writer = SummaryWriter(log_dir=run_dir / "tensorboard_logs")
        else:
            writer = NullSummaryWriter()

        csv_path = run_dir / f"scalars_{start_time_str}.csv"
        csv_fieldnames = [
            "global_step", "local_step", "epoch", "g_total", "d_total", "l1_valid", "l1_hole", "l1_hole_w",
            "hrf_perc", "loss_raw/perc", "loss_raw/fm", "loss_raw/seam", "loss_raw/gate_prior","gate_prior_w",
            "coverage/perc_mask", "coverage/seam_band", "roi/enabled",
            "seam_loss", "inner_l1", "grad_l1", "gate_prior_loss", "tv_inhole",
            "gan_g", "gan_g_roi", "d_roi", "feat_match", "r1_gp", "r1_ema",
            "r1_last_step", "r1_every_effective", "r1_mask_ratio", "flag_r1_next", "d_gap",
            "pct_real_gt1", "pct_fake_lt_neg1",
            "psnr_eval_full", "ssim_eval_full", "psnr_eval_mask", "ssim_eval_mask", "mask_ratio",
            "mask_ratio_max", "pct_big_hole",
            "lpips_eval_mask", "lpips_mask_vgg", "lpips_mask_alex", "ssim_border",
            "fid_eval_full",
            "lr_g", "lr_d", "grad_norm_g", "grad_norm_d",
            "d_real_mean", "d_fake_mean", "d_rf_strength", "d_real_hole", "d_fake_hole",
            "adversarial_weight", "adv_scale", "gan_scale", "gan_cap_base", "gan_cap_now", "tv_inhole_weight_now",
            "eff_adv_w", "eff_roi_adv_w", "cooldown_active", "stall_count", "d_update_every",
            "roi_scale", "roi_adv_w", "gate_temp", "spec_rate", "d_stalled_count", "inst_noise_sigma",
            "latency_p50", "latency_p95", "t_g", "t_d", "flag_d", "flag_r1", "time_step",
            "use_mask_in_d", "use_roi_gan",
            "ctx_phase", "ctx_phase_start", "ctx_fsm_state",
            "prob_real", "prob_fake", "fft_mag_error", "scaler_scale", "cuda_mem_alloc_mb",
            "cuda_mem_reserved_mb", "throughput_imgs_per_s"
        ]
        # Gate diagnostics.

        if hasattr(config, 'FNO_STAGES'):
            for stage_idx in config.FNO_STAGES:
                k = str(stage_idx)
                csv_fieldnames.extend([
                    f"gate/{k}_all", f"gate/{k}_std",
                    f"gate/{k}_boundary", f"gate/{k}_interior",
                    f"prior/{k}_boundary_area", f"prior/{k}_interior_area",
                    f"gate/{k}_hole", f"gate/{k}_hole_std"
                ])
        csv_writer = None
        if dist_ctx.is_main:
            csv_file = open(csv_path, "w", newline="", encoding='utf-8')
            csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames, extrasaction='ignore')
            csv_writer.writeheader()
            print(f" Live CSV logging enabled. Output will be saved to: {csv_path}")

        print(f"Starting run: {run_name}\nResults will be saved in: {run_dir}\nUsing device: {device}")
        print(format_sanity_state(config))
        print("[Config] update_d_every is interpreted as D repeat count per G step, not as a D update interval.")
        print("[Config] FNO spectral dropout semantics: fixed_top_energy_keep_fraction_v2.")
        print("\n" + "=" * 80 + "\n" + " " * 27 + "TRAINING CONFIGURATION" + "\n" + "=" * 80)
        config_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(config).items()}
        for key, value in config_dict.items(): print(f"{str(key):<30}: {str(value)}")
        print("=" * 80)


        meta = {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "start_time": start_time_str,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "config_digest": _config_digest(config),
            "arch_signature": _get_arch_signature(config),
            "signature_payload": _checkpoint_signature_payload(config),
            "config": config_dict,
        }
        meta_path = run_dir / f"run_meta_{start_time_str}.json"
        if dist_ctx.is_main:
            with open(meta_path, "w", encoding="utf-8") as f_meta:
                json.dump(meta, f_meta, indent=2, ensure_ascii=False)
        print(f" Run meta saved to: {meta_path}")

        mask_generator_instance = ZITSStyleMask(config, device='cpu')
        subset_generator = torch.Generator()
        subset_generator.manual_seed(config.GLOBAL_SEED)
        loader_generator = torch.Generator()
        loader_generator.manual_seed(config.GLOBAL_SEED + dist_ctx.rank)

        global_batch_size = int(config.BATCH_SIZE)
        if global_batch_size % dist_ctx.world_size != 0:
            raise ValueError(
                f"Global BATCH_SIZE={global_batch_size} must be divisible by "
                f"WORLD_SIZE={dist_ctx.world_size}."
            )
        local_batch_size = global_batch_size // dist_ctx.world_size
        if local_batch_size < 1:
            raise ValueError("Per-rank batch size must be at least one.")
        config.LOCAL_BATCH_SIZE = local_batch_size
        print(
            f"[DDP] launched={dist_ctx.launched} rank={dist_ctx.rank}/{dist_ctx.world_size} "
            f"global_batch={global_batch_size} local_batch={local_batch_size}"
        )

        transform_list = [transforms.Resize((config.IMG_SIZE, config.IMG_SIZE))]
        random_flip_p = getattr(config, 'RANDOM_FLIP_P', 0.0)
        if random_flip_p > 0:
            print(f"Data Augmentation: Applying Random Horizontal Flip with p={random_flip_p}")
            transform_list.append(transforms.RandomHorizontalFlip(p=random_flip_p))
        transform_list.append(transforms.ToTensor())
        transform = transforms.Compose(transform_list)

        try:
            full_dataset = datasets.ImageFolder(root=config.RAW_DATA_ROOT, transform=transform)
            subset_size = max(1, int(len(full_dataset) * config.DATASET_FRACTION))
            indices = torch.randperm(len(full_dataset), generator=subset_generator)[:subset_size]
            train_dataset = torch.utils.data.Subset(full_dataset, indices)
            print(
                f"\nUsing a subset of {len(train_dataset)} images ({config.DATASET_FRACTION * 100:.1f}%) for training.")


            persistent_workers = bool(getattr(config, 'PERSISTENT_WORKERS', True)) if config.NUM_WORKERS > 0 else False
            prefetch_factor = int(getattr(config, 'PREFETCH_FACTOR', 2)) if config.NUM_WORKERS > 0 else None


            train_sampler = None
            if dist_ctx.world_size > 1:
                train_sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=dist_ctx.world_size,
                    rank=dist_ctx.rank,
                    shuffle=True,
                    seed=int(config.GLOBAL_SEED),
                    drop_last=True,
                )

            train_loader = DataLoader(dataset=train_dataset, batch_size=local_batch_size,
                                      shuffle=(train_sampler is None), sampler=train_sampler,
                                      num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
                                      persistent_workers=persistent_workers,
                                      prefetch_factor=prefetch_factor,
                                      worker_init_fn=seed_worker, generator=loader_generator, drop_last=True)
            if len(train_loader) == 0:
                raise ValueError(
                    f"Dataset subset ({len(train_dataset)}) is too small for global batch "
                    f"{global_batch_size} across {dist_ctx.world_size} ranks."
                )

        except FileNotFoundError:
            print(f" CRITICAL ERROR: Dataset not found at path '{config.RAW_DATA_ROOT}'. Please check YAML config.")
            sys.exit(1)

        cls_seen, vis_indices = set(), []
        if not hasattr(full_dataset, 'class_to_idx') or not full_dataset.class_to_idx:
            try:
                root_to_scan = Path(full_dataset.root)
                classes = sorted([d.name for d in os.scandir(root_to_scan) if d.is_dir()])
                full_dataset.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
            except Exception as e:
                print(f"Warning: Could not automatically find classes: {e}")

        if hasattr(full_dataset, 'samples'):
            for idx in indices.tolist():
                _, target = full_dataset.samples[idx]
                if target not in cls_seen:
                    cls_seen.add(target)
                    vis_indices.append(idx)
                if len(vis_indices) >= config.SAVE_VIZ_PER_EPOCH: break

        if not vis_indices and len(indices) > 0:
            print(" Warning: Could not find diverse classes for visualization set. Using first K images instead.")
            vis_indices = indices.tolist()[:config.SAVE_VIZ_PER_EPOCH]

        if not vis_indices and not getattr(config.EVAL, 'USE_FIXED_VALSET', False):
            print(" CRITICAL ERROR: Dataset appears empty. Cannot create visualization set.")
            vis_images_fixed = torch.randn(1, config.IMG_CHANNEL, config.IMG_SIZE, config.IMG_SIZE).to(device)
        else:
            vis_subset = torch.utils.data.Subset(full_dataset, vis_indices)
            vis_loader = DataLoader(vis_subset, batch_size=len(vis_indices) or 1, shuffle=False, num_workers=0)
            vis_images_fixed = next(iter(vis_loader))[0].to(device, non_blocking=True) * 2. - 1
            print(f" Created a diverse fixed visualization batch with {vis_images_fixed.shape[0]} images.")

        VIZ_MASK_SEED = getattr(config, "VIZ_MASK_SEED", config.GLOBAL_SEED)
        vis_masks_fixed, _ = mask_generator_instance(vis_images_fixed.shape[0], seed=VIZ_MASK_SEED)
        vis_masks_fixed = vis_masks_fixed.to(device)
        print(f" Created a fixed visualization mask batch with seed {VIZ_MASK_SEED}.")

        fixed_eval_loader = None
        fixed_eval_desc = None
        eval_cfg_boot = getattr(config, 'EVAL', None) or {}
        if bool(eval_cfg_boot.get('USE_FIXED_VALSET', False)):
            eval_microbatch_boot = int(eval_cfg_boot.get('EVAL_MICROBATCH', config.BATCH_SIZE))
            eval_workers = int(getattr(config, "EVAL_NUM_WORKERS", min(4, int(getattr(config, "NUM_WORKERS", 0)))))
            fixed_eval_dataset = FixedEvalDataset(
                img_flist_path=eval_cfg_boot.get('VAL_IMG_FLIST', None),
                mask_flist_path=eval_cfg_boot.get('MASK_FLIST_EVAL', None),
                img_size=config.IMG_SIZE,
            )
            fixed_eval_loader = DataLoader(
                fixed_eval_dataset,
                batch_size=eval_microbatch_boot,
                shuffle=False,
                num_workers=eval_workers,
                pin_memory=(device.type == "cuda"),
                persistent_workers=(eval_workers > 0),
            )
            fixed_eval_desc = (
                f"{eval_cfg_boot.get('VAL_IMG_FLIST', None)} "
                f"(n={len(fixed_eval_dataset)}, batch={eval_microbatch_boot}, workers={eval_workers})"
            )
            print(f" Cached fixed eval DataLoader: {fixed_eval_desc}")

        total_steps_in_data = len(train_loader) * config.NUM_EPOCHS
        effective_total_steps = max(1, (config.MAX_STEPS_PER_RUN or total_steps_in_data))
        print(f"Total training steps available: {total_steps_in_data}")
        if config.MAX_STEPS_PER_RUN: print(f"Current run will stop after {config.MAX_STEPS_PER_RUN} steps.")

        generator = DualBranchUformer(img_channel=config.IMG_CHANNEL, out_channel=config.OUT_CHANNEL,
                                      embed_dim=config.EMBED_DIM, num_blocks=config.NUM_BLOCKS, heads=config.HEADS,
                                      encoder_blocks=getattr(config, 'ENCODER_BLOCKS', None),
                                      taylor_num_paths_per_stage=getattr(
                                          config, "TAYLOR_NUM_PATHS_PER_STAGE", (2, 2, 2, 2)
                                      ),
                                      fno_stages=getattr(config, "FNO_STAGES", (3,)),
                                      focusing_factor=config.FOCUSING_FACTOR, use_cpe=config.USE_CPE,
                                      fno_modes_per_stage=config.FNO_MODES_PER_STAGE,
                                      fno_channel_bottleneck=getattr(config, 'FNO_CHANNEL_BOTTLENECK', 0.5),
                                      use_dsdcn=config.USE_DSDCN, dsdcn_backend=config.DSDCN_BACKEND,
                                      dsdcn_mode=config.DSDCN_MODE, dsdcn_clamp=config.DSDCN_CLAMP,
                                      use_cmt=config.USE_CMT, cmt_stages=config.CMT_STAGES,
                                      cmt_alpha_max=config.CMT_ALPHA_MAX, cmt_warmup_steps=config.CMT_WARMUP_STEPS,
                                      cmt_shifted=config.CMT_SHIFTED,
                                      gradient_checkpointing=getattr(config, 'GRADIENT_CHECKPOINTING', False)
                                      ).to(device)
        discriminator = PatchGANDiscriminator(input_nc=config.DISC_INPUT_NC, return_features=True).to(device)

        if getattr(config, "torch_compile", False) and hasattr(torch, "compile"):
            print(f"Compiling models with torch.compile (mode='{config.compile_mode}')...")
            try:

                try:
                    import torch._inductor.config as _ind
                    if hasattr(_ind, "cudagraphs"): _ind.cudagraphs = False
                    if hasattr(_ind, "triton") and hasattr(_ind.triton, "cudagraphs"):
                        _ind.triton.cudagraphs = False
                except Exception:
                    pass


                generator = torch.compile(
                    generator,
                    backend=getattr(config, "compile_backend", "inductor"),
                    mode=getattr(config, "compile_mode", "reduce-overhead"),
                    fullgraph=bool(getattr(config, "compile_fullgraph", False)),
                    dynamic=bool(getattr(config, "compile_dynamic", True)),
                )
                if getattr(config, "compile_d", False):
                    discriminator = torch.compile(
                        discriminator,
                        backend=getattr(config, "compile_backend", "inductor"),
                        mode=getattr(config, "compile_mode", "reduce-overhead"),
                        fullgraph=bool(getattr(config, "compile_fullgraph", False)),
                        dynamic=bool(getattr(config, "compile_dynamic", True)),
                    )
                print(" Models compiled (G compiled: True, D compiled:", bool(getattr(config, "compile_d", False)),
                      ")")
            except Exception as e:
                print(f" Could not compile models, proceeding without compilation. Error: {e}")

        generator_ema = copy.deepcopy(generator)
        for p in generator_ema.parameters(): p.requires_grad = False
        generator_ema.eval()

        if dist_ctx.is_main and config.PROFILE_AT_START:
            profile_and_report_complexity(generator, config=config,
                                          input_shape=(1, config.IMG_CHANNEL + 1, config.IMG_SIZE, config.IMG_SIZE),
                                          group_rules={'Transformer': ['TaylorExpandedAttention', 'GatedFeedForward'],
                                                       'FNO': ['SpectralConv2d'], 'Deformable_Conv': ['DSDCN'],
                                                       'Convolution': ['Conv2d'],
                                                       'Other': ['GatedFusion', 'GroupNorm', 'LayerNorm', 'GELU',
                                                                 'Sigmoid', 'Tanh', 'PReLU', 'Softmax']})

        criterion_l1 = nn.L1Loss().to(device)
        criterion_perceptual = HRFPerceptualLoss(weights_path=str(config.HRF_LOSS_WEIGHTS_PATH), device=device).to(
            device)
        seam_criterion = SeamConsistencyLoss(kernel_size=config.SEAM_KERNEL_SIZE,
                                             grad_weight=getattr(config, 'SEAM_GRAD_WEIGHT', 0.5), use_grad=True)
        criterion_tv = TVLoss(device).to(device)

        config.LEARNING_RATE_G = float(config.LEARNING_RATE_G)
        config.LEARNING_RATE_D = float(config.LEARNING_RATE_D)
        adam_betas_g = getattr(config, 'ADAM_BETAS_G', (.0, .999));
        adam_betas_d = getattr(config, 'ADAM_BETAS_D', (.0, .999))
        optimizer_g = optim.AdamW(generator.parameters(), lr=config.LEARNING_RATE_G, betas=adam_betas_g)
        optimizer_d = optim.AdamW(discriminator.parameters(), lr=config.LEARNING_RATE_D, betas=adam_betas_d)
        fsm.attach_optimizer(optimizer_d)
        scheduler_g = None
        scheduler_d = None
        scaler = GradScaler()

        start_epoch, global_step, best_psnr, resume_gs, d_stalled_count = 0, 0, 0.0, 0, 0
        is_finetune = False
        last_strong_d_step = -10 ** 9
        last_probe = {"d_loss_now": 0.69, "d_rf_strength": 0.0, "strong_d_probe": False, "step": -1}
        recovery_until_step = -1
        r1_last_step, r1_recent_hits, r1_mask_value = -1, 0, 0.0
        r1_ema = torch.zeros((), device=device)
        r1_recent_window = getattr(config, "LOG_INTERVAL", 100);
        r1_ema_m = 0.95

        init_from_g_weights = getattr(config, 'INIT_FROM_G_EMA', None)
        if init_from_g_weights and os.path.exists(init_from_g_weights):
            print(f" Special Resume: Initializing G and G_EMA from weights-only file: {init_from_g_weights}")
            print("   -> Optimizers and schedulers will be reset.")
            try:

                smart_resume(init_from_g_weights, device, generator, generator_ema, discriminator, optimizer_g,
                             optimizer_d, scheduler_g, scheduler_d, scaler, config)

                target_g = unwrap_model(generator)
                target_ema = unwrap_model(generator_ema)
                load_state_dict_report(target_g, target_ema.state_dict(), "G_from_EMA", strict=True)
                start_epoch, global_step, best_psnr, resume_gs, d_stalled_count = 0, 0, 0.0, 0, 0
                print("   -> Model weights loaded. Training will start from step 0.")
            except Exception as e:
                print(f"   ->  ERROR: Could not load weights from {init_from_g_weights}. Starting fresh. Error: {e}")
        else:
            resume_path = config.RESUME_CHECKPOINT
            if config.RESUME and (resume_path or config.RESUME_AUTO):
                if not resume_path or not os.path.exists(resume_path): resume_path = checkpoint_dir / "latest.pth"
                if os.path.exists(resume_path):
                    global_step, best_psnr, start_epoch, loaded_gan_ctrl = smart_resume(
                        resume_path, device, generator, generator_ema, discriminator,
                        optimizer_g, optimizer_d, scheduler_g, scheduler_d, scaler, config
                    )


                    resume_gs = global_step

                    is_finetune = bool(getattr(config, "FINETUNE_MODE", False)) or (
                                selected_mode == 'finetune_from_best')


                    if global_step > 0:
                        print("\n[resume] Forcing LR from config to override checkpoint state...")
                        _apply_lr_from_config(optimizer_g, config.LEARNING_RATE_G)
                        _apply_lr_from_config(optimizer_d, config.LEARNING_RATE_D)
                        _sync_scheduler_base_lrs(scheduler_g, optimizer_g)
                        _sync_scheduler_base_lrs(scheduler_d, optimizer_d)
                        print(
                            f"         LR synced! G_LR={optimizer_g.param_groups[0]['lr']:.2e}, D_LR={optimizer_d.param_groups[0]['lr']:.2e}")

                        use_ema_bootstrap = bool(getattr(config, "INIT_FROM_G_EMA", False)) \
                                            or bool(getattr(config, "FINETUNE_MODE", False))

                        if use_ema_bootstrap:
                            tgtG = unwrap_model(generator)
                            tgtE = unwrap_model(generator_ema)
                            load_state_dict_report(tgtG, tgtE.state_dict(), "G_from_EMA_resume", strict=True)
                            print(" Initialized Generator weights from EMA (explicitly requested).")
                        else:

                            print(" Resume: kept Generator weights from checkpoint (no EMA override).")

                        if loaded_gan_ctrl:
                            last_strong_d_step = loaded_gan_ctrl.get("last_strong_d_step", last_strong_d_step)
                            last_probe = loaded_gan_ctrl.get("last_probe", last_probe)
                            print(
                                f"   -> Restored GAN state: last_strong_d_step={last_strong_d_step}, probe_step={last_probe.get('step')}")

                        recovery_steps = getattr(config, "RECOVERY_STEPS", 0)
                        if recovery_steps > 0:
                            recovery_until_step = global_step + recovery_steps
                            print(
                                f"   -> Recovery Window ACTIVE until step {recovery_until_step} ({recovery_steps} steps)")
                else:
                    print(f"[resume] no checkpoint at '{resume_path}', starting fresh.")


                if is_finetune:
                    print(f" Resumed from step {global_step}. Applying finetune settings.")



                    lr_mult_g = getattr(config, "FINETUNE_LR_MULT_G",
                                        getattr(config, "FINETUNE_LR_MULT", 1.0))
                    lr_mult_d = getattr(config, "FINETUNE_LR_MULT_D",
                                        getattr(config, "FINETUNE_LR_MULT", 1.0))

                    for pg in optimizer_g.param_groups: pg["lr"] *= lr_mult_g
                    for pg in optimizer_d.param_groups: pg["lr"] *= lr_mult_d

                    print(f"   -> Learning rates multiplied by G:{lr_mult_g}, D:{lr_mult_d}.")


                    freeze_steps = getattr(config, "FREEZE_ENCODER_STEPS", 0)
                    if freeze_steps > 0:
                        print(f"   -> Freezing generator encoder for the first {freeze_steps} steps.")
                        for name, p in generator.named_parameters():
                            if 'encoders.' in name or 'downsamples.' in name: p.requires_grad = False

        config.START_GLOBAL_STEP = global_step
        preflight_check(config, sched_mgr, optimizer_g, optimizer_d, global_step, discriminator=discriminator)

        if dist_ctx.launched:
            single_rank_cpu_rng = torch.get_rng_state() if dist_ctx.world_size == 1 else None
            single_rank_cuda_rng = (
                torch.cuda.get_rng_state(device)
                if dist_ctx.world_size == 1 and device.type == "cuda"
                else None
            )
            ddp_kwargs = {
                "broadcast_buffers": True,
                "gradient_as_bucket_view": True,
                "bucket_cap_mb": int(getattr(config, "DDP_BUCKET_CAP_MB", 25)),
            }
            if device.type == "cuda":
                ddp_kwargs.update(
                    device_ids=[dist_ctx.local_rank],
                    output_device=dist_ctx.local_rank,
                )
            generator = DistributedDataParallel(
                generator,
                find_unused_parameters=bool(
                    getattr(config, "DDP_FIND_UNUSED_PARAMETERS_G", False)
                ),
                **ddp_kwargs,
            )
            discriminator = DistributedDataParallel(
                discriminator,
                find_unused_parameters=False,
                **ddp_kwargs,
            )
            barrier()
            print(
                "[DDP] Models wrapped successfully "
                f"(find_unused_G={getattr(config, 'DDP_FIND_UNUSED_PARAMETERS_G', False)})."
            )
            if dist_ctx.world_size == 1:
                torch.set_rng_state(single_rank_cpu_rng)
                if single_rank_cuda_rng is not None:
                    torch.cuda.set_rng_state(single_rank_cuda_rng, device=device)
            else:
                rank_seed = (
                    int(config.GLOBAL_SEED)
                    + int(global_step) * int(dist_ctx.world_size)
                    + int(dist_ctx.rank)
                )
                config.DDP_RANK_SEED = rank_seed
                set_all_seeds(rank_seed)

        amp_mode = str(getattr(config, 'AMP_DTYPE', 'fp16')).lower()
        if amp_mode in ('fp32', 'float32', 'none', 'off', 'disable'):
            amp_enabled, amp_dtype = False, torch.float32
            print("AMP disabled; using fp32 for all operations.")
        elif amp_mode in ('bfloat16', 'bf16') and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            amp_enabled, amp_dtype = True, torch.bfloat16
            print("Using bfloat16 for Automatic Mixed Precision training.")
        else:
            if amp_mode in ('bfloat16', 'bf16'): print(" Warning: bfloat16 not supported, falling back to float16.")
            amp_enabled, amp_dtype = True, torch.float16
            print("Using float16 for Automatic Mixed Precision training.")

        total_start_time = time.time()
        device_type = 'cuda' if 'cuda' in device.type else 'cpu'
        done = False
        no_best_counter = 0
        if config.MAX_STEPS_PER_RUN and global_step >= effective_total_steps:
            print(f"Already at or beyond step budget (GS={global_step} / {effective_total_steps}). Exiting.")
            done = True

        early_stop_counter, best_weighted_psnr_for_early_stop = 0, 0.0
        step_latencies = deque(maxlen=config.LATENCY_WINDOW)
        PROBE_INTERVAL = 16
        cooldown_until_step, stall_count = -1, 0
        grad_accum_steps = int(getattr(config, "GRADIENT_ACCUM_STEPS", 1))
        warmup_steps = getattr(config, 'WARMUP_STEPS', 0)

        def get_recovery_aware_weight(cs, base, rec_cap):


            return min(base, rec_cap) if cs < recovery_until_step else base
        def apply_d_lr_floor(opt):
            floor = getattr(config, "D_LR_FLOOR", 0.0)
            if floor > 0:
                for g in opt.param_groups: g["lr"] = max(g["lr"], floor)

        print("\nStarting training loop...")

        # ======================================================================
        # ----------------- PART 2: MAIN TRAINING LOOP -------------------------
        # ======================================================================
        for epoch in range(start_epoch, config.NUM_EPOCHS):
            if done: break
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            epoch_start_time = time.time()
            data_iter = iter(train_loader)
            for i in range(len(train_loader)):
                if done: break
                local_step = global_step - resume_gs

                sched_mgr.step(global_step)
                state_overrides = fsm.overrides(global_step)
                current_state = state_overrides.get('state', 'Normal') if state_overrides else 'Normal'

                current_phase_name = sched_mgr.current_phase()
                is_rehab = (current_phase_name == 'd_rehab')

                tv_inhole_weight = sched_mgr.value('tv_inhole_weight', getattr(config, 'TV_INHOLE_WEIGHT', 0.15))
                if not is_rehab:
                    tv_inhole_weight = get_recovery_aware_weight(global_step, tv_inhole_weight,
                                                                 min(0.12, tv_inhole_weight))
                roi_scale = sched_mgr.value('roi_scale', 0.0)
                roi_adv_w = sched_mgr.value('roi_adv_weight', 0.0)
                gate_temp = sched_mgr.value('gate_temp', 1.0)
                spec_rate = sched_mgr.value('spectral_dropout_rate', 0.0)
                gate_prior_w = float(sched_mgr.value('gate_prior_weight', 0.0))
                if force_spec_drop is not None:
                    spec_rate = float(force_spec_drop)
                if force_gate_prior is not None:
                    gate_prior_w = float(force_gate_prior)
                l1_hole_w = sched_mgr.value('l1_hole_weight', 12.0)
                update_d_every = sched_mgr.update_d_every()
                r1_every = sched_mgr.r1_every()
                r1_gamma_sched = sched_mgr.r1_gamma()
                override_gan_cap = None
                inst_noise_override = None
                if state_overrides:
                    update_d_every = int(state_overrides.get('update_d_every', update_d_every))
                    r1_every = int(state_overrides.get('r1_every', r1_every))
                    if 'r1_gamma' in state_overrides:
                        r1_gamma_sched = float(state_overrides['r1_gamma'])
                    if 'gan_cap' in state_overrides:
                        override_gan_cap = float(state_overrides['gan_cap'])
                    inst_noise_override = state_overrides.get('inst_noise')
                    lr_override = state_overrides.get('lr_d') or state_overrides.get('lrD') or state_overrides.get('lr_d_target')
                    if lr_override is not None:
                        for pg in optimizer_d.param_groups:
                            pg['lr'] = float(lr_override)
                else:
                    lr_override = None

                if global_step < warmup_steps:
                    set_optimizer_lr_with_warmup(optimizer_g, config.LEARNING_RATE_G, global_step, warmup_steps)
                    set_optimizer_lr_with_warmup(optimizer_d, config.LEARNING_RATE_D, global_step, warmup_steps)

                sigma = 0.0;
                did_r1 = False;
                tD = 0.0;
                r1_penalty = torch.tensor(0.0, device=device)
                fsm_eval_payload = None

                try:
                    step_start_time = time.time()
                    data, _ = next(data_iter)
                    images = data.to(device, non_blocking=True) * 2. - 1.

                    # --- [FIX] Mask Curriculum Logic (Safe Access) ---
                    current_allowed_bins = None
                    mask_curriculum_cfg = getattr(config, 'MASK_CURRICULUM', None)


                    def _safe_get(obj, key, default):
                        if isinstance(obj, dict):
                            return obj.get(key, default)
                        return getattr(obj, key, default)

                    if mask_curriculum_cfg and _safe_get(mask_curriculum_cfg, 'ENABLED', False):
                        schedule = _safe_get(mask_curriculum_cfg, 'SCHEDULE', [])
                        for stage in schedule:

                            end_step = _safe_get(stage, 'end_step', float('inf'))
                            if global_step < end_step:
                                current_allowed_bins = _safe_get(stage, 'allowed_bins', None)
                                break


                    if current_allowed_bins and (global_step % 200 == 1 or global_step < 10):
                        bins_str = ", ".join(current_allowed_bins)
                        print(f"[MaskCurriculum] Step={global_step}, Allowed Bins: [{bins_str}]")
                        if 'writer' in locals() and writer:
                            writer.add_text("MaskCurriculum/Allowed Bins", bins_str, global_step)
                    # -------------------------------------------------


                    masks, mask_ratio_batch_mean = mask_generator_instance(batch_size=images.shape[0],
                                                                           allowed_bins=current_allowed_bins,
                                                                           seed=(
                                                                               config.GLOBAL_SEED
                                                                               + global_step * dist_ctx.world_size
                                                                               + dist_ctx.rank
                                                                           ))

                    # --- [NEW] Calculate Per-Sample Mask Statistics (P0 Requirement) ---
                    # Keep mask-derived control values and ROI boxes on CPU to avoid
                    # synchronizing the training stream once per sample and crop.
                    with torch.no_grad():
                        per_sample_ratios = masks.flatten(1).mean(1)
                        max_ratio_in_batch = per_sample_ratios.max().item()
                        ratio_gt_04 = (per_sample_ratios > 0.4).float().mean().item()
                        roi_bboxes = None
                        if config.USE_ROI_GAN and roi_scale > 0:
                            roi_bboxes = _prepare_roi_bboxes(masks)
                    masks = masks.to(device, non_blocking=True)


                    mask_ratio, ratio_gt_04 = mean_scalars(
                        (mask_ratio_batch_mean, ratio_gt_04),
                        device,
                    )
                    max_ratio_in_batch = max_scalar(max_ratio_in_batch, device)


                    adaptive_roi_cfg = getattr(config, 'ADAPTIVE_ROI_GAN', None)
                    if adaptive_roi_cfg and getattr(adaptive_roi_cfg, 'ENABLED', False):
                        if mask_ratio > getattr(adaptive_roi_cfg, 'MASK_AREA_THRESHOLD', 0.3):
                            roi_adv_w *= getattr(adaptive_roi_cfg, 'WEIGHT_MULTIPLIER', 1.5)



                    base_update_d_every = update_d_every


                    pulsed_d_cfg = getattr(config, 'PULSED_D_TRAINING', None)
                    if pulsed_d_cfg and getattr(pulsed_d_cfg, 'ENABLED', False):
                        start_step = getattr(pulsed_d_cfg, 'START_STEP', 70000)
                        if global_step > start_step:
                            pulse_every = getattr(pulsed_d_cfg, 'PULSE_EVERY', 1500)
                            pulse_duration = getattr(pulsed_d_cfg, 'PULSE_DURATION', 600)


                            if (global_step - start_step) % pulse_every < pulse_duration:
                                update_d_every = getattr(pulsed_d_cfg, 'PULSE_FREQ', base_update_d_every)
                            else:
                                update_d_every = base_update_d_every
                    else:
                        update_d_every = base_update_d_every



                    r1_gamma = r1_gamma_sched

                    masked_images = images * (1. - masks)

                    # ==============================================================
                    # --------- SUB-PART 2.2: GAN CONTROL & PROBING --------------
                    # ==============================================================
                    adv_frac = get_adv_scale(global_step, config.ADV_DELAY_STEPS, config.ADV_WARMUP_STEPS, cap=1.0)
                    adv_scale = adv_frac
                    GAN_COOLDOWN_CAP = getattr(config, "GAN_COOLDOWN_CAP", 0.40)
                    gan_cap_base = override_gan_cap if override_gan_cap is not None else sched_mgr.current_gan_cap()
                    cooldown_active = (global_step < cooldown_until_step)



                    recovery_cap = float(getattr(config, "GAN_RECOVERY_CAP", gan_cap_base))
                    gan_cap_now = min(gan_cap_base, recovery_cap) if cooldown_active else gan_cap_base
                    gan_scale_base = getattr(config, 'GAN_SCALE_BASE', 0.40);
                    gan_scale_mult = getattr(config, 'GAN_SCALE_MULT', 0.60);
                    gan_interim_cap = getattr(config, 'GAN_INTERIM_CAP', 0.60)
                    gan_scale = gan_scale_base + gan_scale_mult * (adv_scale ** 1.2)
                    if roi_scale < 0.5: gan_scale = min(gan_scale, gan_interim_cap)
                    gan_scale = min(gan_scale, gan_cap_now)


                    current_phase_name = sched_mgr.current_phase()
                    is_rehab = (current_phase_name == 'd_rehab')

                    adv_base = sched_mgr.value('adversarial_weight', config.ADVERSARIAL_WEIGHT)


                    recovery_adv_cap = float(getattr(config, "GAN_RECOVERY_CAP", 0.25))

                    if not is_rehab:

                        current_adv_weight = get_recovery_aware_weight(global_step, adv_base, recovery_adv_cap)
                    else:
                        current_adv_weight = adv_base

                    eff_adv_w = current_adv_weight * adv_scale * gan_scale
                    eff_roi_adv_w = roi_adv_w * adv_scale * gan_scale


                    current_probe_interval = PROBE_INTERVAL * 2 if local_step > 2000 else PROBE_INTERVAL
                    do_probe = (global_step % current_probe_interval == 0) or (global_step < 200)


                    from contextlib import nullcontext
                    try:
                        import torch._dynamo as dynamo
                    except Exception:
                        dynamo = None

                    if do_probe:

                        dyn_ctx = dynamo.disable() if (
                                    dynamo is not None and getattr(config, 'torch_compile', False)) else nullcontext()
                        with torch.no_grad(), autocast(device_type=device_type, dtype=amp_dtype,
                                                       enabled=amp_enabled), dyn_ctx:
                            g_mod = unwrap_model(generator)
                            g_out_probe, _ = g_mod(
                                torch.cat([masked_images, masks], dim=1),
                                config=None,
                                global_step=global_step, gate_temp=gate_temp
                            )
                            completed_probe = (masked_images + g_out_probe * masks).clamp(-1, 1)

                            d_probe_fake_input = torch.cat([completed_probe, masks],
                                                           dim=1) if config.USE_MASK_IN_D else completed_probe
                            d_probe_real_input = torch.cat([images, masks], dim=1) if config.USE_MASK_IN_D else images

                            _cg_mark_if_compiled(config)
                            probe_discriminator = unwrap_model(discriminator)
                            probe_fake_d, _ = probe_discriminator(d_probe_fake_input)
                            if getattr(config, 'torch_compile', False): probe_fake_d = probe_fake_d.clone()
                            _cg_mark_if_compiled(config)
                            probe_real, _ = probe_discriminator(d_probe_real_input)
                            if getattr(config, 'torch_compile', False): probe_real = probe_real.clone()


                            loss_d_real_probe, loss_d_fake_probe, _ = calculate_discriminator_loss(
                                probe_real, probe_fake_d,
                                completed_probe, images, masks, config,
                                roi_scale=0, sigma=0.0, roi_adv_w=0, discriminator=discriminator
                            )
                            d_loss_now, d_rf_strength = mean_scalars(
                                (
                                    (loss_d_real_probe + loss_d_fake_probe).item(),
                                    (
                                        probe_real.abs().mean()
                                        + probe_fake_d.abs().mean()
                                    ).item() * 0.5,
                                ),
                                device,
                            )


                        strong_d_probe = (d_loss_now < 0.08) or (d_rf_strength > 4.0)
                        if strong_d_probe:
                            last_strong_d_step = global_step
                            if do_probe:
                                stall_count += 1
                                cooldown_until_step = max(cooldown_until_step,
                                                          global_step + config.GAN_STRONG_COOLDOWN_STEPS)
                        last_probe.update(
                            {"d_loss_now": d_loss_now, "d_rf_strength": d_rf_strength, "strong_d_probe": strong_d_probe,
                             "step": global_step})
                    else:
                        d_loss_now, d_rf_strength, strong_d_probe = last_probe["d_loss_now"], last_probe[
                            "d_rf_strength"], last_probe["strong_d_probe"]

                    noise_sigma_max = getattr(config, 'INST_NOISE_SIGMA_MAX', 0.0)
                    noise_anneal_steps = int(getattr(config, 'INST_NOISE_ANNEAL_STEPS', 1))
                    phase_inst_noise = sched_mgr.value('inst_noise', None)
                    if phase_inst_noise is not None:
                        inst_noise_override = phase_inst_noise

                    if inst_noise_override:
                        if isinstance(inst_noise_override, (tuple, list)) and len(inst_noise_override) >= 2:
                            noise_sigma_max = float(inst_noise_override[0])
                            noise_anneal_steps = int(inst_noise_override[1])
                        elif isinstance(inst_noise_override, dict):
                            noise_sigma_max = float(inst_noise_override.get('sigma', noise_sigma_max))
                            noise_anneal_steps = int(inst_noise_override.get('anneal', inst_noise_override.get('anneal_steps', noise_anneal_steps)))

                    if strong_d_probe or ((global_step - last_strong_d_step) <= config.GAN_STRONG_COOLDOWN_STEPS):
                        anneal_steps = max(1, noise_anneal_steps)
                        k = min(1.0, max(0.0, (global_step - last_strong_d_step) / anneal_steps))
                        sigma = noise_sigma_max * (1.0 - k)

                    soft_stop_skip = (d_loss_now is not None) and (d_loss_now < config.D_SOFT_STOP_THRESHOLD)

                    if getattr(config, "ALLOW_SOFT_STOP_WITH_ROI_ZERO", False):
                        if (roi_scale == 0 and d_loss_now is not None and d_loss_now < 0.10
                                and global_step < 5000):
                            soft_stop_skip = True

                    if global_step < recovery_until_step: soft_stop_skip = False
                    if soft_stop_skip and getattr(config, 'D_SOFT_STOP_LEAK_STEPS', 0) > 0:
                        soft_stop_skip = (global_step % config.D_SOFT_STOP_LEAK_STEPS) != 0

                    if global_step < recovery_until_step:
                        soft_stop_skip = False
                    # ==============================================================
                    # --------- SUB-PART 2.3: GENERATOR UPDATE ---------------------
                    # ==============================================================
                    tG0 = time.time()
                    generator.train()

                    accum_idx = i % grad_accum_steps
                    if accum_idx == 0:
                        optimizer_g.zero_grad(set_to_none=True)

                    runtime_cfg = None if getattr(config, 'torch_compile', False) else config
                    gate_means = {}
                    gate_stats_step = global_step + 1
                    collect_gate_stats = dist_ctx.is_main and (
                            (gate_stats_step % config.LOG_INTERVAL == 0)
                            or (gate_stats_step == effective_total_steps)
                    )

                    with autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                        _cg_mark_if_compiled(config)
                        generated_residual, gate_maps = generator(
                            torch.cat([masked_images, masks], dim=1),
                            config=None if getattr(config, 'torch_compile', False) else config,
                            global_step=global_step,
                            spectral_dropout_rate=spec_rate,
                            gate_temp=gate_temp
                        )
                        if getattr(config, 'torch_compile', False):
                            generated_residual = generated_residual.clone()
                        completed_images = (masked_images + generated_residual * masks).clamp(-1, 1)
                        gate_prior_unweighted = torch.tensor(0.0, device=device)
                        if gate_prior_w > 0 and isinstance(gate_maps, dict) and gate_maps:
                            gate_prior_unweighted = compute_gate_prior_loss(gate_maps, masks)
                        gate_prior_term = gate_prior_unweighted * gate_prior_w
                        with torch.no_grad():
                            if collect_gate_stats and isinstance(gate_maps, dict) and gate_maps:

                                _mask_full = (masks > 0.5).float()

                                _ib, _ob = _inner_outer_bands(_mask_full, k=13)
                                _bound_full = (_ib + _ob).clamp(0, 1)
                                _inter_full = (_mask_full - _bound_full).clamp(0, 1)
                                # -----------------------------------------------------------

                                for k, gm in gate_maps.items():
                                    if gm is None: continue
                                    k_clean = str(k).replace("stage_", "")
                                    gm_mono = gm.mean(dim=1, keepdim=True) if gm.dim() == 4 and gm.size(1) > 1 else gm


                                    gm_all_flat = gm_mono.reshape(-1)
                                    gate_means[f'gate/{k_clean}_all'] = gm_all_flat.mean().item()
                                    gate_means[f'gate/{k_clean}_std'] = gm_all_flat.std().item()


                                    _b_down = F.interpolate(_bound_full, size=gm_mono.shape[-2:], mode='nearest')
                                    _i_down = F.interpolate(_inter_full, size=gm_mono.shape[-2:], mode='nearest')


                                    b_mask = _b_down > 0.5
                                    i_mask = _i_down > 0.5

                                    if b_mask.any():
                                        gate_means[f'gate/{k_clean}_boundary'] = gm_mono[b_mask].mean().item()
                                    else:
                                        gate_means[f'gate/{k_clean}_boundary'] = 0.0

                                    if i_mask.any():
                                        gate_means[f'gate/{k_clean}_interior'] = gm_mono[i_mask].mean().item()
                                    else:
                                        gate_means[f'gate/{k_clean}_interior'] = 0.0



                                    _h_down = F.interpolate(_mask_full, size=gm_mono.shape[-2:], mode='nearest')
                                    h_mask = _h_down > 0.5
                                    if h_mask.any():
                                        gate_means[f'gate/{k_clean}_hole'] = gm_mono[h_mask].mean().item()
                                        gate_means[f'gate/{k_clean}_hole_std'] = gm_mono[h_mask].std().item()
                                    else:
                                        gate_means[f'gate/{k_clean}_hole'] = 0.0
                                        gate_means[f'gate/{k_clean}_hole_std'] = 0.0


                                    gate_means[f'prior/{k_clean}_boundary_area'] = b_mask.float().mean().item()
                                    gate_means[f'prior/{k_clean}_interior_area'] = i_mask.float().mean().item()

                        loss_l1_hole = criterion_l1(completed_images * masks, images * masks)
                        binary_mask_loss_fastpath = bool(
                            getattr(config, "BINARY_MASK_LOSS_FASTPATH", False)
                        )
                        if binary_mask_loss_fastpath:
                            # completed_images copies the binary-mask valid region from images exactly.
                            loss_l1_valid = completed_images.new_tensor(0.0)
                        else:
                            loss_l1_valid = criterion_l1(
                                completed_images * (1 - masks), images * (1 - masks)
                            )
                        # --- [NEW] Dynamic TV Loss Weighting (P1 Requirement) ---

                        current_tv_weight = tv_inhole_weight
                        if max_ratio_in_batch > 0.35:
                            current_tv_weight *= 2.0

                        loss_tv_inhole = criterion_tv(completed_images * masks) * current_tv_weight
                        # --------------------------------------------------------

                        seam_w_phase = float(sched_mgr.value('seam_weight', 0.0))
                        seam_w_sched = get_scheduled_value(global_step, getattr(config, "SEAM_WEIGHT_SCHEDULE", []),
                                                           0.0)
                        seam_w = max(seam_w_phase, seam_w_sched)

                        if seam_w > 0:
                            seam_metrics = seam_criterion(completed_images, images, images, masks)
                            loss_seam_term = float(getattr(config, 'SEAM_LOSS_WEIGHT', 1.0)) * seam_w * seam_metrics[
                                "seam_loss"]
                        else:
                            zero = completed_images.new_tensor(0.0)
                            seam_metrics = {
                                "outer_l1": zero,
                                "inner_l1": zero,
                                "grad_l1": zero,
                                "seam_loss": zero,
                                "inner_px": 0,
                                "outer_px": 0,
                            }
                            loss_seam_term = zero
                        perc_sched_default = get_scheduled_value(
                            global_step, getattr(config, "PERC_WEIGHT_SCHEDULE", []), 0.0
                        )
                        perc_in_w = float(sched_mgr.value('perc_in_weight', perc_sched_default))
                        perc_out_w = float(sched_mgr.value('perc_out_weight', perc_in_w))
                        loss_perceptual = completed_images.new_tensor(0.0)
                        loss_perc_in = completed_images.new_tensor(0.0)
                        loss_perc_out = completed_images.new_tensor(0.0)
                        if perc_in_w > 0 or perc_out_w > 0:
                            perc_mask_in = _make_perc_mask(
                                masks, mode=config.PERC_MASK_MODE, band_px=config.PERC_BAND_WIDTH
                            )
                            with autocast(device_type=device_type, enabled=False):
                                comp_in = (
                                    completed_images * perc_mask_in + images * (1 - perc_mask_in)
                                ).float()
                                loss_perc_in = criterion_perceptual(comp_in, images.float())
                                if not binary_mask_loss_fastpath:
                                    perc_mask_out = 1.0 - masks
                                    comp_out = (
                                        completed_images * perc_mask_out
                                        + images * (1 - perc_mask_out)
                                    ).float()
                                    loss_perc_out = criterion_perceptual(
                                        comp_out, images.float()
                                    )
                            loss_perceptual = loss_perc_in + loss_perc_out
                            loss_perc_term = perc_in_w * loss_perc_in + perc_out_w * loss_perc_out
                        else:
                            if perc_sched_default > 0:
                                perc_mask = _make_perc_mask(
                                    masks, mode=config.PERC_MASK_MODE, band_px=config.PERC_BAND_WIDTH
                                )
                                comp_for_perc = completed_images * perc_mask + images * (1 - perc_mask)
                                with autocast(device_type=device_type, enabled=False):
                                    loss_perceptual = criterion_perceptual(
                                        comp_for_perc.float(), images.float()
                                    )
                            loss_perc_in = loss_perceptual
                            loss_perc_term = perc_sched_default * loss_perceptual
                        auxiliary_discriminator = unwrap_model(discriminator)
                        with _FreezeModuleParams(auxiliary_discriminator):
                            d_in_fake_g = torch.cat([completed_images, masks],
                                                    dim=1) if config.USE_MASK_IN_D else completed_images
                            _cg_mark_if_compiled(config)
                            pred_fake_g, fake_feats = auxiliary_discriminator(d_in_fake_g)
                            if getattr(config, 'torch_compile', False):
                                pred_fake_g = pred_fake_g.clone()
                                if fake_feats:  # Ensure fake_feats is not None
                                    fake_feats = [t.clone() for t in fake_feats]
                            loss_g_gan = -pred_fake_g.mean() if getattr(config, 'GAN_LOSS',
                                                                        'nsgan') == 'hinge' else F.softplus(
                                -pred_fake_g).mean()
                            gan_term = eff_adv_w * loss_g_gan

                            gan_g_roi_term = torch.tensor(0.0, device=device)
                            if config.USE_ROI_GAN and roi_scale > 0:
                                cur_fake_roi_imgs, roi_masks_for_g = build_roi_batch(
                                    completed_images,
                                    masks,
                                    config.D_ROI_SIZE,
                                    bboxes=roi_bboxes,
                                )
                                d_in_fake_roi = torch.cat([cur_fake_roi_imgs, roi_masks_for_g],
                                                          dim=1) if config.USE_MASK_IN_D else cur_fake_roi_imgs
                                _cg_mark_if_compiled(config)
                                pred_fake_roi, _ = auxiliary_discriminator(d_in_fake_roi)
                                if getattr(config, 'torch_compile', False):
                                    pred_fake_roi = pred_fake_roi.clone()
                                gan_g_roi_loss = -pred_fake_roi.mean() if getattr(config, 'GAN_LOSS',
                                                                                  'nsgan') == 'hinge' else F.softplus(
                                    -pred_fake_roi).mean()
                                gan_g_roi_term = gan_g_roi_loss * eff_roi_adv_w

                            with torch.no_grad():
                                d_in_real_fm = torch.cat([images, masks], dim=1) if config.USE_MASK_IN_D else images
                                _cg_mark_if_compiled(config)
                                _, real_feats = auxiliary_discriminator(d_in_real_fm)
                                if getattr(config, 'torch_compile', False):
                                    if real_feats:  # Ensure real_feats is not None
                                        real_feats = [t.clone() for t in real_feats]

                        loss_fm_roi = 0.0;
                        total_level_weight = 0.0
                        if real_feats and fake_feats:
                            for level_idx, (f_r, f_f) in enumerate(zip(real_feats, fake_feats)):

                                m = F.adaptive_max_pool2d(masks, output_size=f_r.shape[-2:])

                                m_dilated = F.max_pool2d(m, 3, 1, 1)
                                edge = (m_dilated - m).clamp(0, 1)
                                m_with_edge = (m + 0.3 * edge).clamp(0, 1)


                                level_w = 1.0 + 0.25 * level_idx


                                num = (f_r - f_f).abs().mul(m_with_edge).sum(dim=(1, 2, 3))
                                den = (m_with_edge.sum(dim=(1, 2, 3)) * f_r.shape[1]).clamp_min(1.0)
                                current_level_loss = (num / den).mean()

                                loss_fm_roi += level_w * current_level_loss
                                total_level_weight += level_w


                        if total_level_weight > 0:
                            loss_fm_roi /= total_level_weight

                        fm_w_phase = sched_mgr.value('fm_weight', None)

                        if fm_w_phase is not None:

                            fm_weight = float(fm_w_phase)
                            if not is_rehab:
                                fm_weight = get_recovery_aware_weight(global_step, fm_weight, min(4.0, fm_weight))
                            fm_scale = 1.0
                        else:

                            fm_scale = get_scheduled_value(global_step, getattr(config, "FM_WEIGHT_SCHEDULE", []), 1.0)
                            fm_weight_base = getattr(config, "FEATURE_MATCH_WEIGHT", 10.0)
                            fm_weight = get_recovery_aware_weight(global_step, fm_weight_base, 4.0)

                        loss_fm_term = fm_weight * fm_scale * loss_fm_roi




                        loss_lpips_term = torch.tensor(0.0, device=device)
                        lpips_w = float(sched_mgr.value('lpips_weight', 0.0))

                        if lpips_w > 0.0 and _LPIPS_AVAILABLE:

                            if 'lpips_train_model' not in globals() or globals().get('lpips_train_model') is None:
                                print(f"Lazy-loading LPIPS model for training at step {global_step}...")
                                globals()['lpips_train_model'] = lpips.LPIPS(net='vgg', spatial=True).to(device).eval()
                                for p in globals()['lpips_train_model'].parameters():
                                    p.requires_grad_(False)

                            lpips_microbatch = max(1, int(getattr(config, "LPIPS_TRAIN_MICROBATCH", images.shape[0])))
                            lpips_num = completed_images.new_tensor(0.0)
                            lpips_den = completed_images.new_tensor(0.0)
                            for mb_start in range(0, images.shape[0], lpips_microbatch):
                                mb_end = min(images.shape[0], mb_start + lpips_microbatch)
                                comp_mb = completed_images[mb_start:mb_end]
                                img_mb = images[mb_start:mb_end]
                                mask_mb = masks[mb_start:mb_end]
                                with autocast(device_type=device_type, enabled=False):
                                    lpips_map = globals()['lpips_train_model'](
                                        comp_mb.float(),
                                        img_mb.float()
                                    )
                                with torch.no_grad():
                                    m_down = F.interpolate(mask_mb, size=lpips_map.shape[-2:], mode='nearest')
                                    m_dilated = F.max_pool2d(m_down, kernel_size=5, stride=1, padding=2)
                                    m_roi = (m_down + 0.5 * (m_dilated - m_down)).clamp(0.0, 1.0)
                                    denom = m_roi.sum()
                                lpips_num = lpips_num + (lpips_map * m_roi).sum()
                                lpips_den = lpips_den + denom
                            loss_lpips_term = lpips_num / lpips_den.clamp_min(1e-8)


                        edge_weight = get_scheduled_value(global_step, getattr(config, "EDGE_WEIGHT_SCHEDULE", []),
                                                          getattr(config, 'EDGE_WEIGHT', 1.0))
                        if not getattr(config, "USE_EDGE_LOSS", True):
                            edge_weight = 0.0
                        loss_edge_term = torch.tensor(0.0, device=device)
                        if edge_weight > 0:
                            pred_grads = seam_metrics.get("_pred_grad")
                            gt_grads = seam_metrics.get("_gt_grad")
                            if pred_grads is None or gt_grads is None:
                                pred_grads = get_sobel_gradients_for_loss(completed_images)
                                gt_grads = get_sobel_gradients_for_loss(images)
                            loss_edge = criterion_l1(pred_grads * masks, gt_grads * masks)
                            loss_edge_term = loss_edge * edge_weight



                        base_spec_l1_weight = get_scheduled_value(
                            global_step,
                            getattr(config, "SPECTRAL_L1_WEIGHT_SCHEDULE", []),
                            getattr(config, "SPECTRAL_L1_WEIGHT", 0.1),
                        )

                        spec_l1_weight = sched_mgr.value("spectral_l1_weight", base_spec_l1_weight)
                        loss_spec_l1_term = torch.tensor(0.0, device=device)
                        if spec_l1_weight > 0:
                            pred_fft = torch.fft.rfft2(completed_images.float(), norm='ortho')
                            gt_fft = torch.fft.rfft2(images.float(), norm='ortho')
                            pred_mag = torch.abs(pred_fft)
                            gt_mag = torch.abs(gt_fft)

                            loss_spec_l1 = criterion_l1(
                                torch.log1p(pred_mag),
                                torch.log1p(gt_mag),
                            )
                            loss_spec_l1_term = loss_spec_l1 * spec_l1_weight


                        total_loss_g = (
                                loss_l1_hole * l1_hole_w +
                                loss_l1_valid * config.L1_VALID_WEIGHT +
                                loss_tv_inhole +
                                loss_seam_term +
                                loss_perc_term +
                                loss_fm_term +
                                gan_term +
                                gan_g_roi_term +
                                (lpips_w * loss_lpips_term) +
                                loss_edge_term +
                                loss_spec_l1_term +
                                gate_prior_term
                        )
                    nonfinite_g = any_true(
                        not bool(torch.isfinite(total_loss_g).detach().item()),
                        device,
                    )
                    if nonfinite_g:
                        print(
                            f"\n\033[91m[FATAL] Generator loss is NaN/Inf at step {global_step}.\033[0m")
                        if accum_idx == 0: optimizer_g.zero_grad(set_to_none=True); optimizer_d.zero_grad(
                            set_to_none=True)
                        if dist_ctx.launched:
                            raise FloatingPointError(
                                f"Non-finite generator loss on at least one rank at step {global_step}."
                            )
                        global_step += 1
                        continue

                    scaler.scale(total_loss_g / grad_accum_steps).backward()
                    tG = time.time() - tG0

                    # ==============================================================
                    # --------- SUB-PART 2.4: DISCRIMINATOR UPDATE -----------------
                    # ==============================================================
                    did_d = not (soft_stop_skip or (local_step < getattr(config, 'D_FREEZE_STEPS', 0)))
                    tD0 = time.time()
                    total_loss_d = torch.tensor(0.0, device=device)
                    grad_norm_d = torch.tensor(0.0, device=device)

                    if did_d:
                        if accum_idx == 0:
                            optimizer_d.zero_grad(set_to_none=True)

                        repeat_d = max(1, int(update_d_every))
                        discriminator.train()
                        training_discriminator = unwrap_model(discriminator)
                        accumulated_d_loss_for_log = torch.tensor(0.0, device=device)

                        if grad_accum_steps == 1:
                            optimizer_d.zero_grad(set_to_none=True)

                        for k in range(repeat_d):

                            d_input_fake = _d_inst_noise(completed_images.detach(), sigma)
                            d_input_real = _d_inst_noise(images, sigma)

                            r1_penalty_in_loop = torch.tensor(0.0, device=device)
                            if k == 0:  # R1 penalty is calculated only once per step
                                use_r1 = getattr(config, "USE_R1", False)
                                #r1_every = getattr(config, "R1_EVERY_STEPS", 16)
                                did_r1 = use_r1 and (global_step % r1_every == 0)


                                if did_r1:
                                    d_input_real.requires_grad_(True)

                                    # CRITICAL: Compute R1 penalty in full precision (FP32) to avoid instability
                                    with autocast(device_type=device_type, enabled=False):
                                        # The input to discriminator must be float32 for grad calculation
                                        d_real_for_r1 = torch.cat([d_input_real.float(), masks],
                                                                  dim=1) if config.USE_MASK_IN_D else d_input_real.float()
                                        pred_real_r1, _ = training_discriminator(d_real_for_r1)

                                        grad_real = torch.autograd.grad(
                                            outputs=pred_real_r1.sum(), inputs=d_input_real,
                                            create_graph=True, retain_graph=True, only_inputs=True
                                        )[0]

                                    r1_region = getattr(config, "R1_REGION", "all")
                                    if r1_region != "all":
                                        if r1_region == "border":
                                            inner_band, outer_band = _inner_outer_bands(masks, k=getattr(config,
                                                                                                         "BORDER_KERNEL",
                                                                                                         33))
                                            r1_mask = (inner_band + outer_band).clamp(0, 1)
                                        else:  # "inhole"
                                            r1_mask = masks

                                        # Ensure mask matches gradient dimensions
                                        if config.USE_MASK_IN_D:
                                            # If D takes mask as input, grad is on the image part only
                                            r1_mask = r1_mask.expand_as(d_input_real)
                                        else:
                                            r1_mask = r1_mask.expand_as(d_input_real)
                                        grad_real = grad_real * r1_mask

                                    r1_gamma = r1_gamma_sched
                                    # The penalty is scaled by r1_every to average out over steps
                                    r1_penalty_in_loop = (grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(
                                        1).mean()) * (r1_gamma / 2.0) * r1_every

                                    d_input_real.requires_grad_(False)
                                    r1_penalty = r1_penalty_in_loop.detach()  # For logging
                                    r1_last_step = global_step
                                    r1_recent_hits = min(r1_recent_window, r1_recent_hits + 1)
                                    if 'r1_mask' in locals(): r1_mask_value = float(r1_mask.mean().item())


                            with autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                                _cg_mark_if_compiled(config)
                                pred_fake_d, _ = discriminator(
                                    torch.cat([d_input_fake, masks], dim=1) if config.USE_MASK_IN_D else d_input_fake)
                                if getattr(config, 'torch_compile', False):
                                    pred_fake_d = pred_fake_d.clone()

                                _cg_mark_if_compiled(config)
                                pred_real, _ = training_discriminator(
                                    torch.cat([d_input_real, masks], dim=1) if config.USE_MASK_IN_D else d_input_real)
                                if getattr(config, 'torch_compile', False):
                                    pred_real = pred_real.clone()
                                loss_d_real, loss_d_fake, d_loss_roi_iter = calculate_discriminator_loss(pred_real,
                                                                                                         pred_fake_d,
                                                                                                         completed_images,
                                                                                                         images, masks,
                                                                                                         config,
                                                                                                         roi_scale,
                                                                                                         sigma,
                                                                                                         roi_adv_w,
                                                                                                         training_discriminator,
                                                                                                         roi_bboxes=roi_bboxes)
                                d_loss_roi = d_loss_roi_iter.detach()  # for logging

                                main_loss = loss_d_real + loss_d_fake + d_loss_roi_iter
                            nonfinite_d = any_true(
                                not bool(
                                    torch.isfinite(
                                        main_loss + (r1_penalty_in_loop if k == 0 else 0)
                                    ).detach().item()
                                ),
                                device,
                            )
                            if nonfinite_d:
                                raise FloatingPointError(
                                    f"Non-finite discriminator loss on at least one rank "
                                    f"at step {global_step}, repeat {k}."
                                )
                            accumulated_d_loss_for_log += (main_loss +
                                                           (r1_penalty_in_loop if k == 0 else 0)).detach()


                            if grad_accum_steps == 1:
                                loss_to_backward = (main_loss / repeat_d) + (r1_penalty_in_loop if k == 0 else 0)
                                scaler.scale(loss_to_backward).backward()
                                if config.GRAD_CLIP_NORM > 0:
                                    grad_norm_d = torch.nn.utils.clip_grad_norm_(
                                        discriminator.parameters(), max_norm=config.GRAD_CLIP_NORM)
                            else:
                                loss_to_backward = (main_loss / repeat_d) + (r1_penalty_in_loop if k == 0 else 0)
                                scaler.scale(loss_to_backward / grad_accum_steps).backward()

                        total_loss_d = accumulated_d_loss_for_log / repeat_d
                        if grad_accum_steps == 1:
                            scaler.unscale_(optimizer_d)
                            if config.GRAD_CLIP_NORM > 0:
                                grad_norm_d = torch.nn.utils.clip_grad_norm_(
                                    discriminator.parameters(), max_norm=config.GRAD_CLIP_NORM
                                )
                            scaler.step(optimizer_d)
                    else:
                        with torch.no_grad():
                            total_loss_d = torch.tensor(d_loss_now)

                    tD = time.time() - tD0

                    # ======================================================
                    # --- SUB-PART 2.5: OPTIMIZER STEP, SCHEDULER, EMA UPDATE ------
                    # ======================================================

                    if accum_idx == grad_accum_steps - 1:
                        # 1. Step Generator
                        scaler.unscale_(optimizer_g)
                        grad_norm_g = torch.nn.utils.clip_grad_norm_(generator.parameters(),
                                                                     max_norm=config.GRAD_CLIP_NORM)
                        scaler.step(optimizer_g)

                        # 2. Step Discriminator (if it was updated)
                        if did_d and grad_accum_steps > 1:
                            scaler.unscale_(optimizer_d)
                            grad_norm_d = torch.nn.utils.clip_grad_norm_(discriminator.parameters(),
                                                                         max_norm=config.GRAD_CLIP_NORM)
                            scaler.step(optimizer_d)


                        scaler.update()


                        if global_step >= warmup_steps:

                            lr_d_target_fsm = state_overrides.get('lr_d') or state_overrides.get(
                                'lrD') or state_overrides.get('lr_d_target')

                            if lr_d_target_fsm is not None:

                                for pg in optimizer_d.param_groups:
                                    pg['lr'] = float(lr_d_target_fsm)
                            else:

                                lr_d_target_phase = sched_mgr.value("lr_d_target", None)
                                if lr_d_target_phase is not None:
                                    phase_start_step = sched_mgr.value("_phase_start_step", -1)
                                    should_reset = sched_mgr.value("reset_d_optimizer", False)

                                    if should_reset and global_step == phase_start_step:
                                        current_phase_name = sched_mgr.current_phase()
                                        print(
                                            f"\n[Intervention] Resetting Discriminator optimizer state at start of phase '{current_phase_name}' with LR={lr_d_target_phase:.2e}")
                                        override_lr_and_reset(optimizer_d, lr_d_target_phase)
                                    else:
                                        for pg in optimizer_d.param_groups:
                                            pg['lr'] = float(lr_d_target_phase)


                            lr_g_target_phase = sched_mgr.value("lr_g_target", None)
                            if lr_g_target_phase is not None:
                                for pg in optimizer_g.param_groups:
                                    pg['lr'] = float(lr_g_target_phase)


                            _sync_scheduler_base_lrs(scheduler_g, optimizer_g)
                            _sync_scheduler_base_lrs(scheduler_d, optimizer_d)


                            apply_d_lr_floor(optimizer_d)


                        if dist_ctx.is_main:
                            with torch.no_grad():
                                ema_m = config.EMA_MOMENTUM
                                for p, q in zip(
                                        unwrap_model(generator).parameters(),
                                        unwrap_model(generator_ema).parameters()):
                                    q.data.mul_(ema_m).add_(p.data, alpha=1 - ema_m)
                    # --- END OF MOVE ---

                    if device.type == "cuda" and bool(getattr(config, "SYNC_TIMING", False)):
                        torch.cuda.synchronize()
                    time_step = time.time() - step_start_time
                    step_latencies.append(time_step * 1000)
                    global_step += 1

                    if config.MAX_STEPS_PER_RUN and global_step >= effective_total_steps:
                        print(f"Reached step budget (GS={global_step}/{effective_total_steps}).")
                        done = True
                        break
                    r1_recent_hits = max(0, r1_recent_hits - 1)
                    r1_ema = r1_ema_m * r1_ema + (1 - r1_ema_m) * r1_penalty

                    # ==============================================================
                    # --------- SUB-PART 2.6: LOGGING & VISUALIZATION ------------
                    # ==============================================================
                    log_due = (
                        (global_step % config.LOG_INTERVAL == 0)
                        or (global_step == effective_total_steps)
                    )
                    should_log = dist_ctx.is_main and log_due
                    if should_log:

                        steps_done = (i + 1);
                        epoch_elapsed_time = time.time() - epoch_start_time
                        time_per_step_avg = epoch_elapsed_time / max(1, steps_done);
                        eta_seconds = time_per_step_avg * (effective_total_steps - global_step)
                        eta_str = format_time(eta_seconds)
                        latencies_np = np.array(step_latencies);
                        p50 = np.percentile(latencies_np, 50) if len(latencies_np) > 0 else 0;
                        p95 = np.percentile(latencies_np, 95) if len(latencies_np) > 0 else 0
                        lr_g_eff = optimizer_g.param_groups[0]['lr']
                        lr_d_eff = optimizer_d.param_groups[0]['lr']
                        base_lr_g = float(optimizer_g.param_groups[0].get('initial_lr', lr_g_eff))
                        base_lr_d = float(optimizer_d.param_groups[0].get('initial_lr', lr_d_eff))


                        if device.type == "cuda":
                            cuda_mem_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                            cuda_mem_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
                        else:
                            cuda_mem_alloc_mb = 0.0
                            cuda_mem_reserved_mb = 0.0


                        effective_batch = config.BATCH_SIZE * int(getattr(config, "GRADIENT_ACCUM_STEPS", 1))
                        throughput_imgs_per_s = effective_batch / max(time_per_step_avg, 1e-6)

                        phase_snapshot = sched_mgr.context()
                        phase_name_for_log = sched_mgr.current_phase()
                        phase_start_step = None
                        if isinstance(phase_snapshot, dict):
                            phase_start_step = phase_snapshot.get('_phase_start_step', None)

                        with torch.no_grad():
                            r_raw, f_raw = (pred_real.detach(), pred_fake_d.detach()) if 'pred_real' in locals() else (
                                torch.tensor(0.), torch.tensor(0.))

                            r32, f32 = r_raw.float(), f_raw.float()
                            d_real_mean_raw, d_fake_mean_raw = r32.mean().item(), f32.mean().item()
                            l1_gap = (r32 - f32).abs().mean().item()
                            pct_real_gt1 = float(((r32 > 1.0).float().mean().item()) if r32.numel() > 0 else 0.0)
                            pct_fake_lt_neg1 = float(((f32 < -1.0).float().mean().item()) if f32.numel() > 0 else 0.0)

                            d_real_prob = torch.sigmoid(r32).mean().item() if r32.numel() > 0 else 0.0
                            d_fake_prob = torch.sigmoid(f32).mean().item() if f32.numel() > 0 else 0.0

                        log_str = (
                            f'E:{epoch + 1}, S:{steps_done}/{len(train_loader)}, GS:{global_step}/{effective_total_steps}, ETA:{eta_str} | '
                            f'G_Loss: {total_loss_g.item():.3f}, D_Loss: {total_loss_d.item():.3f} | '
                            f'Gap:{l1_gap:.2f} Conf(R/F):{pct_real_gt1:.0%}/{pct_fake_lt_neg1:.0%}'
                        )

                        log_str += (
                            f" | LR G/D: {lr_g_eff:.2e}/{lr_d_eff:.2e} (Base: {base_lr_g:.2e}/{base_lr_d:.2e})"
                        )
                        log_str += (
                            f" | G Parts-> L1H:{(loss_l1_hole * l1_hole_w).item():.2f} FM:{loss_fm_term.item():.2f} Seam:{loss_seam_term.item():.2f} Perc:{loss_perc_term.item():.4f} GAN:{(gan_term + gan_g_roi_term).item():.2f} Gate:{gate_prior_term.detach().item():.4f}"
                        )
                        log_str += (
                            f" | Lat(ms) p50/p95:{p50:.1f}/{p95:.1f} | adv:{adv_scale:.2f}/{gan_cap_now:.2f} gan:{gan_scale:.2f}"
                        )
                        log_str += f" | pctR>1:{pct_real_gt1:.2f} pctF<-1:{pct_fake_lt_neg1:.2f}"

                        if global_step < recovery_until_step: log_str += " \033[96m(Recovery Window)\033[0m"
                        stall_thr = float(getattr(config, 'D_STALLED_GAP_ABS', 0.03))

                        if (l1_gap < stall_thr) and (global_step > config.ADV_WARMUP_STEPS):
                            d_stalled_count += 1

                            log_str += f" \033[93m(D STALLED WARNING! Count: {d_stalled_count} | Gap={l1_gap:.3f})\033[0m"
                        else:


                            d_stalled_count = 0
                        if d_stalled_count >= getattr(config, "D_STALL_TOLERANCE", 16):
                            print(
                                f"\n[Self-Heal] D stalled for {d_stalled_count} steps. Triggering recovery mode for {getattr(config, 'RECOVERY_STEPS', 800)} steps.")
                            recovery_until_step = max(recovery_until_step,
                                                      global_step + getattr(config, "RECOVERY_STEPS", 800))
                            d_stalled_count = 0
                        if not did_d: log_str += f" (D skip, L={d_loss_now:.3f}, |R/F|={d_rf_strength:.2f})"
                        print(log_str)

                        with torch.no_grad():
                            w_real = F.interpolate(masks, size=r_raw.shape[-2:], mode='nearest');
                            w_fake = F.interpolate(masks, size=f_raw.shape[-2:], mode='nearest')

                            def hole_mean_with(w_, x): num = (x * w_).sum(); den = w_.sum().clamp_min(1e-6); return (
                                        num / den).item() if den > 0 else 0.0

                            d_real_hole_val = hole_mean_with(w_real, r32);
                            d_fake_hole_val = hole_mean_with(w_fake, f32)

                        raw_perc = float(loss_perceptual.detach().item()) if isinstance(
                            loss_perceptual, torch.Tensor
                        ) else float(loss_perceptual)
                        raw_fm_val = float(loss_fm_roi.detach().item()) if isinstance(
                            loss_fm_roi, torch.Tensor
                        ) else float(loss_fm_roi)
                        raw_seam = float(seam_metrics["seam_loss"].detach().item()) if isinstance(
                            seam_metrics["seam_loss"], torch.Tensor
                        ) else float(seam_metrics["seam_loss"])
                        with torch.no_grad():
                            mask_mean = float(masks.mean().item())
                            inner_band, outer_band = _inner_outer_bands(
                                masks, k=int(getattr(config, "SEAM_KERNEL_SIZE", 7))
                            )
                            seam_band_ratio = float((inner_band + outer_band).clamp_max(1.0).mean().item())
                            roi_enabled = 1.0 if float(sched_mgr.value('roi_scale', 0.0)) > 0 else 0.0

                        use_mask_in_d_flag = 1.0 if getattr(config, 'USE_MASK_IN_D', False) else 0.0
                        use_roi_gan_flag = 1.0 if (float(roi_scale) > 0 and float(eff_roi_adv_w) > 0) else 0.0
                        phase_name_str = phase_name_for_log if phase_name_for_log is not None else "None"
                        phase_start_val = int(phase_start_step) if phase_start_step is not None else -1

                        if gate_maps is not None and isinstance(gate_maps, dict) and gate_maps and (global_step % config.METRICS_INTERVAL == 0):
                            for k, gm in gate_maps.items():
                                if gm is None: continue
                                gm_mono = gm.mean(dim=1, keepdim=True) if gm.dim() == 4 and gm.size(1) > 1 else gm
                                writer.add_histogram(f"Gate_Hist/{k}_all", gm_mono, global_step)
                                gm_down = F.adaptive_avg_pool2d(gm_mono, masks.shape[-2:])
                                gm_down_mask = F.adaptive_avg_pool2d(masks, masks.shape[-2:])
                                hole_vals = gm_down[gm_down_mask > 0.5]
                                if hole_vals.numel() > 0:
                                    writer.add_histogram(f"Gate_Hist/{k}_hole", hole_vals, global_step)

                        log_data = {"global_step": global_step, "local_step": local_step, "epoch": epoch + 1,
                                    "g_total": total_loss_g.item(), "d_total": total_loss_d.item(),
                                    "l1_valid": loss_l1_valid.item() * config.L1_VALID_WEIGHT,
                                    "l1_hole": loss_l1_hole.item() * l1_hole_w, "l1_hole_w": l1_hole_w,
                                    "hrf_perc": loss_perc_term.item(),
                                    "loss_raw/perc": raw_perc,
                                    "loss_raw/fm": raw_fm_val,
                                    "loss_raw/seam": raw_seam,
                                    "coverage/perc_mask": mask_mean,
                                    "coverage/seam_band": seam_band_ratio,
                                    "roi/enabled": roi_enabled,
                                    "use_mask_in_d": use_mask_in_d_flag,
                                    "use_roi_gan": use_roi_gan_flag,
                                    "seam_loss": loss_seam_term.item(),
                                    "inner_l1": seam_metrics["inner_l1"].item(),
                                    "grad_l1": seam_metrics["grad_l1"].item(), "tv_inhole": loss_tv_inhole.item(),
                                    "gate_prior_loss": gate_prior_term.detach().item(),
                                    "gate_prior_w": gate_prior_w,
                                    "loss_raw/gate_prior": gate_prior_unweighted.detach().item(),
                                    "gan_g": loss_g_gan.item(), "gan_g_roi": gan_g_roi_term.item(),
                                    "d_roi": d_loss_roi.item(), "feat_match": loss_fm_term.item(),
                                    "r1_gp": r1_penalty.item(), "r1_ema": r1_ema.item(),
                                    "r1_last_step": int(r1_last_step),
                                    "r1_every_effective": int(r1_recent_hits),
                                    "r1_mask_ratio": r1_mask_value if did_r1 else 0.0,
                                    "flag_r1_next": int(((global_step + 1) % int(r1_every)) == 0), "d_gap": l1_gap,
                                    "pct_real_gt1": pct_real_gt1,
                                    "pct_fake_lt_neg1": pct_fake_lt_neg1,
                                    "mask_ratio": mask_ratio,
                                    "mask_ratio_max": max_ratio_in_batch,
                                    "pct_big_hole": ratio_gt_04,
                                    "lr_g": lr_g_eff, "lr_d": lr_d_eff,
                                    "grad_norm_g": grad_norm_g.item(), "grad_norm_d": grad_norm_d.item(),
                                    "d_real_mean": d_real_mean_raw, "d_fake_mean": d_fake_mean_raw,
                                    "d_rf_strength": d_rf_strength, "d_real_hole": d_real_hole_val,
                                    "d_fake_hole": d_fake_hole_val, "adversarial_weight": float(current_adv_weight),
                                    "adv_scale": float(adv_scale), "gan_scale": float(gan_scale),
                                    "gan_cap_base": float(gan_cap_base), "gan_cap_now": float(gan_cap_now),
                                    "tv_inhole_weight_now": float(tv_inhole_weight),
                                    "eff_adv_w": float(eff_adv_w), "eff_roi_adv_w": float(eff_roi_adv_w),
                                    "cooldown_active": int(cooldown_active), "stall_count": int(stall_count),
                                    "d_update_every": int(update_d_every), "roi_scale": float(roi_scale),
                                    "roi_adv_w": float(roi_adv_w), "gate_temp": gate_temp, "spec_rate": spec_rate,
                                    "d_stalled_count": d_stalled_count, "inst_noise_sigma": float(sigma),
                                    "latency_p50": p50,
                                    "latency_p95": p95, "t_g": tG, "t_d": tD, "flag_d": int(did_d),
                                    "flag_r1": int(did_r1), "time_step": time_step,
                                    "ctx_phase": phase_name_str,
                                    "ctx_phase_start": phase_start_val,
                                    "ctx_fsm_state": current_state,
                                    "prob_real": d_real_prob,
                                    "prob_fake": d_fake_prob,
                                    "scaler_scale": float(scaler.get_scale()),
                                    "cuda_mem_alloc_mb": float(cuda_mem_alloc_mb),
                                    "cuda_mem_reserved_mb": float(cuda_mem_reserved_mb),
                                    "throughput_imgs_per_s": float(throughput_imgs_per_s),
                                    }

                        if global_step % config.METRICS_INTERVAL == 0:
                            generator_ema.eval()
                            try:
                                completed_eval, gt_eval, masks_eval = [], [], []

                                eval_cfg = getattr(config, 'EVAL', None) or {}

                                use_fixed_eval = bool(eval_cfg.get('USE_FIXED_VALSET', False))
                                if use_fixed_eval:
                                    if fixed_eval_loader is None:
                                        raise RuntimeError("Fixed eval requested but cached loader was not initialized.")
                                    print(f"EVAL(FIXED) on {fixed_eval_desc}...")
                                    ema_forward_cfg = None if getattr(config, 'torch_compile', False) else config
                                    with torch.no_grad(), autocast(device_type=device_type, dtype=amp_dtype,
                                                                   enabled=amp_enabled):
                                        for eval_imgs, eval_masks in fixed_eval_loader:
                                            vi, vm = (eval_imgs.to(device) * 2.) - 1., eval_masks.to(device)
                                            vmasked = vi * (1. - vm)
                                            gout, _ = generator_ema(
                                                torch.cat([vmasked, vm], dim=1),
                                                config=ema_forward_cfg,
                                                global_step=global_step,
                                            )
                                            completed_eval.append((vmasked + gout * vm).clamp(-1, 1).cpu());
                                            gt_eval.append(vi.cpu());
                                            masks_eval.append(vm.cpu())
                                    completed_es, vis_images_eval, vis_masks_es = torch.cat(completed_eval,
                                                                                            dim=0), torch.cat(gt_eval,
                                                                                                              dim=0), torch.cat(
                                        masks_eval, dim=0)
                                else:
                                    ema_forward_cfg = None if getattr(config, 'torch_compile', False) else config
                                    with torch.no_grad(), autocast(device_type=device_type, dtype=amp_dtype,
                                                                   enabled=amp_enabled):
                                        outs = []
                                        for i_micro in range(0, vis_images_fixed.size(0), config.EVAL_MICROBATCH):
                                            vi, vm = vis_images_fixed[
                                                     i_micro:i_micro + config.EVAL_MICROBATCH], vis_masks_fixed[
                                                                                                i_micro:i_micro + config.EVAL_MICROBATCH]
                                            vmasked = vi * (1. - vm)
                                            gout, _ = generator_ema(
                                                torch.cat([vmasked, vm], dim=1),
                                                config=ema_forward_cfg,
                                                global_step=global_step,
                                            )
                                            outs.append((vmasked + gout * vm).clamp(-1, 1))
                                        completed_es, vis_images_eval, vis_masks_es = torch.cat(outs,
                                                                                                dim=0), vis_images_fixed, vis_masks_fixed

                                psnr_eval_full, ssim_eval_full, psnr_eval_mask, ssim_eval_mask = calculate_masked_metrics(
                                    completed_es, vis_images_eval, vis_masks_es)
                                log_data.update({"psnr_eval_full": psnr_eval_full, "ssim_eval_full": ssim_eval_full,
                                                 "psnr_eval_mask": psnr_eval_mask, "ssim_eval_mask": ssim_eval_mask})


                                try:
                                    with torch.no_grad():
                                        fft_pred = torch.fft.rfft2(completed_es.float(), norm='ortho')
                                        fft_gt = torch.fft.rfft2(vis_images_eval.float(), norm='ortho')
                                        fft_mag_error = torch.mean(torch.abs(torch.abs(fft_pred) - torch.abs(fft_gt))).item()
                                except Exception as _e_fft:
                                    fft_mag_error = 0.0

                                log_data["fft_mag_error"] = fft_mag_error

                                adv_metrics = {}
                                if (lpips_vgg_eval or lpips_alex_eval or getattr(config, 'EVAL_BORDER_SSIM', True)):
                                    adv_metrics_batches = []
                                    for i_micro in range(0, completed_es.size(0), config.EVAL_MICROBATCH):
                                        batch_metrics = calculate_advanced_eval_metrics(
                                            completed_es[i_micro:i_micro + config.EVAL_MICROBATCH].to(device,
                                                                                                      non_blocking=True),
                                            vis_images_eval[i_micro:i_micro + config.EVAL_MICROBATCH].to(device,
                                                                                                         non_blocking=True),
                                            vis_masks_es[i_micro:i_micro + config.EVAL_MICROBATCH].to(device,
                                                                                                      non_blocking=True),
                                            config,
                                            lpips_vgg=lpips_vgg_eval, lpips_alex=lpips_alex_eval)
                                        adv_metrics_batches.append(batch_metrics)
                                    for k in adv_metrics_batches[0].keys():
                                        valid_vals = [d.get(k) for d in adv_metrics_batches if
                                                      d.get(k) is not None and not np.isnan(d.get(k))]
                                        if valid_vals: adv_metrics[k] = np.mean(valid_vals)
                                    log_data.update(adv_metrics)
                                    if 'lpips_mask_vgg' in log_data: log_data['lpips_eval_mask'] = log_data[
                                        'lpips_mask_vgg']
                            except Exception as e:
                                print(f"Warning: Could not compute metrics for eval. Error: {e}")
                                psnr_eval_full, ssim_eval_full, psnr_eval_mask, ssim_eval_mask = 0.0, 0.0, 0.0, 0.0

                            metrics_str = (
                                f"[Metrics] Full PSNR/SSIM: {psnr_eval_full:.2f}/{ssim_eval_full:.4f} | Mask PSNR/SSIM: {psnr_eval_mask:.2f}/{ssim_eval_mask:.4f} | Best(Full): {best_psnr:.2f}")
                            metrics_str += (
                                f" | lrD={optimizer_d.param_groups[0]['lr']:.2e}"
                                f" | updD={update_d_every} | r1Every={r1_every} | ganCap={gan_cap_now:.2f}"
                                f" | effAdv={eff_adv_w:.2f} | State={current_state}"
                            )
                            # ======================================================================

                            # ======================================================================
                            compute_fid_in_train = bool(eval_cfg.get('COMPUTE_FID_IN_TRAIN', False))
                            if compute_fid_in_train and _FID_AVAILABLE and 'completed_es' in locals() and completed_es is not None:
                                print(f"[Metrics] Starting in-training FID calculation...", flush=True)
                                with tempfile.TemporaryDirectory() as temp_dir:
                                    real_dir = Path(temp_dir) / "real"
                                    fake_dir = Path(temp_dir) / "fake"
                                    real_dir.mkdir();
                                    fake_dir.mkdir()

                                    max_samples = getattr(config.EVAL, 'FID_EVAL_SAMPLES', 1000)
                                    num_samples_to_save = min(len(completed_es), max_samples)

                                    for k in range(num_samples_to_save):
                                        save_image(denorm_minus1_1_to_0_1(completed_es[k]),
                                                   fake_dir / f"img_{k:05d}.png")
                                        save_image(denorm_minus1_1_to_0_1(vis_images_eval[k]),
                                                   real_dir / f"img_{k:05d}.png")

                                    try:
                                        metrics = calculate_metrics(
                                            input1=str(real_dir), input2=str(fake_dir),
                                            cuda=torch.cuda.is_available(), fid=True, isc=False, kid=False,
                                            verbose=False
                                        )
                                        fid_score = metrics.get('frechet_inception_distance', float('nan'))
                                        log_data['fid_eval_full'] = fid_score
                                        metrics_str += f" | FID: {fid_score:.3f}"
                                    except Exception as e:
                                        print(f"Warning: In-training FID calculation failed. Error: {e}")
                            # ======================================================================

                            # ======================================================================
                            if 'lpips_mask_alex' in log_data and not np.isnan(log_data[
                                                                                  'lpips_mask_alex']): metrics_str += f" | LPIPS-A(M): {log_data['lpips_mask_alex']:.4f}"
                            if 'lpips_mask_vgg' in log_data and not np.isnan(log_data[
                                                                                 'lpips_mask_vgg']): metrics_str += f" | LPIPS-V(M): {log_data['lpips_mask_vgg']:.4f}"
                            if 'ssim_border' in log_data and not np.isnan(
                                log_data['ssim_border']): metrics_str += f" | SSIM-B: {log_data['ssim_border']:.4f}"
                            print(metrics_str)
                            fsm_eval_payload = {
                                "global_step": int(global_step),
                                "d_gap": float(log_data.get('d_gap', 0.0)),
                                "psnr_eval_full": float(psnr_eval_full),
                            }
                            fsm.on_eval(
                                fsm_eval_payload["global_step"],
                                fsm_eval_payload["d_gap"],
                                fsm_eval_payload["psnr_eval_full"],
                            )

                            new_best = False
                            if psnr_eval_full > best_psnr:
                                best_psnr = psnr_eval_full
                                best_path = save_full_checkpoint(checkpoint_dir, "best_generator_ema.pth",
                                                                 epoch=epoch + 1, global_step=global_step,
                                                                 best_psnr=best_psnr, generator=generator,
                                                                 generator_ema=generator_ema,
                                                                 discriminator=discriminator, optimizer_g=optimizer_g,
                                                                 optimizer_d=optimizer_d, scheduler_g=scheduler_g,
                                                                 scheduler_d=scheduler_d, scaler=scaler,
                                                                 d_stalled_count=d_stalled_count, Config=config,
                                                                 last_strong_d_step=last_strong_d_step,
                                                                 last_probe=last_probe)
                                print(f"New best (full-eval, EMA): {best_psnr:.2f}dB -> {best_path}")
                                no_best_counter = 0
                                new_best = True

                                try:
                                    generator_ema.eval()
                                    ema_forward_cfg = None if getattr(config, 'torch_compile', False) else config
                                    with torch.no_grad(), autocast(device_type=device_type, dtype=amp_dtype,
                                                                   enabled=amp_enabled):

                                        vis_images_for_best = vis_images_eval[:config.SAVE_VIZ_PER_EPOCH].to(device)
                                        vis_masks_for_best = vis_masks_es[:config.SAVE_VIZ_PER_EPOCH].to(device)

                                        vis_masked_best = vis_images_for_best * (1. - vis_masks_for_best)
                                        gout_best, _ = generator_ema(
                                            torch.cat([vis_masked_best, vis_masks_for_best], dim=1),
                                            config=ema_forward_cfg,
                                            global_step=global_step, spectral_dropout_rate=0.0)
                                        completed_best = (vis_masked_best + gout_best * vis_masks_for_best).clamp(-1, 1)

                                    viz_best_dir = checkpoint_dir.parent / "viz_best";
                                    viz_best_dir.mkdir(parents=True, exist_ok=True)
                                    out_path = viz_best_dir / f"gs_{global_step:06d}_psnr_{best_psnr:.2f}.png"
                                    save_visualization_grid(images=vis_masked_best, masks=vis_masks_for_best,
                                                            preds=completed_best, gts=vis_images_for_best,
                                                            out_path=str(out_path),
                                                            border_kernel_size=int(getattr(config, "BORDER_KERNEL", 7)),
                                                            max_samples=int(getattr(config, "SAVE_VIZ_PER_EPOCH", 8)),
                                                            viz_upscale=config.VIZ_UPSCALE,
                                                            viz_per_row=config.VIZ_PER_ROW,
                                                            viz_save_individual=config.VIZ_SAVE_INDIVIDUAL)
                                    print(f"Saved best visualization -> {out_path}")
                                except Exception as e:
                                    print(f"Warning: failed to export best visualization: {e}")

                                if config.SAVE_BEST_EMA_WEIGHTS:
                                    ema_sd = unwrap_model(generator_ema).state_dict()
                                    wpath = save_weights_only(checkpoint_dir, "best_generator_ema.weights.pth", ema_sd)
                                    print(f"Saved EMA weights-only: {wpath}")

                            hard_es_cfg = getattr(config, 'EARLY_STOP_HARD', None)

                            if hard_es_cfg and hard_es_cfg.get('ENABLED',
                                                               False) and global_step >= config.MIN_STEPS_BEFORE_EARLY_STOP:
                                window = int(hard_es_cfg.get('NO_NEW_BEST_IN_LAST_EVALS', 10))
                                if not new_best:
                                    no_best_counter += 1
                                if no_best_counter >= window and not done:
                                    hard_stop_path = save_full_checkpoint(
                                        checkpoint_dir,
                                        "best_generator_ema.pth",
                                        epoch=epoch + 1,
                                        global_step=global_step,
                                        best_psnr=best_psnr,
                                        generator=generator,
                                        generator_ema=generator_ema,
                                        discriminator=discriminator,
                                        optimizer_g=optimizer_g,
                                        optimizer_d=optimizer_d,
                                        scheduler_g=scheduler_g,
                                        scheduler_d=scheduler_d,
                                        scaler=scaler,
                                        d_stalled_count=d_stalled_count,
                                        Config=config,
                                        last_strong_d_step=last_strong_d_step,
                                        last_probe=last_probe,
                                    )
                                    print(
                                        f"HARD EARLY STOP: No new best in last {window} evals. Saved best EMA -> {hard_stop_path}"
                                    )
                                    if config.SAVE_BEST_EMA_WEIGHTS:
                                        ema_sd = unwrap_model(generator_ema).state_dict()
                                        wpath = save_weights_only(
                                            checkpoint_dir, "best_generator_ema.weights.pth", ema_sd
                                        )
                                        print(f"Saved EMA weights-only: {wpath}")
                                    no_best_counter = 0
                                    done = True

                        if 'gate_means' in locals() and gate_means: log_data.update(gate_means)
                        for key, value in log_data.items():
                            if isinstance(value, (int, float)) and math.isfinite(value):
                                category = "Metrics"
                                if key.startswith("gate/"): category, key = "Gate_Means", key.replace("gate/", "")
                                writer.add_scalar(f"{category}/{key.replace('_', ' ').title()}", value, global_step)
                        if global_step % getattr(config, 'TBOARD_FLUSH_EVERY', 300) == 0: writer.flush()
                        csv_writer.writerow(log_data);
                        csv_file.flush()

                        for _name in ['completed_es', 'vis_images_eval', 'vis_masks_es',
                                      'completed_eval', 'gt_eval', 'masks_eval', 'gout', 'vi', 'vm', 'vmasked']:
                            if _name in locals():
                                try:
                                    del locals()[_name]
                                except Exception:
                                    pass
                        if bool(getattr(config, "EMPTY_CACHE_AFTER_LOG", False)):
                            import gc;
                            gc.collect()
                            if device.type == "cuda":
                                torch.cuda.empty_cache()

                    # ==============================================================
                    # --------- SUB-PART 2.7: CHECKPOINTING & EARLY STOPPING -----
                    # ==============================================================
                    if dist_ctx.is_main and global_step % config.CHECKPOINT_INTERVAL == 0:
                        save_full_checkpoint(checkpoint_dir, "latest.pth", epoch=epoch + 1, global_step=global_step,
                                             best_psnr=best_psnr, generator=generator, generator_ema=generator_ema,
                                             discriminator=discriminator, optimizer_g=optimizer_g,
                                             optimizer_d=optimizer_d, scheduler_g=scheduler_g, scheduler_d=scheduler_d,
                                             scaler=scaler, d_stalled_count=d_stalled_count, Config=config,
                                             last_strong_d_step=last_strong_d_step, last_probe=last_probe)
                    if is_finetune and config.FREEZE_ENCODER_STEPS > 0 and local_step == config.FREEZE_ENCODER_STEPS:
                        print(" thawing generator encoder.")
                        target_generator = unwrap_model(generator)
                        for name, p in target_generator.named_parameters():
                            if 'encoders.' in name or 'downsamples.' in name: p.requires_grad = True

                    early_stop_gate = max(recovery_until_step,
                                          resume_gs + getattr(config, "MIN_STEPS_BEFORE_EARLY_STOP", 0))
                    # === Early Stop (Pareto-style dual gate) ===
                    if (dist_ctx.is_main
                            and hasattr(config, 'EARLY_STOP')
                            and (global_step > early_stop_gate)
                            and (global_step % config.METRICS_INTERVAL == 0)
                            and ("psnr_eval_mask" in log_data)):


                        current_psnr_w = 0.7 * log_data["psnr_eval_mask"] + 0.3 * log_data["psnr_eval_full"]
                        lpips_key = config.EARLY_STOP.get('USE_LPIPS_KEY', "lpips_mask_alex")
                        current_lpips = log_data.get(lpips_key, None)


                        if 'best_psnr_w_for_es' not in locals():
                            best_psnr_w_for_es = current_psnr_w
                            best_lpips_for_es = float("inf") if current_lpips is None else current_lpips
                            early_stop_counter = 0
                            print(
                                f"[ES] Init baselines -> PSNR*: {best_psnr_w_for_es:.3f}, LPIPS*: {best_lpips_for_es:.6f}")


                        delta_db = config.EARLY_STOP.get('DELTA_DB', 0.005)
                        delta_lpips = config.EARLY_STOP.get('DELTA_LPIPS', 0.004)

                        improved_psnr = (current_psnr_w - best_psnr_w_for_es) > delta_db
                        improved_lpips = (current_lpips is not None) and (
                                    (best_lpips_for_es - current_lpips) > delta_lpips)


                        if improved_psnr or improved_lpips:
                            if improved_psnr:  best_psnr_w_for_es = current_psnr_w
                            if improved_lpips: best_lpips_for_es = current_lpips
                            early_stop_counter = 0
                            print(
                                f"[ES] Improvement -> PSNR*: {best_psnr_w_for_es:.3f}, LPIPS*: {best_lpips_for_es:.6f} (Patience Reset)")
                        else:
                            early_stop_counter += 1
                            print(f"[ES] No improvement ({early_stop_counter}/{config.EARLY_STOP.get('PATIENCE', 40)}) "
                                  f"| PSNR_wDelta < {delta_db:.4f}, LPIPSDelta < {delta_lpips:.4f}")


                        if early_stop_counter >= config.EARLY_STOP.get('PATIENCE', 40):
                            print(
                                f"EARLY STOPPING TRIGGERED after {config.EARLY_STOP.get('PATIENCE', 40)} checks with no improvement.")
                            done = True

                    if log_due:
                        control_state = broadcast_object(
                            {
                                "done": bool(done),
                                "recovery_until_step": int(recovery_until_step),
                                "d_stalled_count": int(d_stalled_count),
                                "best_psnr": float(best_psnr),
                                "fsm_eval": fsm_eval_payload,
                            } if dist_ctx.is_main else None,
                            src=0,
                            device=device,
                        )
                        done = bool(control_state["done"])
                        recovery_until_step = int(control_state["recovery_until_step"])
                        d_stalled_count = int(control_state["d_stalled_count"])
                        best_psnr = float(control_state["best_psnr"])
                        fsm_eval_payload = control_state.get("fsm_eval")
                        if fsm_eval_payload is not None and not dist_ctx.is_main:
                            fsm.on_eval(
                                int(fsm_eval_payload["global_step"]),
                                float(fsm_eval_payload["d_gap"]),
                                float(fsm_eval_payload["psnr_eval_full"]),
                            )

                except StopIteration:
                    break
                except Exception as e:
                    print(f"An error occurred at step {global_step}: {e}")
                    import traceback
                    traceback.print_exc()
                    done = True
                    if dist_ctx.launched:
                        raise
                    break

            # ==================================================================
            # ----------------- PART 3: END OF EPOCH ACTIONS -------------------
            # ==================================================================
            if dist_ctx.is_main and hasattr(mask_generator_instance, 'report_and_reset_histogram'):
                mask_generator_instance.report_and_reset_histogram(epoch + 1)
            if dist_ctx.is_main and (epoch + 1) % config.VIS_INTERVAL == 0:
                generator_ema.eval()
                with torch.no_grad(), autocast(device_type=device_type, dtype=amp_dtype, enabled=amp_enabled):
                    vis_masks, _ = mask_generator_instance(vis_images_fixed.shape[0], seed=config.GLOBAL_SEED)
                    vis_masks = vis_masks.to(device)
                    vis_masked_images = vis_images_fixed * (1. - vis_masks)
                    vis_gate_temp = get_scheduled_value(global_step, getattr(config, "GATE_TEMP_SCHEDULE", []), 1.0)
                    generated_ema_vis, _ = generator_ema(torch.cat([vis_masked_images, vis_masks], dim=1),
                                                         config=config, global_step=global_step,
                                                         spectral_dropout_rate=0.0, gate_temp=vis_gate_temp)
                    completed_ema_vis = vis_masked_images + generated_ema_vis * vis_masks
                    vis_path_latest = vis_dir / f"epoch_{(epoch + 1):03d}_latest.png"
                    save_visualization_grid(images=vis_masked_images, masks=vis_masks, preds=completed_ema_vis,
                                            gts=vis_images_fixed, out_path=str(vis_path_latest),
                                            border_kernel_size=int(getattr(config, "SEAM_KERNEL_SIZE", 7)),
                                            max_samples=int(getattr(config, "SAVE_VIZ_PER_EPOCH", 8)),
                                            viz_upscale=int(getattr(config, "VIZ_UPSCALE", 1)),
                                            viz_per_row=int(getattr(config, "VIZ_PER_ROW", 4)),
                                            viz_save_individual=bool(getattr(config, "VIZ_SAVE_INDIVIDUAL", False)))
                generator_ema.train()
            if dist_ctx.is_main and config.SAVE_LATEST_AT_EPOCH_END:
                save_full_checkpoint(checkpoint_dir, "latest.pth", epoch=epoch + 1, global_step=global_step,
                                     best_psnr=best_psnr, generator=generator, generator_ema=generator_ema,
                                     discriminator=discriminator, optimizer_g=optimizer_g, optimizer_d=optimizer_d,
                                     scheduler_g=scheduler_g, scheduler_d=scheduler_d, scaler=scaler,
                                     d_stalled_count=d_stalled_count, Config=config,
                                     last_strong_d_step=last_strong_d_step, last_probe=last_probe)
                print(f" Synced latest.pth at end of epoch {epoch + 1}")
            barrier()

        # ======================================================================
        # ----------------- PART 4: CLEANUP AND EXIT ---------------------------111
        # ======================================================================
        print(f"\n======== TRAINING COMPLETE (RUN MODE: {config.RUN_MODE}) ========")
        total_training_time = time.time() - total_start_time
        print(f"Total training time: {format_time(total_training_time)}")
        writer.close()

    except Exception as e:
        print(f"\n\n A critical error occurred during script execution: {e}")
        import traceback
        traceback.print_exc()
        if dist_ctx.launched:
            raise
    finally:
        if logger: sys.stdout = original_stdout; logger.close()
        if csv_file and not csv_file.closed: csv_file.close(); print(" CSV log file closed.")
        cleanup_distributed()


if __name__ == '__main__':
    main()
