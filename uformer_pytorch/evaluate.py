




import torch
import torch.nn.functional as F
import os
import argparse
import hashlib
import sys
import time
import shutil
import tempfile
import numpy as np
import pandas as pd
from collections.abc import Mapping
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from skimage.metrics import structural_similarity
from PIL import Image

# ==============================================================================

# ==============================================================================
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

if str(project_root) not in sys.path:
    print(f"[Init] Adding project root to sys.path: {project_root}")
    sys.path.append(str(project_root))


try:
    from config import load_config
except ImportError:
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from config import load_config


try:
    from uformer_pytorch.my_model import DualBranchUformer
    from utils.amp import autocast
    from utils.checkpoint_safety import load_state_dict_report

    try:
        from complexity_utils import profile_and_report_complexity
    except ImportError:
        try:
            from utils.complexity_utils import profile_and_report_complexity
        except ImportError:
            profile_and_report_complexity = None

except ImportError as e:
    print(f"[Warning] Model imports failed ({e}). This is OK if you are only evaluating 'folder' mode.")
    DualBranchUformer = None


    from torch.cuda.amp import autocast as _torch_autocast



    def autocast(enabled=True):
        return _torch_autocast(enabled=enabled)


    profile_and_report_complexity = None
    load_state_dict_report = None


try:
    from baselines.wrappers import HINTWrapper
except ImportError:
    baselines_path = project_root / 'baselines'
    if str(baselines_path) not in sys.path:
        sys.path.append(str(baselines_path))
    try:
        from wrappers import HINTWrapper
    except ImportError:
        HINTWrapper = None


try:
    import lpips

    _LPIPS_AVAILABLE = True
except ImportError:
    lpips = None
    _LPIPS_AVAILABLE = False
    print("WARNING: lpips library not found.")


try:
    from torch_fidelity import calculate_metrics

    _FID_AVAILABLE = True
except ImportError:
    calculate_metrics = None
    _FID_AVAILABLE = False
    print("WARNING: torch-fidelity not found. FID disabled.")


# ==============================================================================

# ==============================================================================

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def torch_load_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def select_generator_state(raw_state, preferred='auto'):
    meta = {
        'checkpoint_format': '',
        'checkpoint_global_step': '',
        'checkpoint_epoch': '',
        'checkpoint_state_key': '',
    }
    if isinstance(raw_state, Mapping):
        meta['checkpoint_format'] = str(raw_state.get('format', 'state_dict'))
        meta['checkpoint_global_step'] = raw_state.get('global_step', '')
        meta['checkpoint_epoch'] = raw_state.get('epoch', '')

        if preferred != 'auto':
            if preferred not in raw_state:
                raise KeyError(f"Requested checkpoint state '{preferred}' not found. Available keys: {list(raw_state.keys())[:30]}")
            meta['checkpoint_state_key'] = preferred
            return raw_state[preferred], meta

        for key in ('emaG', 'netG', 'state_dict', 'model', 'G', 'generator'):
            if key in raw_state and isinstance(raw_state[key], Mapping):
                meta['checkpoint_state_key'] = key
                return raw_state[key], meta

        if raw_state and all(torch.is_tensor(v) for v in raw_state.values()):
            meta['checkpoint_state_key'] = 'weights_only'
            return raw_state, meta

    raise ValueError("Could not find a generator state_dict in checkpoint.")


def validate_state_match(model, state_dict, min_fraction=0.80):
    model_state = model.state_dict()
    matched = 0
    for key, value in state_dict.items():
        target = model_state.get(key)
        if target is not None and hasattr(value, 'shape') and value.shape == target.shape:
            matched += 1
    fraction = matched / max(1, len(model_state))
    if fraction < min_fraction:
        raise RuntimeError(
            f"Checkpoint/model tensor match too low: {matched}/{len(model_state)} "
            f"({fraction:.1%}). Refusing silent partial load."
        )
    print(f"[ckpt:evaluate_model] matched tensors: {matched}/{len(model_state)} ({fraction:.1%})")

class PairedFlistDataset(Dataset):
    def __init__(self, img_flist_path, mask_flist_path, img_size, max_images=0, stride=1):
        self.img_size = img_size
        try:
            with open(img_flist_path, 'r', encoding='utf-8-sig') as f:
                self.img_paths = [Path(line.strip()) for line in f if line.strip()]
            with open(mask_flist_path, 'r', encoding='utf-8-sig') as f:
                self.mask_paths = [Path(line.strip()) for line in f if line.strip()]
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Flist not found: {e}")

        if stride > 1: self.img_paths = self.img_paths[::stride]
        if max_images > 0: self.img_paths = self.img_paths[:max_images]

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        print(f" Dataset initialized: {len(self.img_paths)} pairs. (Resize to {img_size}x{img_size})")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.img_paths[idx]

            mask_path = self.mask_paths[idx % len(self.mask_paths)]

            img = self.img_transform(Image.open(img_path).convert('RGB'))
            mask = self.mask_transform(Image.open(mask_path).convert('L'))


            return img, mask, str(img_path.resolve()), str(mask_path.resolve())
        except Exception as e:
            print(f"Error loading index {idx}: {e}")

            return torch.zeros(3, self.img_size, self.img_size), \
                torch.zeros(1, self.img_size, self.img_size), "err", "err"


class ImageFolderEvaluator(torch.nn.Module):
    """Evaluate an external prediction folder by basename or numeric ID."""

    def __init__(self, pred_dir, img_ext=None, name_mode='basename'):
        super().__init__()
        self.pred_dir = Path(pred_dir)
        self.img_ext = img_ext
        self.name_mode = name_mode
        print(f" Mode: Folder Evaluation from {self.pred_dir} (Match by: {self.name_mode})")

    def forward(self, inp, config=None, img_paths=None, indices=None):


        device = inp.device
        outs = []


        if self.name_mode == 'id' and indices is None:
            raise ValueError("ImageFolderEvaluator in 'id' mode requires 'indices' input!")
        if self.name_mode == 'basename' and img_paths is None:
            raise ValueError("ImageFolderEvaluator in 'basename' mode requires 'img_paths' input!")

        batch_size = len(indices) if indices is not None else len(img_paths)

        for i in range(batch_size):

            if self.name_mode == 'id':

                fname = f"{indices[i]:06d}"
                if self.img_ext:
                    fname += self.img_ext
                else:
                    fname += ".png"
            else:

                p = img_paths[i]
                if self.img_ext:
                    fname = Path(p).stem + self.img_ext
                else:
                    fname = Path(p).name

            pred_path = self.pred_dir / fname


            if not pred_path.exists():

                # print(f" Warning: Missing {fname}")
                pred = torch.zeros(3, inp.shape[2], inp.shape[3])
            else:
                try:
                    img = Image.open(pred_path).convert('RGB')
                    pred = transforms.ToTensor()(img)  # [3, H, W], 0~1


                    if pred.shape[-2:] != inp.shape[-2:]:
                        pred = F.interpolate(pred.unsqueeze(0), size=inp.shape[-2:],
                                             mode='bilinear', align_corners=False).squeeze(0)
                except Exception as e:
                    print(f"Error reading {pred_path}: {e}")
                    pred = torch.zeros(3, inp.shape[2], inp.shape[3])


            pred = (pred * 2.0) - 1.0
            outs.append(pred)

        return torch.stack(outs).to(device), None


def _bbox_from_mask_np(m_np):
    ys, xs = np.where(m_np > 0.5)
    if len(xs) == 0: return None
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def _inner_outer_bands(mask_tensor, k=33):
    pad = k // 2
    dil = F.max_pool2d(mask_tensor, kernel_size=k, stride=1, padding=pad)
    ero = 1.0 - F.max_pool2d(1.0 - mask_tensor, kernel_size=k, stride=1, padding=pad)
    return (mask_tensor - ero).clamp(0, 1).cpu().numpy(), (dil - mask_tensor).clamp(0, 1).cpu().numpy()


def calculate_base_metrics_gpu(restored_bchw, gt_bchw, mask_bchw):
    B, C, H, W = gt_bchw.shape
    # L1 Loss (MAE) * 100
    l1_map = torch.abs(restored_bchw - gt_bchw)
    l1_full = l1_map.mean(dim=(1, 2, 3)) * 100.0
    mask_sum = mask_bchw.sum(dim=(1, 2, 3)).clamp(min=1e-6)
    l1_mask = ((l1_map * mask_bchw).sum(dim=(1, 2, 3)) / mask_sum) * 100.0

    # PSNR
    mse_full = ((restored_bchw - gt_bchw) ** 2).mean(dim=(1, 2, 3))
    psnr_full = 10.0 * torch.log10(1.0 / (mse_full + 1e-9))
    diff_sq_mask = ((restored_bchw - gt_bchw) ** 2) * mask_bchw
    mse_mask = diff_sq_mask.sum(dim=(1, 2, 3)) / mask_sum
    psnr_mask = 10.0 * torch.log10(1.0 / (mse_mask + 1e-9))

    # SSIM
    ssim_full_list, ssim_mask_list = [], []
    r_np = restored_bchw.cpu().numpy().transpose(0, 2, 3, 1)
    g_np = gt_bchw.cpu().numpy().transpose(0, 2, 3, 1)
    m_np = mask_bchw.cpu().numpy().transpose(0, 2, 3, 1)[:, :, :, 0]

    win_size = min(7, H, W)
    if win_size % 2 == 0: win_size -= 1

    for i in range(B):
        val = structural_similarity(g_np[i], r_np[i], data_range=1.0, channel_axis=-1, win_size=win_size)
        ssim_full_list.append(val)
        box = _bbox_from_mask_np(m_np[i])
        if box:
            y0, y1, x0, x1 = box
            h_b, w_b = y1 - y0, x1 - x0
            if h_b >= 7 and w_b >= 7:
                val_m = structural_similarity(g_np[i, y0:y1, x0:x1], r_np[i, y0:y1, x0:x1],
                                              data_range=1.0, channel_axis=-1, win_size=7)
                ssim_mask_list.append(val_m)
            else:
                ssim_mask_list.append(val)
        else:
            ssim_mask_list.append(val)

    return (psnr_full.tolist(), ssim_full_list, l1_full.tolist(),
            psnr_mask.tolist(), ssim_mask_list, l1_mask.tolist())


def calculate_advanced_metrics(restored, gt, mask, lpips_vgg, lpips_alex, config):
    res_in = restored.unsqueeze(0).clamp(-1, 1)
    gt_in = gt.unsqueeze(0).clamp(-1, 1)
    mask_in = mask.unsqueeze(0)

    l_alex_m, l_alex_f = 0, 0

    with torch.no_grad():
        if lpips_alex:
            map_a = lpips_alex(res_in, gt_in)
            l_alex_f = map_a.mean().item()
            map_a_m = F.interpolate(map_a, size=mask_in.shape[-2:], mode='bilinear')
            l_alex_m = (map_a_m * mask_in).sum() / mask_in.sum().clamp(min=1e-6)
            l_alex_m = l_alex_m.item()

    # Border SSIM
    k = getattr(config, 'BORDER_KERNEL', 33)
    inner, outer = _inner_outer_bands(mask_in, k)
    border_mask = inner[0, 0] + outer[0, 0]
    box = _bbox_from_mask_np(border_mask)
    ssim_border = 1.0
    if box:
        y0, y1, x0, x1 = box
        if (y1 - y0) >= 7 and (x1 - x0) >= 7:
            g_np = (gt.cpu().numpy().transpose(1, 2, 0) + 1) / 2
            r_np = (restored.cpu().numpy().transpose(1, 2, 0) + 1) / 2
            ssim_border = structural_similarity(
                g_np[y0:y1, x0:x1], r_np[y0:y1, x0:x1],
                data_range=1.0, channel_axis=-1, win_size=7
            )

    return 0, l_alex_m, ssim_border, 0, l_alex_f


def benchmark_inference_speed(model, device, config, img_size, use_amp=False, amp_dtype=None, runs=50, warmup=20):
    """Measure synchronized single-image latency at the requested resolution."""
    print(
        f"  Benchmarking Speed (Resolution={img_size}x{img_size}, BS=1, "
        f"AMP={use_amp}, dtype={amp_dtype}, {runs} runs)..."
    )
    model.eval()


    dummy_input = torch.randn(1, 4, img_size, img_size).to(device)


    if isinstance(model, ImageFolderEvaluator):
        print("   (Folder mode detected, skipping speed test)")
        return 0.0

    with torch.no_grad():
        for _ in range(warmup):
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out, _ = model(dummy_input, config=config)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()
    with torch.no_grad():
        for _ in range(runs):
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out, _ = model(dummy_input, config=config)
    end_event.record()
    torch.cuda.synchronize()

    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / runs
    print(f"  Avg Time: {avg_time_ms:.2f} ms")
    return avg_time_ms


def get_mask_bins(bin_edges):
    return list(zip(bin_edges[:-1], bin_edges[1:]))


def bin_label(ratio, bins):
    for lo, hi in bins:
        if lo <= ratio < hi: return f"{int(lo * 100)}-{int(hi * 100)}%"
    if ratio >= bins[-1][0]: return f"{int(bins[-1][0] * 100)}-{int(bins[-1][1] * 100)}%"
    return "out"


def main():
    parser = argparse.ArgumentParser(description="Evaluate FoTa-Net checkpoints or result folders")
    parser.add_argument('--run_mode', type=str, default='public')
    parser.add_argument('--config', type=str, default=None)

    parser.add_argument('--model_type', type=str, default='ours', choices=['ours', 'hint', 'folder'])
    parser.add_argument('--folder_path', type=str, default=None, help="Folder containing external predictions")
    parser.add_argument('--folder_ext', type=str, default=None, help="Optional prediction extension, e.g. .png")
    parser.add_argument('--folder_name_mode', type=str, default='basename', choices=['basename', 'id'],
                        help="Match predictions by source basename or zero-padded numeric id")

    parser.add_argument('--hint_root', type=str, default=str(current_dir / 'hint'))
    parser.add_argument('--checkpoint', type=str, nargs='+', default=[])
    parser.add_argument('--checkpoint_state', type=str, default='auto',
                        choices=['auto', 'emaG', 'netG', 'state_dict', 'model', 'G', 'generator'],
                        help="For full checkpoints, choose generator weights. auto prefers emaG then netG.")

    parser.add_argument('--eval_manifest', type=str, required=True)
    parser.add_argument('--mask_flist_eval', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--img_size', type=int, default=256, help="Evaluation image resolution (default: 256)")
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--compute_fid', action='store_true')
    parser.add_argument('--no_compositing', action='store_true',
                        help="Do NOT composite output with GT background (Raw Output)")
    parser.add_argument('--bins', type=float, nargs='+', default=[0.0, 0.2, 0.4, 0.6001])

    args = parser.parse_args()
    config = load_config(run_mode=args.run_mode, config_path=args.config)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.DEVICE)

    amp_enabled = bool(args.amp)
    amp_dtype = None
    if amp_enabled:
        amp_mode = str(getattr(config, 'AMP_DTYPE', 'fp16')).lower()
        if amp_mode in ('fp32', 'float32', 'none', 'off', 'disable'):
            amp_enabled = False
            print("Eval AMP disabled by AMP_DTYPE; using fp32.")
        elif amp_mode in ('bf16', 'bfloat16') and device.type == 'cuda' and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            print("Eval AMP dtype: bfloat16 (from config AMP_DTYPE).")
        else:
            amp_dtype = torch.float16
            if amp_mode in ('bf16', 'bfloat16'):
                print(" Eval AMP requested bf16, but bf16 is unavailable; falling back to fp16.")
            else:
                print("Eval AMP dtype: float16.")

    # LPIPS
    lpips_alex = None
    if _LPIPS_AVAILABLE:
        try:
            lpips_alex = lpips.LPIPS(net='alex', spatial=True).to(device).eval()
        except:
            pass

    val_dataset = PairedFlistDataset(args.eval_manifest, args.mask_flist_eval, args.img_size)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    MASK_BINS = get_mask_bins(args.bins)

    temp_fid_dir = tempfile.TemporaryDirectory()
    temp_root = Path(temp_fid_dir.name)
    path_real_all = temp_root / 'real_all'
    if args.compute_fid and _FID_AVAILABLE:
        path_real_all.mkdir(parents=True, exist_ok=True)

    fid_fake_dirs = {}
    if args.compute_fid and _FID_AVAILABLE:
        for b in MASK_BINS:
            b_lbl = bin_label(b[0], MASK_BINS)
            fid_fake_dirs[b_lbl] = temp_root / 'fake' / b_lbl
            fid_fake_dirs[b_lbl].mkdir(parents=True, exist_ok=True)

    loop_targets = [args.folder_path] if args.model_type == 'folder' else args.checkpoint
    if not loop_targets:
        print("Error: provide --checkpoint or --folder_path.")
        sys.exit(1)

    for target_path in loop_targets:
        try:
            if args.model_type == 'folder':
                model_name = "Ext_" + Path(target_path).name
            else:
                model_name = Path(target_path).stem

            print(f"\n{'=' * 40}\nProcessing: {model_name} @ {args.img_size}x{args.img_size}\n{'=' * 40}")

            ckpt_meta = {
                'checkpoint_format': '',
                'checkpoint_global_step': '',
                'checkpoint_epoch': '',
                'checkpoint_state_key': '',
            }
            ckpt_sha256 = ''
            if args.model_type == 'folder':
                if not args.folder_path:
                    raise ValueError("--folder_path is required for folder evaluation.")
                model = ImageFolderEvaluator(args.folder_path, args.folder_ext, args.folder_name_mode).to(device)
            elif args.model_type == 'hint':
                model = HINTWrapper(args.hint_root, target_path).to(device)
                model_name = "HINT_" + model_name
            elif args.model_type == 'ours':
                model = DualBranchUformer(
                    img_channel=config.IMG_CHANNEL, out_channel=config.OUT_CHANNEL,
                    embed_dim=config.EMBED_DIM, num_blocks=config.NUM_BLOCKS, heads=config.HEADS,
                    encoder_blocks=getattr(config, 'ENCODER_BLOCKS', None),
                    taylor_num_paths_per_stage=getattr(config, 'TAYLOR_NUM_PATHS_PER_STAGE', (2, 2, 2, 2)),
                    focusing_factor=getattr(config, 'FOCUSING_FACTOR', 6),
                    fno_stages=config.FNO_STAGES, fno_modes_per_stage=config.FNO_MODES_PER_STAGE,
                    fno_channel_bottleneck=getattr(config, 'FNO_CHANNEL_BOTTLENECK', 0.5),
                    use_dsdcn=getattr(config, 'USE_DSDCN', False),
                    dsdcn_backend=getattr(config, 'DSDCN_BACKEND', 'auto'),
                    dsdcn_mode=getattr(config, 'DSDCN_MODE', 'compat'),
                    dsdcn_clamp=getattr(config, 'DSDCN_CLAMP', 1.0),
                    use_cmt=getattr(config, 'USE_CMT', True),
                    cmt_stages=getattr(config, 'CMT_STAGES', (1, 2, 3)),
                    cmt_alpha_max=getattr(config, 'CMT_ALPHA_MAX', 0.2),
                    cmt_warmup_steps=getattr(config, 'CMT_WARMUP_STEPS', 2000),
                    cmt_shifted=getattr(config, 'CMT_SHIFTED', True),
                    use_cpe=getattr(config, 'USE_CPE', True)
                ).to(device)
                raw_state = torch_load_checkpoint(target_path)
                state, ckpt_meta = select_generator_state(raw_state, args.checkpoint_state)
                validate_state_match(model, state)
                if load_state_dict_report is not None:
                    load_state_dict_report(model, state, "evaluate_model", strict=False)
                else:
                    model.load_state_dict(state, strict=False)
                ckpt_sha256 = sha256_file(target_path)
                print(
                    f"[ckpt:evaluate_model] path={target_path} sha256={ckpt_sha256} "
                    f"state_key={ckpt_meta['checkpoint_state_key']} "
                    f"global_step={ckpt_meta['checkpoint_global_step']}"
                )
                model_name = "Ours_" + model_name

            if not isinstance(model, ImageFolderEvaluator):
                model.eval()

            avg_time_ms = benchmark_inference_speed(
                model, device, config=config, img_size=args.img_size,
                use_amp=amp_enabled, amp_dtype=amp_dtype
            )


            if args.model_type == 'ours' and profile_and_report_complexity:
                try:
                    print(f"Profiling complexity for {args.img_size}x{args.img_size} input...")
                    profile_and_report_complexity(
                        model, config, input_shape=(1, 4, args.img_size, args.img_size)
                    )
                except Exception as e:
                    print(f"Complexity profile failed: {e}")
                    pass


            results = []
            sample_counter = 0


            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Eval")):
                img, mask, img_p, mask_p = batch
                img, mask = img.to(device), mask.to(device)

                img_neg = img * 2.0 - 1.0
                masked_img_neg = img_neg * (1. - mask)
                inp = torch.cat([masked_img_neg, mask], dim=1)


                B_size = len(img_p)
                batch_indices = [sample_counter + j for j in range(B_size)]

                with torch.no_grad(), autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):

                    if args.model_type == 'folder':
                        out, _ = model(inp, config=config, img_paths=img_p, indices=batch_indices)
                    else:
                        out, _ = model(inp, config=config)


                    if args.no_compositing:
                        completed_neg = out
                    else:
                        completed_neg = masked_img_neg + out * mask

                completed_neg = completed_neg.clamp(-1, 1)
                completed_01 = (completed_neg + 1) / 2.0

                pf, sf, l1f, pm, sm, l1m = calculate_base_metrics_gpu(completed_01, img, mask)

                for j in range(B_size):
                    global_idx = batch_indices[j]
                    ratio = mask[j].mean().item()
                    b_str = bin_label(ratio, MASK_BINS)

                    _, lm_alex, ssim_b, _, lf_alex = calculate_advanced_metrics(
                        completed_neg[j], img_neg[j], mask[j], None, lpips_alex, config
                    )


                    results.append({
                        'id': global_idx,
                        'img_name': Path(img_p[j]).name,
                        'mask_name': Path(mask_p[j]).name,
                        'img_path': str(img_p[j]),
                        'mask_path': str(mask_p[j]),
                        'mask_ratio': ratio,
                        'bin': b_str,
                        'psnr_full': pf[j], 'ssim_full': sf[j], 'l1_full': l1f[j], 'lpips_full': lf_alex,
                        'psnr_mask': pm[j], 'ssim_mask': sm[j], 'l1_mask': l1m[j], 'lpips_mask': lm_alex,
                        'ssim_border': ssim_b,
                        'checkpoint_path': str(Path(target_path).resolve()) if args.model_type == 'ours' else '',
                        'checkpoint_sha256': ckpt_sha256,
                        'checkpoint_state_key': ckpt_meta['checkpoint_state_key'],
                        'checkpoint_global_step': ckpt_meta['checkpoint_global_step'],
                        'checkpoint_epoch': ckpt_meta['checkpoint_epoch'],
                    })


                    if args.compute_fid and _FID_AVAILABLE:
                        fname = f"{global_idx:06d}_{Path(img_p[j]).name}"

                        save_image(img[j], path_real_all / fname)

                        if b_str in fid_fake_dirs:
                            save_image(completed_01[j], fid_fake_dirs[b_str] / fname)

                sample_counter += B_size


            df = pd.DataFrame(results)

            fid_scores = {}
            fid_Ns = {}
            if args.compute_fid and _FID_AVAILABLE:
                print(">>> Calculating FID (Real_All vs Fake_Bin)...")
                for b_str, fake_path in fid_fake_dirs.items():
                    n_samples = len(list(fake_path.glob('*')))
                    fid_Ns[b_str] = n_samples
                    if n_samples > 10:
                        try:
                            val = calculate_metrics(input1=str(path_real_all), input2=str(fake_path),
                                                    cuda=True, fid=True, verbose=False)['frechet_inception_distance']
                            fid_scores[b_str] = val
                        except Exception as e:
                            print(f" FID Error ({b_str}): {e}")
                            fid_scores[b_str] = float('nan')
                    else:
                        fid_scores[b_str] = float('nan')


            summary = df.drop(columns=['id'], errors='ignore').groupby('bin').mean(numeric_only=True)

            summary['fid'] = [fid_scores.get(b, float('nan')) for b in summary.index]
            summary['N_samples'] = [fid_Ns.get(b, 0) for b in summary.index]
            summary['time_ms'] = avg_time_ms


            print("\n" + "=" * 95)
            print(f"   Final Results: {model_name} | Res: {args.img_size}x{args.img_size} | Time: {avg_time_ms:.2f}ms")
            print("=" * 95)
            print(f"{'Bin':<10} | {'PSNR':<8} | {'SSIM':<8} | {'L1(%)':<8} | {'LPIPS':<8} | {'FID':<8} | {'N':<5}")
            print("-" * 95)

            target_bins = ['0-20%', '20-40%', '40-60%']
            for b_str in target_bins:
                if b_str in summary.index:
                    r = summary.loc[b_str]
                    n_samp = int(r.get('N_samples', 0)) if args.compute_fid else '-'
                    fid_val = r['fid']
                    print(
                        f"{b_str:<10} | {r['psnr_full']:8.4f} | {r['ssim_full']:8.4f} | {r['l1_full']:8.4f} | {r['lpips_full']:8.4f} | {fid_val:8.4f} | {n_samp:<5}")
                else:
                    print(f"{b_str:<10} |   N/A    |   N/A    |   N/A    |   N/A    |   N/A    |  0  ")
            print("-" * 95)


            df.to_csv(out_dir / f"detailed_{model_name}_res{args.img_size}.csv", index=False)
            summary.to_csv(out_dir / f"summary_{model_name}_res{args.img_size}.csv")
            print(f" Saved to {out_dir}")

        except Exception as e:
            print(f" Error processing {target_path}: {e}")
            import traceback
            traceback.print_exc()

    temp_fid_dir.cleanup()


if __name__ == '__main__':
    main()
