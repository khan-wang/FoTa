"""Run FoTa-Net inference from image and mask file lists."""

import os
import sys
import time
import argparse
from pathlib import Path
from collections.abc import Mapping

import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

def disable_tf32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def load_flist(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Flist not found: {p}")
    with p.open("r", encoding="utf-8-sig") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    return lines

def save_tensor_as_png(img_01_chw: torch.Tensor, save_path: Path):
    """
    img_01_chw: [3,H,W] in [0,1]
    """
    img = (img_01_chw.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(img).save(str(save_path))

def load_generator_state(checkpoint_path: str):
    """Load a raw state dict or select generator weights from a full checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Unsupported checkpoint object: {type(checkpoint).__name__}")

    for key in ("emaG", "state_dict", "netG", "model", "generator"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping) and value:
            return value, key
    return checkpoint, "raw_state_dict"


def main():
    parser = argparse.ArgumentParser(description="FoTa-Net file-list inference")

    parser.add_argument("--config", type=str, required=True, help="Path to a YAML preset")
    parser.add_argument("--run_mode", type=str, default="public", help="Optional config/env preset")
    parser.add_argument("--checkpoint", type=str, required=True, help="FoTa-Net .pth checkpoint")
    parser.add_argument("--image_flist", type=str, required=True, help="Path to image flist")
    parser.add_argument("--mask_flist", type=str, required=True, help="Path to mask flist")
    parser.add_argument("--output_dir", type=str, required=True, help="Dir to save predicted images")

    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--amp", action="store_true", help="Use autocast AMP")
    parser.add_argument("--no_compositing", action="store_true", help="Save raw output (no paste-back)")
    parser.add_argument("--save_input_gt", action="store_true", help="Also save masked input + gt")
    parser.add_argument("--save_ids", type=str, default=None, help="Optional txt: one idx per line, only save those")
    parser.add_argument("--max_images", type=int, default=-1, help="For debug, only process first N images")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup steps for timing")
    parser.add_argument("--disable_tf32", action="store_true", help="Disable TF32 for consistency")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cuda:0 or cpu")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    sys.path.append(str(project_root))

    from config import load_config
    from uformer_pytorch.my_model import DualBranchUformer
    from utils.checkpoint_safety import load_state_dict_report
    try:
        from utils.amp import autocast
    except Exception:
        from torch.cuda.amp import autocast

    if args.disable_tf32:
        disable_tf32()

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    pred_dir = out_root / "pred"
    pred_dir.mkdir(parents=True, exist_ok=True)

    input_dir = out_root / "input"
    gt_dir = out_root / "gt"
    if args.save_input_gt:
        input_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

    save_ids = None
    if args.save_ids is not None:
        save_ids = set()
        with open(args.save_ids, "r", encoding="utf-8-sig") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    save_ids.add(int(ln))

    img_paths = load_flist(args.image_flist)
    mask_paths = load_flist(args.mask_flist)
    if not img_paths:
        raise ValueError("Image file list is empty.")
    if not mask_paths:
        raise ValueError("Mask file list is empty.")
    if args.max_images > 0:
        img_paths = img_paths[:args.max_images]

    img_tf = T.Compose([
        T.Resize((args.img_size, args.img_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
    ])
    mask_tf = T.Compose([
        T.Resize((args.img_size, args.img_size), interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor(),
    ])

    config = load_config(run_mode=args.run_mode, config_path=args.config)

    model = DualBranchUformer(
        img_channel=config.IMG_CHANNEL, out_channel=config.OUT_CHANNEL,
        embed_dim=config.EMBED_DIM, num_blocks=config.NUM_BLOCKS, heads=config.HEADS,
        encoder_blocks=getattr(config, "ENCODER_BLOCKS", None),
        taylor_num_paths_per_stage=getattr(config, "TAYLOR_NUM_PATHS_PER_STAGE", (2, 2, 2, 2)),
        focusing_factor=getattr(config, "FOCUSING_FACTOR", 6),
        fno_stages=config.FNO_STAGES, fno_modes_per_stage=config.FNO_MODES_PER_STAGE,
        fno_channel_bottleneck=getattr(config, 'FNO_CHANNEL_BOTTLENECK', 0.5),
        use_dsdcn=getattr(config, 'USE_DSDCN', False),
        dsdcn_backend=getattr(config, "DSDCN_BACKEND", "auto"),
        dsdcn_mode=getattr(config, "DSDCN_MODE", "compat"),
        dsdcn_clamp=getattr(config, "DSDCN_CLAMP", 1.0),
        use_cmt=getattr(config, 'USE_CMT', True),
        cmt_stages=getattr(config, "CMT_STAGES", (1, 2, 3)),
        cmt_alpha_max=getattr(config, "CMT_ALPHA_MAX", 0.2),
        cmt_warmup_steps=getattr(config, "CMT_WARMUP_STEPS", 2000),
        cmt_shifted=getattr(config, "CMT_SHIFTED", True),
        use_cpe=getattr(config, 'USE_CPE', True)
    ).to(device)

    state, state_key = load_generator_state(args.checkpoint)
    load_state_dict_report(model, state, "inference", strict=False)
    print(f"Loaded checkpoint state: {state_key}")
    model.eval()

    use_cuda_timer = (device.type == "cuda")
    if use_cuda_timer:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()

    total_ms = 0.0
    count_timed = 0

    pbar = tqdm(range(len(img_paths)), desc="Infer(Ours)", unit="img")
    with torch.no_grad():
        for idx in pbar:
            img_path = img_paths[idx]
            mask_path = mask_paths[idx % len(mask_paths)]

            img_pil = Image.open(img_path).convert("RGB")
            mask_pil = Image.open(mask_path).convert("L")
            img = img_tf(img_pil).unsqueeze(0).to(device)            # [1,3,H,W], [0,1]
            mask = mask_tf(mask_pil).unsqueeze(0).to(device)          # [1,1,H,W], [0,1]
            mask = (mask > 0.5).float()

            img_neg = img * 2.0 - 1.0                                 # [-1,1]
            masked_img_neg = img_neg * (1.0 - mask)                   # hole=0
            inp = torch.cat([masked_img_neg, mask], dim=1)            # [1,4,H,W]

            if idx < args.warmup:
                with autocast(enabled=args.amp):
                    out, _ = model(inp, config=config)
                continue

            if use_cuda_timer:
                starter.record()
                with autocast(enabled=args.amp):
                    out, _ = model(inp, config=config)
                ender.record()
                torch.cuda.synchronize()
                step_ms = starter.elapsed_time(ender)
            else:
                t0 = time.time()
                with autocast(enabled=args.amp):
                    out, _ = model(inp, config=config)
                step_ms = (time.time() - t0) * 1000.0

            total_ms += step_ms
            count_timed += 1
            pbar.set_postfix({"PureTime(ms)": f"{step_ms:.2f}", "Avg(ms)": f"{(total_ms/max(count_timed,1)):.2f}"})

            if args.no_compositing:
                completed_neg = out
            else:
                completed_neg = masked_img_neg + out * mask

            completed_01 = ((completed_neg.clamp(-1, 1) + 1.0) / 2.0).squeeze(0)

            if (save_ids is None) or (idx in save_ids):
                fname = f"{idx:06d}.png"
                save_tensor_as_png(completed_01, pred_dir / fname)

                if args.save_input_gt:
                    masked_01 = ((masked_img_neg + 1.0) / 2.0).squeeze(0)
                    save_tensor_as_png(masked_01, input_dir / fname)
                    save_tensor_as_png(img.squeeze(0), gt_dir / fname)

    if count_timed > 0:
        avg_ms = total_ms / count_timed
        print(f"\n[Done] Timed images: {count_timed}")
        print(f"[Stats] Average Pure Inference Time: {avg_ms:.3f} ms @ {args.img_size}x{args.img_size}")
        print(f"[Stats] Saved to: {out_root.resolve()}")

if __name__ == "__main__":
    main()
