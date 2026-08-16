"""Visualization and boundary-aware evaluation utilities."""

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.util import img_as_float32
import numpy as np
from pathlib import Path

# ==============================================================================

# ==============================================================================

def _inner_outer_bands(mask: torch.Tensor, k: int = 7):
    """Return inner and outer mask bands for a mask where one denotes a hole."""
    pad = k // 2

    dil = F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)
    ero = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=k, stride=1, padding=pad)

    outer = (dil - mask).clamp(0, 1)
    inner = (mask - ero).clamp(0, 1)
    return inner, outer


def sobel_magnitude(x: torch.Tensor):
    """Compute the per-channel Sobel gradient magnitude."""
    # Sobel kernels
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)

    kx = kx.repeat(x.size(1), 1, 1, 1)
    ky = ky.repeat(x.size(1), 1, 1, 1)


    gx = F.conv2d(x, kx, padding=1, groups=x.size(1))
    gy = F.conv2d(x, ky, padding=1, groups=x.size(1))

    return torch.sqrt(gx * gx + gy * gy + 1e-6)


_sobel_mag = sobel_magnitude


class SeamConsistencyLoss:
    """Measure pixel and optional gradient consistency in the inner seam band."""

    def __init__(self, kernel_size: int = 7, inner_weight: float = 1.0, grad_weight: float = 0.5,
                 use_grad: bool = True):
        self.k = int(kernel_size)
        self.wi = float(inner_weight)
        self.wg = float(grad_weight)
        self.use_grad = bool(use_grad)

    def __call__(self, completed: torch.Tensor, gt: torch.Tensor, images_bg: torch.Tensor, mask: torch.Tensor):

        inner, outer = _inner_outer_bands(mask, self.k)
        C = completed.size(1)
        denom = lambda band: (band.sum() * C + 1e-8)


        inner_l1 = (torch.abs(completed - gt) * inner).sum() / denom(inner)


        if self.use_grad:
            band = inner
            pred_grad = sobel_magnitude(completed)
            gt_grad = sobel_magnitude(gt)
            grad_l1 = (torch.abs(pred_grad - gt_grad) * band).sum() / denom(band)
        else:
            pred_grad = None
            gt_grad = None
            grad_l1 = completed.new_tensor(0.0)


        seam_loss = self.wi * inner_l1 + self.wg * grad_l1

        return {
            "outer_l1": completed.new_tensor(0.0),
            "inner_l1": inner_l1,
            "grad_l1": grad_l1,
            "seam_loss": seam_loss,
            "inner_px": 0,
            "outer_px": 0,
            "_pred_grad": pred_grad,
            "_gt_grad": gt_grad,
        }


# ==============================================================================

# ==============================================================================

def denorm_minus1_1_to_0_1(x):
    """Denormalizes a tensor from [-1, 1] to [0, 1]"""
    return (x.clamp(-1, 1) + 1) * 0.5


@torch.no_grad()
def save_visualization_grid(
        images, masks, preds, gts, out_path, border_kernel_size,
        max_samples=16, pad=2,

        viz_upscale: int = 1,
        viz_per_row: int = 4,
        viz_save_individual: bool = False
):
    """
    Saves a grid of visualization images with enhanced options for clarity.
    Layout per sample strip: [input | border_band | prediction | ground_truth]
    """

    out_path = Path(out_path).with_suffix('.png')

    B = min(max_samples, images.size(0))

    imgs = denorm_minus1_1_to_0_1(images[:B].cpu())
    prds = denorm_minus1_1_to_0_1(preds[:B].cpu())
    gtru = denorm_minus1_1_to_0_1(gts[:B].cpu())
    msks = masks[:B].cpu()

    dilated_mask = F.max_pool2d(msks, border_kernel_size, 1, border_kernel_size // 2)
    border_band = (dilated_mask - msks).clamp(0, 1).expand(-1, 3, -1, -1)


    all_strips = []
    for i in range(B):

        strip_i = torch.stack([
            imgs[i],
            border_band[i],
            prds[i],
            gtru[i]
        ], dim=0)


        if viz_upscale > 1:
            strip_i = F.interpolate(strip_i, scale_factor=viz_upscale, mode='nearest')

        all_strips.append(strip_i)


        if viz_save_individual:
            individual_dir = out_path.parent / (out_path.stem + "_individual")
            individual_dir.mkdir(exist_ok=True)
            individual_path = individual_dir / f"sample_{i:02d}.png"

            individual_grid = make_grid(strip_i, nrow=4, padding=pad, pad_value=1)
            save_image(individual_grid, individual_path)


    tiles = torch.cat(all_strips, dim=0)


    final_nrow = viz_per_row * 4

    grid = make_grid(tiles, nrow=final_nrow, padding=pad, pad_value=1)
    save_image(grid, out_path)

    if viz_save_individual:
        print(f" Visualization grid saved to '{out_path}' (and individual samples in '{out_path.stem}_individual/')")
    else:
        print(f" Visualization grid saved to '{out_path}'")


def _bbox_from_mask(m, pad=3):
    ys, xs = np.where(m > 0.5)
    if len(xs) == 0 or len(ys) == 0: return None
    y0, y1 = max(0, ys.min() - pad), min(m.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(m.shape[1], xs.max() + 1 + pad)
    return y0, y1, x0, x1


@torch.no_grad()
def calculate_masked_metrics(preds, gts, masks):
    """
    Calculates both full-image and masked metrics (PSNR/SSIM).
    Returns: A tuple (psnr_full, ssim_full, psnr_mask, ssim_mask)
    """
    P = denorm_minus1_1_to_0_1(preds.cpu()).numpy().transpose(0, 2, 3, 1)
    T = denorm_minus1_1_to_0_1(gts.cpu()).numpy().transpose(0, 2, 3, 1)
    M = masks.cpu().numpy()[:, 0]

    ps_full, ss_full, ps_mask, ss_mask, cnt = 0.0, 0.0, 0.0, 0.0, 0
    for p_img, t_img, m_img in zip(P, T, M):
        p_float = img_as_float32(p_img)
        t_float = img_as_float32(t_img)

        # --- Full Image Metrics ---
        mse_full = np.mean((p_float - t_float) ** 2)
        ps_full += 10 * np.log10(1.0 / (mse_full + 1e-9))

        win_size = min(7, p_float.shape[0], p_float.shape[1])
        if win_size % 2 == 0: win_size -= 1
        if win_size >= 7:
            ss_full += ssim(t_float, p_float, data_range=1.0, channel_axis=-1, win_size=win_size)

        # --- Masked Metrics ---
        diff_mask = (p_float - t_float)[m_img > 0.5]
        if diff_mask.size == 0:
            ps_mask += 10 * np.log10(1.0 / 1e-9)  # If no mask, mask PSNR is effectively infinite
            ss_mask += 1.0  # If no mask, SSIM is perfect
        else:
            mse_mask = np.mean(diff_mask ** 2)
            ps_mask += 10 * np.log10(1.0 / (mse_mask + 1e-9))

            box = _bbox_from_mask(m_img)
            if box:
                y0, y1, x0, x1 = box
                p_box, t_box = p_float[y0:y1, x0:x1], t_float[y0:y1, x0:x1]
                h_box, w_box, _ = p_box.shape
                win_size_mask = min(7, h_box, w_box)
                if win_size_mask % 2 == 0: win_size_mask -= 1
                if win_size_mask >= 7:
                    ss_mask += ssim(t_box, p_box, data_range=1.0, channel_axis=-1, win_size=win_size_mask)

        cnt += 1

    return (ps_full / cnt, ss_full / cnt, ps_mask / cnt, ss_mask / cnt) if cnt > 0 else (0.0, 0.0, 0.0, 0.0)
