"""Fourier neural operator blocks used by FoTa-Net."""

import torch
import torch.nn as nn
import torch.fft
import math

from utils.debug import assert_finite_maybe, sanitize_maybe


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes_height, modes_width):
        super().__init__()
        self.in_channels = in_channels;
        self.out_channels = out_channels
        self.modes_height = modes_height;
        self.modes_width = modes_width
        self.scale = (1 / (in_channels * out_channels))
        self.weights_real = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes_height, self.modes_width))
        self.weights_imag = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, self.modes_height, self.modes_width))

    def compl_mul2d(self, input, weights_real, weights_imag):
        cinput = input.unsqueeze(2);
        weights = torch.complex(weights_real, weights_imag)
        return torch.einsum("bixy,ioxy->boxy", cinput.squeeze(2), weights)

    @staticmethod
    def energy_keep_mask(coeffs: torch.Tensor, spectral_dropout_rate: float) -> torch.Tensor:
        """Return a channel-shared mask that keeps the highest-energy coefficients.

        `spectral_dropout_rate` is the fraction to drop. For example, rate=0.01
        keeps about 99% of the frequency coefficients.
        """
        B, _, H, W = coeffs.shape
        num_coeffs = H * W
        if num_coeffs <= 0:
            return torch.ones(B, 1, H, W, device=coeffs.device, dtype=torch.bool)

        drop_rate = float(spectral_dropout_rate)
        keep_count = int(round(num_coeffs * (1.0 - drop_rate)))
        keep_count = max(1, min(num_coeffs, keep_count))

        energies = (coeffs.real ** 2 + coeffs.imag ** 2).mean(dim=1, keepdim=True)
        flat_energies = energies.reshape(B, 1, num_coeffs)
        keep_idx = flat_energies.topk(keep_count, dim=-1, largest=True, sorted=False).indices
        keep_flat = torch.zeros_like(flat_energies, dtype=torch.bool)
        keep_flat.scatter_(-1, keep_idx, True)
        return keep_flat.view(B, 1, H, W)

    def forward(self, x, spectral_dropout_rate=0.0, config=None, global_step=0):
        B, C, H, W = x.shape

        assert_finite_maybe("FNO_Block/rfft_in", x, config, global_step)
        x_ft = torch.fft.rfft2(x.float(), norm="ortho")
        assert_finite_maybe("FNO_Block/rfft_out", x_ft, config, global_step)

        # Clamp retained modes to the current feature resolution.
        Hf, Wf = x_ft.shape[-2], x_ft.shape[-1]
        modes_h = min(self.modes_height, Hf)
        modes_w = min(self.modes_width, Wf)

        out_ft = torch.zeros(B, self.out_channels, Hf, Wf, device=x.device, dtype=torch.cfloat)

        if modes_h > 0 and modes_w > 0:
            out_ft[:, :, :modes_h, :modes_w] = self.compl_mul2d(
                x_ft[:, :, :modes_h, :modes_w],
                self.weights_real[:, :, :modes_h, :modes_w],
                self.weights_imag[:, :, :modes_h, :modes_w]
            )

            if self.training and spectral_dropout_rate > 0:
                target_coeffs = out_ft[:, :, :modes_h, :modes_w]
                shared_mask = self.energy_keep_mask(target_coeffs, spectral_dropout_rate)
                out_ft[:, :, :modes_h, :modes_w] *= shared_mask.to(dtype=out_ft.dtype)

        x = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
        assert_finite_maybe("FNO_Block/irfft_out", x, config, global_step)

        x = sanitize_maybe(x, config)
        return x


class FNOBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes_height, modes_width):
        super().__init__()
        self.spectral_conv = SpectralConv2d(in_channels, out_channels, modes_height, modes_width)
        # Local high-frequency compensation.
        self.local_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.activation = nn.GELU()

    def forward(self, x, spectral_dropout_rate=0.0, config=None, global_step=0):
        x1 = self.spectral_conv(x, spectral_dropout_rate, config=config, global_step=global_step)
        x2 = self.local_conv(x)
        return self.activation(x1 + x2)

    @staticmethod
    def count_fno(module, x, y):
        B, C, H, W = x[0].shape;
        N = H * W
        fft_macs = (5 * N * math.log2(N)) * B * C
        spec_conv = module.spectral_conv;
        modes = spec_conv.modes_height * spec_conv.modes_width
        comp_mul_macs = (modes * spec_conv.in_channels * spec_conv.out_channels * 4) * B
        local = module.local_conv
        # depthwise conv: in_channels == out_channels, groups == in_channels
        local_macs = local.kernel_size[0] * local.kernel_size[1] * local.out_channels * H * W * B / local.groups
        module.total_ops += torch.DoubleTensor([fft_macs + comp_mul_macs + local_macs])
