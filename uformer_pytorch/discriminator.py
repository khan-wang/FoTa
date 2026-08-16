"""PatchGAN discriminator used by FoTa-Net training."""
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as SN
from typing import List, Tuple

class PatchGANDiscriminator(nn.Module):
    """
    Defines a PatchGAN discriminator using Spectral Normalization (SN).
    Now correctly returns intermediate features when requested.
    """

    def __init__(self, input_nc=4, ndf=64, n_layers=4, return_features=False):
        super().__init__()
        self.return_features = return_features
        self.in_channels = int(input_nc)
        self._actual_input_nc = int(input_nc)

        kw = 4
        padw = (kw - 1) // 2

        self.layers = nn.ModuleList()

        # Initial layer
        self.layers.append(
            nn.Sequential(
                SN(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
                nn.LeakyReLU(0.2, inplace=False)
            )
        )

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            self.layers.append(
                nn.Sequential(
                    SN(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw)),
                    nn.LeakyReLU(0.2, inplace=False)
                )
            )

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.layers.append(
            nn.Sequential(
                SN(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw)),
                nn.LeakyReLU(0.2, inplace=False)
            )
        )

        # Final layer
        self.final_conv = SN(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))
    def forward(self, x) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        The forward pass now extracts features from intermediate layers if
        `self.return_features` is True.
        """
        features = []
        feat_x = x
        for layer in self.layers:
            feat_x = layer(feat_x)
            if self.return_features:
                features.append(feat_x)

        out = self.final_conv(feat_x)

        return out, features
