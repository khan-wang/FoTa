# Third-party notices

FoTa-Net is distributed under the Apache License 2.0. The repository also
contains or derives limited implementation patterns from the projects below.

## LaMa

- Project: Resolution-robust Large Mask Inpainting with Fourier Convolutions
- Source: https://github.com/advimman/lama
- License: Apache License 2.0
- Use in FoTa-Net: high-receptive-field perceptual-loss encoder conventions and
  large-mask training conventions.

## ZITS++

- Project: ZITS++: Image Inpainting by Improving the Incremental Transformer on
  Structural Priors
- Source: https://github.com/ewrfcas/ZITS-PlusPlus
- License: Apache License 2.0
- Use in FoTa-Net: mask-bank organization and sampling conventions.

## Uformer

- Project: Uformer: A General U-Shaped Transformer for Image Restoration
- Source: https://github.com/ZhendongWang6/Uformer
- License: MIT License
- Use in FoTa-Net: parts of the U-shaped encoder-decoder scaffold and utility
  organization.

## Taylor-expanded attention reference

The Taylor-expanded attention in this repository is a project implementation
of the mathematical formulation described by MB-TaylorFormerV2. No upstream
repository license was available during the release audit, so this repository
does not claim or redistribute that upstream code under an assumed license.

Users must also comply with the licenses of datasets, pretrained feature
extractors, Python packages, and model files that they download separately.

