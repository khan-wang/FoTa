# Dataset preparation

FoTa-Net does not redistribute Places2, CelebA-HQ, or external mask datasets.
Download each dataset from its official source and comply with its terms.

## Image folders

Training uses `torchvision.datasets.ImageFolder`, so every image must be below
at least one class directory. A single placeholder class is sufficient for
CelebA-HQ.

```text
data/
  places2/
    train/
      airport_terminal/
        *.jpg
      bedroom/
        *.jpg
  celeba_hq/
    train/
      images/
        *.png
```

The public presets point to these paths. Override `RAW_DATA_ROOT` in a local
YAML file if your data are stored elsewhere.

## Mask bank

Training masks are binary images with white pixels denoting the missing region.
The sampler expects area-binned directories:

```text
data/masks/
  irregular/
    mask_rates_5_10/
    mask_rates_10_20/
    mask_rates_20_30/
    mask_rates_30_40/
    mask_rates_40_50/
  coco/
    mask_rates_5_10/
    mask_rates_10_20/
    mask_rates_20_30/
    mask_rates_30_40/
    mask_rates_40_50/
```

PNG, JPG, and JPEG masks are supported. Masks are resized with nearest-neighbor
interpolation and binarized at 127.

## Perceptual-loss encoder

Training uses the same ADE20K ResNet-50 dilated encoder convention as LaMa and
ZITS++. Download `encoder_epoch_20.pth` from the FoTa-Net
[checkpoint folder](https://drive.google.com/drive/folders/1X6vLWR9ukzFEY1VXR-LjsppxwlVAlEX5?usp=sharing)
or the [upstream CSAIL source](http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth),
then keep this directory structure:

```text
perceptual-weights/
  ade20k/
    ade20k-resnet50dilated-ppm_deepsup/
      encoder_epoch_20.pth
```

The expected SHA256 checksum is
`d7dcb0234a2c1fd23d490d48c2c2fc5c39dc2b0ce39085b2f6f7e867fdd5d304`.

Set `HRF_LOSS_WEIGHTS_PATH` to the `perceptual-weights` directory in your local
training overlay. This file is not required for inference or standalone
evaluation.

## File lists for inference and evaluation

Inference and evaluation use UTF-8 text files containing one path per line.
Generate an image list with:

```bash
python tools/make_flist.py --root data/places2/val --out data/manifests/images.flist
python tools/make_mask_flist.py --help
```

Paths may be absolute or relative to the working directory. The mask convention
is `1 = hole` after binarization. Keep image and mask ordering fixed when
reproducing paired metrics.
