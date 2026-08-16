# FoTa-Net model zoo

Pretrained checkpoints are stored outside Git. The checkpoint folder is
available on [Google Drive](https://drive.google.com/drive/folders/1X6vLWR9ukzFEY1VXR-LjsppxwlVAlEX5?usp=sharing).
Additional release assets will use the standardized filenames below.

| Dataset | Resolution | Seed | Expected filename | SHA256 | Release status |
|---|---:|---:|---|---|---|
| Places2 | 256 | 3407 | `fota_places2_256_seed3407_ema.weights.pth` | `71d48d781d47a1b02d141f94c8dc503430474a3d722e48f8b7fce32d19e0349c` | available in checkpoint folder |
| Places2 | 256 | 2027 | `fota_places2_256_seed2027_ema.weights.pth` | `1d3af45e27b7f748d4dd228ae439b7e9a1b52e75d8310d26df83befbb655860f` | staged for release |
| Places2 | 256 | 4099 | `fota_places2_256_seed4099_ema.weights.pth` | `671bc56c9d90a36cfdcb9e63f2a85ef8a28dd52c25b2064f6621c9ca62677ae8` | staged for release |
| Places2 | 512 | 3407 | `fota_places2_512_ema.weights.pth` | `6c3e6c995467354512f893991c8f1630adfbdcce79e54ee855c95de73617b3d8` | available in checkpoint folder |
| CelebA-HQ | 256 | 3407 | `fota_celebahq_256_ema.weights.pth` | `9d2cb6074401bfbf80ba033be1373614822967c40f9a6989fe1dd8416689d884` | available in checkpoint folder |
| CelebA-HQ | 512 | 3407 | `fota_celebahq_512_ema.weights.pth` | `347f1eb7533df698d03c81ce701953fb3acc719465e11abc6b6fff4f12f56d5a` | available in checkpoint folder |
| CelebA-HQ | 1024 | 3407 | `fota_celebahq_1024_ema.weights.pth` | `90d2124846dd7da54ef95c94fa117ffe8b3ac040641141da6713abb093982d39` | available in checkpoint folder |

## Verify a download

Linux:

```bash
sha256sum weights/fota_places2_256_seed3407_ema.weights.pth
```

PowerShell:

```powershell
Get-FileHash weights/fota_places2_256_seed3407_ema.weights.pth -Algorithm SHA256
```

FoTa-Net accepts a raw state dictionary as well as checkpoints containing
`emaG`, `state_dict`, `netG`, `model`, or `generator`.
