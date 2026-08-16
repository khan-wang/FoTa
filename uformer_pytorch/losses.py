"""Training losses for FoTa-Net.

The high-receptive-field perceptual encoder is adapted from LaMa and ZITS++.
See THIRD_PARTY_NOTICES.md for attribution and license information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
from torchvision import models
from utils.amp import autocast

# ==============================================================================
#           High-Receptive Field (HRF) Perceptual Loss
# ==============================================================================
# Adapted from the Apache-2.0 LaMa implementation for standalone use.

class HRFPerceptualLoss(nn.Module):
    """
    High-Receptive Field Perceptual Loss using a pre-trained ResNet50-based
    semantic segmentation model (trained on ADE20k).
    """

    def __init__(self, weights_path: str, device: torch.device):
        super().__init__()
        self.device = device

        # --- Build Encoder using the correct, modified ResNet architecture ---
        self.encoder = ResNet50Encoder(weights_path).to(self.device).eval()

        # --- Freeze all parameters ---
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss().to(self.device)

        # --- Normalization values for ADE20k pre-training ---
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        ...
        """
        # --- Denormalize from [-1, 1] to [0, 1] ---
        pred_01 = (pred + 1) / 2.0
        target_01 = (target + 1) / 2.0

        # --- Normalize for the segmentation model ---
        pred_norm = (pred_01 - self.mean) / self.std
        target_norm = (target_01 - self.mean) / self.std

        # --- Extract features ---

        with autocast(device_type=self.device.type, enabled=False):
            pred_feats = self.encoder(pred_norm.float())
            target_feats = self.encoder(target_norm.float())

        # --- Calculate L1 loss on the extracted features ---
        return self.criterion(pred_feats, target_feats)

# ==============================================================================
#           Internal implementation of the modified ResNet for HRF Loss
# ==============================================================================
# These classes retain compatibility with the LaMa/ZITS++ pretrained encoder.

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 128
        super(ResNet, self).__init__()
        # --- THIS IS THE MODIFIED PART ---
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        # --- END OF MODIFIED PART ---
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class ResNet50Encoder(nn.Module):
    def __init__(self, weights_path):
        super().__init__()
        # Use the custom ResNet definition
        resnet = ResNet(Bottleneck, [3, 4, 6, 3])

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu1 = resnet.relu1
        self.conv2 = resnet.conv2
        self.bn2 = resnet.bn2
        self.relu2 = resnet.relu2
        self.conv3 = resnet.conv3
        self.bn3 = resnet.bn3
        self.relu3 = resnet.relu3
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Apply dilated convolutions to increase receptive field
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        self.ppm = PPM(2048, 512, [1, 2, 3, 6])

        encoder_weight_path = os.path.join(weights_path, 'ade20k', 'ade20k-resnet50dilated-ppm_deepsup',
                                           'encoder_epoch_20.pth')
        if not os.path.exists(encoder_weight_path):
            raise FileNotFoundError(f"HRF Perceptual Loss weights not found at: {encoder_weight_path}")


        # self.load_state_dict(torch.load(encoder_weight_path, map_location='cpu', weights_only=True), strict=False)


        try:
            state = torch.load(encoder_weight_path, map_location='cpu', weights_only=True)
        except TypeError:
            state = torch.load(encoder_weight_path, map_location='cpu')
        self.load_state_dict(state, strict=False)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # The PPM module is part of the original decoder, not the encoder feature extractor for perceptual loss.
        # We return the output of layer4 to match the intended use.
        return x


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super().__init__()
        self.features = []
        for bin_size in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin_size),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)
