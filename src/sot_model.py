"""
SiamFC: Fully-Convolutional Siamese Network for Single Object Tracking (SOT).

Paper: Bertinetto et al., "Fully-Convolutional Siamese Networks for Object Tracking", ECCVW 2016.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiamFCBackbone(nn.Module):
    """Convolutional Feature Extractor for Siamese Tracker."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Conv1: 3 -> 96
            nn.Conv2d(3, 96, kernel_size=11, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv2: 96 -> 256
            nn.Conv2d(96, 256, kernel_size=5, stride=1, groups=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv3: 256 -> 384
            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            # Conv4: 384 -> 384
            nn.Conv2d(384, 384, kernel_size=3, stride=1, groups=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            # Conv5: 384 -> 256
            nn.Conv2d(384, 256, kernel_size=3, stride=1, groups=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class SiamFC(nn.Module):
    """SiamFC Tracking Model with depthwise cross-correlation."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = SiamFCBackbone()
        self.upsample = nn.Upsample(size=(17, 17), mode="bilinear", align_corners=True)

    @staticmethod
    def _cross_correlation(z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Cross-correlation between template features z and search features x.
        z shape: (B, C, Hz, Wz)
        x shape: (B, C, Hx, Wx)
        """
        b, c, hx, wx = x.size()
        x_reshaped = x.view(1, b * c, hx, wx)
        z_reshaped = z.view(b * c, 1, z.size(2), z.size(3))

        out = F.conv2d(x_reshaped, z_reshaped, groups=b * c)
        out = out.view(b, c, out.size(2), out.size(3))
        score = out.mean(dim=1, keepdim=True)
        return score

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        z: Exemplar/Template crop (B, 3, 127, 127)
        x: Search region crop (B, 3, 255, 255)
        Output: 2D target location score map (B, 1, 17, 17)
        """
        feat_z = self.backbone(z)
        feat_x = self.backbone(x)
        score_map = self._cross_correlation(feat_z, feat_x)
        return self.upsample(score_map)


def load_sot_model(checkpoint_path: str, device: torch.device | str = "cpu") -> SiamFC:
    """Load a trained SiamFC checkpoint."""
    model = SiamFC()
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
