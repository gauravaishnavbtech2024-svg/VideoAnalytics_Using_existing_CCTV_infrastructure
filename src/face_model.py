"""
Face Detection CNN Model.

A lightweight CNN trained on CASIA-WebFace 112x112 dataset to learn
face feature representations. Used to verify that a detected region
contains a real human face (face vs non-face binary classification).

Architecture: MobileFaceNet-inspired lightweight face embedding network.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for efficiency."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.dw(x)), inplace=True)
        x = F.relu(self.bn2(self.pw(x)), inplace=True)
        return x


class FaceDetectorNet(nn.Module):
    """
    Lightweight Face Detection/Embedding CNN.

    Input:  (B, 3, 112, 112) face crop
    Output: (B, num_classes) logits for face identity classification

    During inference, the penultimate 128-d embedding is used as a
    face feature vector to verify if a crop is a real face.
    """

    def __init__(self, num_classes: int = 10572, embedding_dim: int = 128) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        # Feature extractor
        self.features = nn.Sequential(
            # Stage 1: 112x112 -> 56x56
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Stage 2: 56x56 -> 28x28
            DepthwiseSeparableConv(64, 64),
            DepthwiseSeparableConv(64, 128, stride=2),

            # Stage 3: 28x28 -> 14x14
            DepthwiseSeparableConv(128, 128),
            DepthwiseSeparableConv(128, 256, stride=2),

            # Stage 4: 14x14 -> 7x7
            DepthwiseSeparableConv(256, 256),
            DepthwiseSeparableConv(256, 512, stride=2),

            # Stage 5: 7x7 -> 1x1
            DepthwiseSeparableConv(512, 512),
            nn.AdaptiveAvgPool2d(1),
        )

        # Face embedding head
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

        # Classification head (for training only)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 128-d face embedding (for inference / face verification)."""
        feat = self.features(x)
        emb = self.embedding(feat)
        return F.normalize(emb, p=2, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning class logits (for training)."""
        feat = self.features(x)
        emb = self.embedding(feat)
        logits = self.classifier(emb)
        return logits


def load_face_model(checkpoint_path: str, device: torch.device | str = "cpu", num_classes: int = 10572) -> FaceDetectorNet:
    """Load a trained FaceDetectorNet checkpoint."""
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict):
        if "num_classes" in state:
            num_classes = state["num_classes"]
        state_dict = state.get("model_state_dict", state)
    else:
        state_dict = state

    if "classifier.weight" in state_dict:
        num_classes = state_dict["classifier.weight"].shape[0]

    model = FaceDetectorNet(num_classes=num_classes)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model
