"""Simple CNN for CIFAR-10 smoke training."""

from __future__ import annotations

import torch
from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(config: dict) -> nn.Module:
    name = config.get("model", {}).get("name", "simple_cnn").lower()
    if name != "simple_cnn":
        raise ValueError(f"Only simple_cnn is supported in this stage, got: {name}")
    return SimpleCNN(num_classes=int(config.get("model", {}).get("num_classes", 10)))
