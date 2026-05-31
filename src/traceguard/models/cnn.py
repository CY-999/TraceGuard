"""Simple CNN for CIFAR-10 smoke training."""

from __future__ import annotations

import torch
from torch import nn

from traceguard.models.resnet import resnet18_cifar, resnet18_tiny


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
    num_classes = int(config.get("model", {}).get("num_classes", 10))
    if name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    if name == "resnet18_cifar":
        return resnet18_cifar(num_classes=num_classes)
    if name == "resnet18_tiny":
        return resnet18_tiny(num_classes=num_classes)
    raise ValueError(
        "Unknown model.name. Supported models are: simple_cnn, "
        f"resnet18_cifar, resnet18_tiny. Got: {name}"
    )
