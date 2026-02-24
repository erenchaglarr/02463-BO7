from __future__ import annotations

import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(
        self,
        depth: int,
        kernel_size: int,
        dropout_rate: float,
        num_classes: int,
        base_channels: int = 32,
    ) -> None:
        super().__init__()

        if depth < 1:
            raise ValueError("depth must be >= 1")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd (3, 5, 7, 9)")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError("dropout_rate must be in [0, 1]")

        layers = []
        in_channels = 1  # Omniglot is grayscale
        out_channels = base_channels

        # Conv blocks: Conv -> BN -> ReLU -> MaxPool -> Dropout2d
        # Padding keeps spatial size stable before pooling.
        for i in range(depth):
            out_channels = min(base_channels * (2 ** min(i, 3)), 256)

            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout2d(p=dropout_rate),
                ]
            )
            in_channels = out_channels

        self.features = nn.Sequential(*layers)

        # Adaptive pooling makes classifier input independent of image size
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x