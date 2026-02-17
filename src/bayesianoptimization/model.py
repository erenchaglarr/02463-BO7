from torch import nn
import torch

class CNNModel(nn.Module):
    def __init__(self, num_conv_blocks=3, kernel_size=3, dropout_rate=0.3):
        super().__init__()
        layers = []
        in_channels = 1
        for _ in range(num_conv_blocks):
            layers.append(nn.Conv2d(in_channels, 32, kernel_size=kernel_size, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = 32
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(32, 50)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = CNNModel()
    x = torch.rand(1)
    print(f"Output shape of model: {model(x).shape}")
