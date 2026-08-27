from __future__ import annotations
import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels), nn.ReLU(True), nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3), nn.InstanceNorm2d(channels))
    def forward(self, x: torch.Tensor) -> torch.Tensor: return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, base: int = 64, blocks: int = 6):
        super().__init__()
        layers: list[nn.Module] = [nn.ReflectionPad2d(3), nn.Conv2d(in_channels, base, 7), nn.InstanceNorm2d(base), nn.ReLU(True)]
        channels = base
        for _ in range(2):
            layers += [nn.Conv2d(channels, channels * 2, 3, 2, 1), nn.InstanceNorm2d(channels * 2), nn.ReLU(True)]
            channels *= 2
        layers += [ResidualBlock(channels) for _ in range(blocks)]
        for _ in range(2):
            layers += [nn.ConvTranspose2d(channels, channels // 2, 3, 2, 1, output_padding=1), nn.InstanceNorm2d(channels // 2), nn.ReLU(True)]
            channels //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(channels, out_channels, 7), nn.Tanh()]
        self.net = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, channels: int, base: int = 64):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(channels, base, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        previous = base
        for multiplier, stride in ((2, 2), (4, 2), (8, 1)):
            current = min(base * multiplier, 512)
            layers += [nn.Conv2d(previous, current, 4, stride, 1), nn.InstanceNorm2d(current), nn.LeakyReLU(0.2, True)]
            previous = current
        layers += [nn.Conv2d(previous, 1, 4, 1, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.net(x)

def init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None: nn.init.zeros_(module.bias)

