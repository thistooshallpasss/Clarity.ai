import torch
from torch import nn

class ResidualBlock(nn.Module):
    """
    Ek Residual Block, jo Generator ka building block hai.
    """
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    """
    Yeh Generator class hai jise app.py import karne ki koshish kar raha hai.
    """
    def __init__(self, in_channels=3, num_residual_blocks=16):
        super(Generator, self).__init__()
        self.initial_block = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )
        self.upsampling_blocks = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )
        self.final_conv = nn.Conv2d(64, in_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        initial = self.initial_block(x)
        x = self.residual_blocks(initial)
        x = self.conv_block2(x)
        x = self.upsampling_blocks(initial + x)
        return torch.tanh(self.final_conv(x))


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        
        def conv_block(in_feat, out_feat, stride=1, batch_norm=True):
            layers = [nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=stride, padding=1)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            conv_block(64, 64, stride=2),
            conv_block(64, 128),
            conv_block(128, 128, stride=2),
            conv_block(128, 256),
            conv_block(256, 256, stride=2),
            conv_block(256, 512),
            conv_block(512, 512, stride=2),
        )
        self.final_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return self.final_layers(x)