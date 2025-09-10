import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, f=32):
        super().__init__()
        self.down1 = DoubleConv(in_channels, f)
        self.down2 = DoubleConv(f, f*2)
        self.down3 = DoubleConv(f*2, f*4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(f*4, f*2, 2, stride=2)
        self.conv2 = DoubleConv(f*4, f*2)
        self.up1 = nn.ConvTranspose2d(f*2, f, 2, stride=2)
        self.conv1 = DoubleConv(f*2, f)
        self.head = nn.Conv2d(f, out_channels, 1)
    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        u2 = self.up2(x3)
        u2 = self.conv2(torch.cat([u2, x2], dim=1))
        u1 = self.up1(u2)
        u1 = self.conv1(torch.cat([u1, x1], dim=1))
        return self.head(u1)

def build_unet(in_channels: int, out_channels: int):
    return UNet(in_channels=in_channels, out_channels=out_channels)

