import torch
from torch import nn
import torch.nn.functional as F

class conv_base(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(conv_base, self).__init__()
        # 2 * in_channels = out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)

        return x

class down_block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(down_block, self).__init__()
        # 2 * in_channels = out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv_base(in_channels, out_channels),
        )

    def forward(self, x):
        x = self.block(x)

        return x


class up_block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(up_block, self).__init__()
        # in_channels = 2 * out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = conv_base(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[-2] - x1.size()[-2]
        diffX = x2.size()[-1] - x1.size()[-1]

        # pad the size of x1 to the size of x2
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2,x1], dim=1)
        x = self.conv(x)

        return x


class U_Net(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super(U_Net, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.head = conv_base(in_channels, 64)

        self.down1 = down_block(64, 128)
        self.down2 = down_block(128, 256)
        self.down3 = down_block(256, 512)
        self.down4 = down_block(512, 1024)

        self.up4 = up_block(1024, 512)
        self.up3 = up_block(512, 256)
        self.up2 = up_block(256, 128)
        self.up1 = up_block(128, 64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.head(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.out(x)

        return x


if __name__ == '__main__':
    model = U_Net(3, 8)
    x = torch.rand(2,3,512, 512)
    y = model(x)
    #y.shape = (2, 8, 512, 512)
    print(y.shape)