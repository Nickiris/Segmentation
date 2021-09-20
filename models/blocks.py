import torch
import torch.nn as nn
import torch.nn.functional as F

# blocks in network
def conv3x3(in_planes, out_planes, stride=1, dilation=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=(stride, stride),
                     padding=dilation, bias=bias, dilation=(dilation, dilation))

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1), stride=(stride, stride),
                     padding=0, bias=bias)


class Bottleneck(nn.Module):
    # resnet bottleneck
    expansion = 4

    def __init__(self, in_channels, out_channels, norm_layer, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(in_planes=in_channels, out_planes=out_channels)
        self.bn1 = norm_layer(out_channels)

        self.conv2 = conv3x3(in_planes=out_channels, out_planes=out_channels, stride=stride,
                             dilation=dilation)
        self.bn2 = norm_layer(out_channels)

        self.conv3 = conv1x1(in_planes=out_channels, out_planes=out_channels*self.expansion)
        self.bn3 = norm_layer(out_channels*self.expansion)

        self.relu = nn.ReLU()
        self.downsample = downsample


    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = x + identity
        x = self.relu(x)

        return x

if __name__ == '__main__':
    x = torch.rand(1, 64, 56, 56)
    model = Bottleneck(in_channels=64, out_channels=64, downsample=conv1x1(64, 64*4),
                       norm_layer=nn.BatchNorm2d)
    y = model(x)
    print(y.shape)