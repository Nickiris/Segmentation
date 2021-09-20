import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import Bottleneck, conv1x1, conv3x3

"""
RefineNet:

    Challenges:

        (1)repeated subsampling operation like pooling or convolutional striding lead to significant
    decrease in the initial image resolution.
        (2)previous deconvolution operations are not able to recover the low-level visual features which
    are lost after the downsampling operation.Low-level visual features are essential for accurate prediction
    on the boundaries or details.
        (3)previous state-of-the-art method DeepLab is computational expensive because of high-dimentional
    features,and introduce coarse sub-sampling of features which leads to a loss of important detals.
        (4)how to effectively exploit middle layer features remains an open question and deserves more 
    attention.In this paper,RefineNet is proposed to address this challenge.

    Schemes:

        (1)explicitly exploits all information along the down-sampling process to enable high resolution
    prediction using long-range residual connections.Deeper layers capture the high-level segmantic features,
    lower layers capture the fine-grained features.
        (2)introduce chained residual pooling,which captures rich background context.


    Architectures:

        (1)multi-path refinement(RefineNet) exploits features at multiple levels of abstraction.RefineNet
    refines low-resolution segmantic features with fine-grained low-level features in a recursive manner.
    Cascaded RefineNets which employ residual connections can be trained end-to-end.
        (2)chained residual pooling is able to capture background context from a large image region.
        (3)ResNet models pre-trained for ImageNet are adopted as foundamental building block.


    Improvement:

        (1)RefineNet is too sophisticated to easily build it.

    Results:

        (1)mIoU score of 83.4 on the Pascal VOC 2012 by RefineNet-Res152.
        (2)mIoU score of 68.8 on the Person-Part dataset by RefineNet-Res152.
        (3)mIoU score of 46.5, pixel acc score of 73.6 and mean acc score of 58.9 on NYUDv2 by RefineNet-
    Res152.
        (4)mIoU score of 73.6 on Cityscapestestset by RefineNet-Res101.
        (5)mIoU score of 47.3 on Pascal Context by RefineNet-Res152.
        (6)mIoU score of 45.9, pixel acc score of 80.6 and mean acc score of 58.5 on SUN-RGBD dataset by 
    RefineNet-Res152.
        (7)mIoU score of 40.7 on ADE20K val set by RefineNet-Res152.


"""



class ResidualConvUnit(nn.Module):
    # The block mainly fine-tunes the pre-trained ResNet weight.

    def __init__(self, channels, features):
        super(ResidualConvUnit, self).__init__()

        self.conv1 = conv3x3(channels, features)
        self.conv2 = conv3x3(features, features)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if channels != features:
            self.downsample = conv1x1(channels, features)

    def forward(self, x):
        identity = x

        x = self.relu(x)
        x = self.conv1(x)

        x = self.relu(x)
        x = self.conv2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity

        return x



class ChainedResidualPool(nn.Module):
    # The proposed chained residual pooling aims to capture background context from a large
    # image region.
    def __init__(self, features, block_nums):
        super(ChainedResidualPool, self).__init__()

        # Can the max-pool layer be reused?
        self.maxpool = nn.MaxPool2d(kernel_size=5, padding=2, stride=1)
        # following convolutions  serve as a weighting layer for the summation fusion.
        self.convs = [conv3x3(features, features) for _ in range(block_nums)]
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.relu(x)
        identity = x

        for conv in self.convs:
            x = self.maxpool(x)
            x = conv(x)
            identity += x

        return identity


# todo There are some problems.
class MultiResolutionFusion(nn.Module):
    # All path inputs are then fused into a high-resolution feature map
    # by the multi-resolution fusion block.
    def __init__(self, channels, features, bottum):
        super(MultiResolutionFusion, self).__init__()
        # Convolutions apply for input adaption.
        self.conv1 = conv3x3(channels, features)
        if bottum is not True:
            self.conv2 = conv3x3(features, features)

        self.bottum = bottum


    def forward(self, *xs):

        if self.bottum is not True:
            up_size = xs[0].size()[-2:]
            x0, x1 = self.conv1(xs[0]), self.conv2(xs[1])
            x1 = F.interpolate(x1, size=up_size, mode='bilinear', align_corners=True)
            x = x0 + x1
        else:
            x = self.conv1(xs[0])

        return x


# todo There are some problems.
class RefineNetBlock(nn.Module):

    def __init__(self, channels, features, bottum=False, out_nums=1):
        super(RefineNetBlock, self).__init__()
        self.adaptive_conv1 = nn.Sequential(
            ResidualConvUnit(channels, features),
            ResidualConvUnit(features, features),
        )

        if bottum is not True:
            self.adaptive_conv2 = nn.Sequential(
                    ResidualConvUnit(features * 2 if channels==1024 else features, features),
                    ResidualConvUnit(features, features),
            )
            self.multiresidualfusion = MultiResolutionFusion(features, features, bottum=bottum)

        self.bottum = bottum


        self.chainedresdualpool = ChainedResidualPool(features, block_nums=3)

        # Refinet-4 has two ResidualConvUnit
        # The goal here is to employ non-linearity operations on the multi-path fused feature
        # maps to generate features for further processing or for final prediction.
        self.outconv = nn.Sequential(*[ResidualConvUnit(features, features) for _ in range(out_nums)])


    def forward(self, *xs):
        x0 = self.adaptive_conv1(xs[0])
        if self.bottum is not True:
            x1 = self.adaptive_conv2(xs[1])
            x = self.multiresidualfusion(x0, x1)
        else:
            x = x0
        x = self.chainedresdualpool(x)
        x = self.outconv(x)

        return x



class RefineNet(nn.Module):

    in_channels = 64
    dilation = 1

    def __init__(self,
                 num_classes,
                 block,
                 norm_layer,
                 features = 256,
                 block_nums=(3, 4, 23, 3)
                 ):
        super(RefineNet, self).__init__()

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=(7, 7), stride=(2, 2), padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64,  block_nums[0], norm_layer)
        self.layer2 = self._make_layer(block, 128, block_nums[1], norm_layer, stride=2)
        self.layer3 = self._make_layer(block, 256, block_nums[2], norm_layer, stride=2)
        self.layer4 = self._make_layer(block, 512, block_nums[3], norm_layer, stride=2)

        # self.refinenet1 = RefineNetBlock(256,  features, out_nums=2)
        # self.refinenet2 = RefineNetBlock(512,  features)
        # self.refinenet3 = RefineNetBlock(1024, features)
        # self.refinenet4 = RefineNetBlock(2048, features * 2, bottum=True)

        # four cascaded RefineNet
        # adaptive_convs
        self.adaptive_conv1 = self._make_adaptive_conv(256,  features)
        self.adaptive_conv2 = self._make_adaptive_conv(512,  features)
        self.adaptive_conv3 = self._make_adaptive_conv(1024, features)
        self.adaptive_conv4 = self._make_adaptive_conv(2048, features * 2)

        # Convolutions before Fusion block
        self.conv3x3_1 = conv3x3(features, features)
        self.conv3x3_2_1 = conv3x3(features, features)
        self.conv3x3_2_2 = conv3x3(features, features)
        self.conv3x3_3_1 = conv3x3(features, features)
        self.conv3x3_3_2 = conv3x3(features, features)
        # channels: 512 --> 256
        self.conv3x3_4 = conv3x3(features * 2, features)

        self.chainedresidualpool1 = ChainedResidualPool(features, features)
        self.chainedresidualpool2 = ChainedResidualPool(features, features)
        self.chainedresidualpool3 = ChainedResidualPool(features, features)
        self.chainedresidualpool4 = ChainedResidualPool(features * 2, features * 2)

        # RefineNet-1 has two ResidualConvUnit
        self.outputconv1 = self._make_adaptive_conv(features, features)
        self.outputconv2 = ResidualConvUnit(features, features)
        self.outputconv3 = ResidualConvUnit(features, features)
        self.outputconv4 = ResidualConvUnit(features * 2, features * 2)

        self.outputconv = conv1x1(features, num_classes)


        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def _make_adaptive_conv(self, channels, out_channels):
        layers = nn.Sequential(
            ResidualConvUnit(channels, out_channels),
            ResidualConvUnit(out_channels, out_channels),
        )
        return layers


    def _make_layer(self, block, channels, block_nums, norm_layer, stride=1, dilation=1):
        downsample = None
        previous_dilation = self.dilation
        self.dilation = dilation

        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, block.expansion * channels, stride=stride),
                norm_layer(block.expansion * channels),
            )

        layers = []
        layers.append(block(self.in_channels, channels, stride=stride, dilation=previous_dilation,
                            norm_layer=norm_layer, downsample=downsample))
        self.in_channels = block.expansion * channels
        for _ in range(1, block_nums):
            layers.append(block(self.in_channels, channels, dilation=dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def forward(self, x):
        input_size = x.size()[-2:]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # x4 = self.refinenet4(x4)
        # x3 = self.refinenet3(x3, x4)
        # x2 = self.refinenet2(x2, x3)
        # x1 = self.refinenet1(x1, x2)

        # RefineNet-4 directly go through the MultiResidualFusion block
        x4 = self.adaptive_conv4(x4)
        x4 = self.chainedresidualpool4(x4)
        x4 = self.outputconv4(x4)
        # channels: 512 --> 256
        # the Convolutions also re-scale the feature values appropriately along different paths
        # before Fusion Block on RefineNet-3
        x4 = self.conv3x3_4(x4)
        x4 = F.interpolate(x4, size=x3.size()[-2:], mode='bilinear', align_corners=True)

        x3 = self.adaptive_conv3(x3)
        x3 = self.conv3x3_3_1(x3)
        x3 += x4
        x3 = self.chainedresidualpool3(x3)
        x3 = self.outputconv3(x3)
        # before Fusion Block on RefineNet-2
        x3 = self.conv3x3_3_2(x3)
        x3 = F.interpolate(x3, size=x2.size()[-2:], mode='bilinear', align_corners=True)

        x2 = self.adaptive_conv2(x2)
        x2 = self.conv3x3_2_1(x2)

        x2 += x3
        x2 = self.chainedresidualpool2(x2)
        x2 = self.outputconv2(x2)
        # before Fusion Block on RefineNet-1
        x2 = self.conv3x3_2_2(x2)
        x2 = F.interpolate(x2, size=x1.size()[-2:], mode='bilinear', align_corners=True)

        x1 = self.adaptive_conv1(x1)
        x1 = self.conv3x3_1(x1)
        x1 += x2
        x1 = self.chainedresidualpool1(x1)
        x1 = self.outputconv1(x1)

        # channels 256 --> num_classes
        out = self.outputconv(x1)
        # up-sample the 1/4 size of image resolution to full image resolution
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)

        return out


if __name__ == '__main__':
    x = torch.rand(2, 3, 224, 224)
    model = RefineNet(num_classes=2, block=Bottleneck, norm_layer=nn.BatchNorm2d)
    y = model(x)
    print(y.shape)




