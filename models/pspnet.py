import torch
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import Bottleneck



class PyramidPooling(nn.Module):
    # the number of pyramid levels and size of each level can be modified.

    def __init__(self, channels, norm_layer, pooling_levels):
        super(PyramidPooling, self).__init__()

        out_channels = channels // len(pooling_levels)
        self.stages = [self._make_stage(channels, out_channels,
                            norm_layer, level) for level in pooling_levels]

        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels+out_channels * len(pooling_levels), out_channels,
                      kernel_size=(1, 1), bias=False),
            norm_layer(out_channels),
            nn.ReLU(),
        )


    def _make_stage(self, channels, out_channels, norm_layer, level):
        layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((level, level)),
            nn.Conv2d(channels, out_channels, kernel_size=(1, 1), bias=False),
            norm_layer(out_channels),
            nn.ReLU(),
        )
        return layers


    def forward(self, x):
        input_size = x.size()[-2:]
        pyramid = [x]
        pyramid.extend([F.interpolate(stage(x), size=input_size, mode='bilinear', align_corners=True) for stage in self.stages])
        output = torch.cat(pyramid, dim=1)

        return self.bottleneck(output)


class PSPNet(nn.Module):

    _in_channels =64
    _dilation = 1
    def __init__(self,
                 num_classes,
                 block,
                 norm_layer,
                 auxiliary = False,  # resnet4b
                 block_nums=(3, 4, 23, 3),
                 pooling_levels=(1, 2, 3, 6)
                 ):
        super(PSPNet, self).__init__()

        self.stage0 = nn.Sequential(
            nn.Conv2d(3, self._in_channels, kernel_size=(7, 7), padding=3, stride=(2, 2), bias=False),
            norm_layer(self._in_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2),
        )

        self.stage1 = self._make_stage(block=block, channels=64, block_num=block_nums[0],
                                       norm_layer=norm_layer)
        self.stage2 = self._make_stage(block=block, channels=128, block_num=block_nums[1],
                                       norm_layer=norm_layer, stride=2)
        self.stage3 = self._make_stage(block=block, channels=256, block_num=block_nums[2],
                                       norm_layer=norm_layer, dilation=2)
        self.stage4 = self._make_stage(block=block, channels=512, block_num=block_nums[3],
                                       norm_layer=norm_layer, dilation=4)

        if auxiliary is not False:
            # if backbone is ResNet101
            self.auxlayer = nn.Sequential(
                    nn.Conv2d(1024, 256, kernel_size=(3, 3), padding=1, bias=False),
                    norm_layer(256),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Conv2d(256, num_classes, kernel_size=(1, 1), bias=False),
            )
        self.auxiliary = auxiliary

        self.master_branch = nn.Sequential(
            PyramidPooling(channels=2048, norm_layer=norm_layer,
                           pooling_levels=pooling_levels),
            nn.Conv2d(2048 // len(pooling_levels), num_classes,
                      kernel_size=(1, 1), bias=False),
        )


    def _make_stage(self, block, channels, block_num, norm_layer, stride=1, dilation=1):

        downsample = None
        previous_dilation = self._dilation
        self._dilation = dilation

        if stride != 1 or self._in_channels != block.expansion * channels:
            downsample = nn.Sequential(
                            nn.Conv2d(self._in_channels, block.expansion * channels,
                                   kernel_size=(1, 1), stride=(stride, stride), bias=False),
                            norm_layer(block.expansion * channels),
            )
        layers = []
        layers.append(block(self._in_channels, channels, norm_layer,
                            stride=stride, dilation=previous_dilation, downsample=downsample))
        self._in_channels = block.expansion * channels
        for _ in range(1, block_num):
            layers.append(block(self._in_channels, channels, dilation=dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def forward(self, x):
        input_size = x.size()[-2:]

        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x_aux = self.stage3(x)
        x = self.stage4(x_aux)
        x = self.master_branch(x)
        outputs = [F.interpolate(x, size=input_size,
                                 mode='bilinear', align_corners=True)]
        if self.auxiliary:
            auxiliary_out = self.auxlayer(x_aux)
            outputs.append(auxiliary_out)

        return outputs


if __name__ == '__main__':
    # pool = PyramidPooling(channels=2048,  norm_layer=nn.BatchNorm2d, pooling_levels=(1, 2, 3, 6))
    x = torch.rand(3, 3, 224, 224)
    model = PSPNet(num_classes=21, block=Bottleneck, norm_layer=nn.BatchNorm2d,
                   auxiliary=True)
    y = model(x)
    print(y[0].shape, y[1].shape)