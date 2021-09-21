import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.blocks import Bottleneck
import pydensecrf.densecrf as dcrf

"""
DeepLabv1
    Details:
        SGD-Momentum, momentum: 0.9, weight decay: 0.0005, loss function is cross entropy 
        mini-batch: 20, initial learning rate: 0.001, 0.1 for the final classifier layer, multiplying the learning rate by 0.1 at every
    2000 iterations.
    
    Others:
        损失函数：交叉熵之和。
        训练数据label：对原始Ground Truth进行下采样8倍，得到训练label。
        预测数据label：对预测结果进行双线性上采样8倍，得到预测结果。

    Architecture:
        网络主干是修改的VGG16，最后两个最大池化的stride设为1，最后两个池化后的卷积为dilated卷积。
        网络的全连接层改为卷积层，参数设置参考FOV。
        网络的输出结果需要经过10次CRF的迭代。

"""


class DeepLabv1(nn.Module):

    def __init__(self, num_classes, init_weights=True):
        super(DeepLabv1, self).__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(), nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        )
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Sequential(
            # DeepLab CRF 7×7 即 kernel_size = 7, dilation = 4
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=12, dilation=12),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(1024, num_classes, kernel_size=1, bias=False),
        )
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        input_size = x.size()[-2:]
        x = self.features(x)
        X = self.avg_pool(x)
        x = self.classifier(x)

        return x



class DenseCRF(object):

    def __init__(self, w1, w2, alpha, beta, gamma, iterations):

        self.w1 = w1
        self.w2 = w2
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.iterations = iterations


    def _unary_from_softmax(self, proba, scale=None, clip=1e-5):
        num_classes = proba.shape[0]
        if scale is not None:
            assert 0 < scale <= 1, "`scale` needs to be in (0,1]"
            uniform = np.ones(proba.shape) / num_classes
            proba = scale * proba + (1 - scale) * uniform

        if clip is not None:
            proba = np.clip(proba, clip, 1.0)

        return -np.log(proba).reshape(num_classes, -1).astype(np.float32)


    def __call__(self, image, proba):
        # proba为softmax处理后的结果
        num_classes, h, w = proba.shape

        C, H, W = proba.shape

        U = self._unary_from_softmax(proba)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)

        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.gamma, compat=self.w2)
        d.addPairwiseBilateral(
            sxy=self.alpha, srgb=self.beta, rgbim=image, compat=self.w1
        )

        Q = d.inference(self.iterations)
        Q = np.array(Q).reshape((C, H, W))

        return Q


class ASPPv1(nn.Module):
    # modified in deeplabv3

    def __init__(self, num_classes, channels, out_channels, rates=(6, 12, 18, 24)):
        super(ASPPv1, self).__init__()

        self.aspp = [self._make_layer(num_classes, channels, out_channels, rate) for rate in rates]


    def _make_layer(self, num_classes, channels, out_channels, rate):

        layers = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=(3, 3), padding=rate, dilation=rate, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(out_channels, num_classes, kernel_size=(1, 1), bias=False),
        )

        return layers


    def forward(self, x):
        return sum([layer(x) for layer in self.aspp])


"""
DeepLabv2

    challenges:
               （1）降低的特征分辨率
               （2）多尺度目标的分割
               （3）卷积的invariance性质导致定位精度差
               
    schemes:
            （1）采用atrous convolutions
            （2）采用atrous spatial pyramid pooling（ASPP)：采用不同采样率的并行空洞卷积
            （3）采用full-connected Conditional Random Field
            
    architecture:
            （1）backbone:带有dilation系数的VGG16或者ResNet101
            （2）ASPP:接收backbone传入的feature map，进行多尺度信息聚合
            （3）DenseCRF:获得精度的定位边界
    
    improvements:
            （1）backbone由VGG16换成了ResNet101
            （2）采用了Atrous Spatial Pyramid Pooling layer融合多尺度信息
            （3）采用了poly策略动态地调整学习率
    
    results: all results are under the  ResNet101 + Multi-Scale + Pretrained on MS-COCO + Random Rescaling + ASPP + CRF.
            （1）Pascal VOC 2012 val: 77.69% mIoU, test: 79.7% mIoU.
            （2）Pascal Context val: 45.7% mIoU.
            （3）PASCAL-Person-Part val: 64.94% mIoU，未使用ASPP（ASPP和LFOV对该数据集几乎没有影响）.
            （4）Cityscapes val:71.4% mIou, test: 70.4% mIoU, trained with full resolution images, without Multi-Scale due to limited GPU
            memories.
            
"""



class DeepLabv2(nn.Module):

    _in_channels = 64
    _dilation = 1

    def __init__(self, num_classes, norm_layer,
                       block_nums=(3, 4, 23, 3),  # ResNet101
                       rates=(6, 12, 18, 24), block=Bottleneck):

        super(DeepLabv2, self).__init__()

        # ResNet 101
        self.conv1 = nn.Conv2d(3, self._in_channels, kernel_size=(7, 7), padding=3, stride=(2, 2), bias=False)
        self.norm_layer1 = norm_layer(self._in_channels)
        self.relu = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = self._make_layer(block, 64,  nums=block_nums[0], norm_layer=norm_layer)
        self.stage2 = self._make_layer(block, 128, nums=block_nums[1], norm_layer=norm_layer, stride=2)
        self.stage3 = self._make_layer(block, 256, nums=block_nums[2], norm_layer=norm_layer, dilation=2)
        self.stage4 = self._make_layer(block, 512, nums=block_nums[3], norm_layer=norm_layer, dilation=4)

        # ResNet101 output channels are 2048
        self.aspp = ASPPv1(num_classes, channels=2048,  out_channels=2048,
                           rates=rates)

    def _make_layer(self, block, channels, nums, norm_layer, stride=1, dilation=1):

        downsample = None
        previous_dilation = self._dilation
        self._dilation = dilation

        if stride != 1 or self._in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self._in_channels, channels * block.expansion,
                          kernel_size=(1, 1), stride=(stride, stride), bias=False),
                norm_layer(channels * block.expansion),
        )

        layers = []
        layers.append(Bottleneck(self._in_channels, channels, stride=stride,
                                    dilation=previous_dilation, downsample=downsample,
                                    norm_layer=norm_layer))
        self._in_channels = channels * block.expansion
        for _ in range(1, nums):
            layers.append(Bottleneck(self._in_channels, channels, dilation=dilation,
                                        norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def forward(self, x):

        x = self.conv1(x)
        x = self.norm_layer1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.aspp(x)

        return x



"""
DeepLabv3
    Challenges:

        (1)the reduced feature resolution caused by consecutive pooling operations of convolution 
    striding may impede dense prediction tasks due to invariance to local image transformation.
        (2)the existence of objects at multiple scales.
        (3)atrous rate becomes larger, the weights that are applied to valid feature region become smaller. 

    Schemes:

        (1)advocate the use of atrous convolution.

        (2)apply an image pyramid to extract features from each scale input,that is multi-scale input.
        (3)encoder-decoder structure exploits multi-scale features from the encoder and recovers the
    spital resolution from decoder.
        (4)extra modules are cascaded on the top of network,i.e. DenseCRF, extra convolutional layers for
    capturing long range information(context).
        (5)spatial pyramid pooling.

        (6)apply atrous spatial pyramid pooling with global average pooling

    Architectures:

        (1)propose to augment ASPP with image-level features.
        (2)employ a hierarchy of grids of different sizes,output_stride means the ratio of 
    input resolution to output resolution,Multi_Grid = (r1, r2, r3) for three convolutional layers within
    block4 to block7, the three convolutional layers rates = 2 * (r1, r2, r3)

        (3)apply poly learning rate policy with power=0.9,batch_size is equal to the output_stride(16).
        (4)training on the trainaug set with 30k iterations and learning rate is 0.007,then freeze batch
    normallization parameters, employ output_stride=8 and training on the trainval set with another 30k 
    iterations and smaller base learning rate=0.001.
        (5)data augmentation applies randomly scaling the input images (from 0.5 to 2.0) and randomly 
    left-right flipping during training.

    Improvements:

        (1)

    Results:

        (1)attains a mIoU score of 85.7% on the PASCAL VOC 2012 test without DenseCRF post-processing.

"""


class ASPP(nn.Module):
    def __init__(self, num_classes, features, rates):
        super(ASPP, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1x1_1 = self._make_layer(features, kernel_size=1)
        self.conv3x3_1 = self._make_layer(features, kernel_size=3, dilation=rates[0],
                                          padding=rates[0])
        self.conv3x3_2 = self._make_layer(features, kernel_size=3, dilation=rates[1],
                                          padding=rates[1])
        self.conv3x3_3 = self._make_layer(features, kernel_size=3, dilation=rates[2],
                                          padding=rates[2])

        self.conv1x1_2 = self._make_layer(features, kernel_size=1)
        self.conv1x1_3 = self._make_layer(256 * 5, kernel_size=1)
        self.output = nn.Conv2d(256, num_classes, kernel_size=(1, 1), bias=False)


    def _make_layer(self, features, kernel_size, dilation=1, padding=0, bias=False):
        layers = nn.Sequential(
            nn.Conv2d(features, 256, kernel_size=(kernel_size, kernel_size), padding=padding,
                      dilation=(dilation, dilation), bias=bias),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        return layers


    def forward(self, x):
        input_size = x.size()[-2:]

        out1 = self.conv1x1_1(x)
        out2 = self.conv3x3_1(x)
        out3 = self.conv3x3_2(x)
        out4 = self.conv3x3_3(x)

        img_pool = self.avgpool(x)
        img_pool = self.conv1x1_2(img_pool)
        img_pool = F.interpolate(img_pool, size=input_size, mode='bilinear',
                                 align_corners=True)

        out = torch.cat([out1, out2, out3, out4, img_pool], dim=1)
        out = self.conv1x1_3(out)

        out = self.output(out)

        return out


class DeepLabv3(nn.Module):

    in_channels = 64
    dilation = 1

    def __init__(self,
                 num_classes,
                 block,
                 norm_layer,
                 block_nums,
                 rates,
                 multi_grid,
                 output_stride=16,
                 freeze_bn=False):
        super(DeepLabv3, self).__init__()

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=(7, 7), padding=3,
                               stride=(2, 2), bias=False)
        self.bn1 = norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        strides = None
        dilations = None
        if output_stride == 16:
            strides = (1, 2, 2, 1)
            dilations = (1, 1, 1, 2)
        elif output_stride == 8:
            strides = (1, 2, 1, 1)
            dilations = (1, 1, 2, 4)

        self.layer1 = self._make_layer(block, 64,  block_nums=block_nums[0], norm_layer=norm_layer,
                                       stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(block, 128, block_nums=block_nums[1], norm_layer=norm_layer,
                                       stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(block, 256, block_nums=block_nums[2], norm_layer=norm_layer,
                                       stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_layer(block, 512, block_nums=block_nums[3], norm_layer=norm_layer,
                                       stride=strides[3], dilation=dilations[3])
        self.layer5 = self._make_layer(block, 512, block_nums=multi_grid[0], norm_layer=norm_layer,
                                       dilation=2 * multi_grid[0])
        self.layer6 = self._make_layer(block, 512, block_nums=multi_grid[1], norm_layer=norm_layer,
                                       dilation=2 * multi_grid[1])
        self.layer7 = self._make_layer(block, 512, block_nums=multi_grid[2], norm_layer=norm_layer,
                                       dilation=2 * multi_grid[2])


        self.aspp = ASPP(num_classes=num_classes, features=2048, rates=rates)

        if freeze_bn is not False:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def _make_layer(self, block, channels, block_nums, norm_layer, stride=1, dilation=1):
        downsample = None
        previous_dilation = self.dilation
        self.dilation = dilation

        if stride != 1 or self.in_channels != block.expansion * channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, block.expansion * channels, kernel_size=(1, 1),
                          stride=(stride, stride), bias=False),
                norm_layer(block.expansion * channels),
            )

        layers = []
        layers.append(block(self.in_channels, channels, dilation=previous_dilation,
                            stride=stride, norm_layer=norm_layer, downsample=downsample))
        self.in_channels = block.expansion * channels
        for _ in range(1, block_nums):
            layers.append(block(self.in_channels, channels, dilation=dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        inpu_size = x.size()[-2:]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        x = self.aspp(x)
        x = F.interpolate(x, size=inpu_size, mode='bilinear', align_corners=True)

        return x



if __name__ == '__main__':

    import matplotlib.pyplot as plt
    # model = DeepLabv2(num_classes=21, norm_layer=nn.BatchNorm2d)
    # x = torch.rand(1, 3, 224, 224)
    # with torch.no_grad():
    #     y = model(x)
    #     print(y.shape)
    #     CRF = DenseCRF(w1=5, w2=3, alpha=140, beta=5, gamma=3, iterations=10)
    #     # input image's dtype is uint8.
    #     mask = CRF(x.squeeze(0).byte().numpy().transpose(1, 2, 0)[:28, :28, :], y.squeeze(0).numpy())
    # output = np.argmax(mask, axis=0)

    model = DeepLabv3(num_classes=21, norm_layer=nn.BatchNorm2d, rates=(6, 12, 18), output_stride=8,
                      multi_grid=(1, 2, 1), block=Bottleneck, block_nums=(3, 4, 23, 3))
    # when model is on training, BatchNorm2d requires more than one input image.
    model.eval()
    x = torch.rand(1, 3, 224, 224)
    y = model(x)
    output = torch.argmax(y, dim=1)
    plt.imshow(output.squeeze(0))
    plt.show()

        