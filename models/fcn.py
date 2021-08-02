import torch
import torchvision
from torch import nn
import numpy as np
import torch.nn.functional as F

class FCNHead(nn.Module):
    def __init__(self, in_channels, channels):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1,  bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            # Dropout随机对张量中的某些元素置零
            # Dropout2d随机对某个通道置零
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1),
        )

    def forward(self, x):
        return self.block(x)


class FCN8s(nn.Module):
    def __init__(self, num_classes=21, bakcbone=None, aux=False):
        super(FCN8s, self).__init__()
        self.backbone = bakcbone
        self.pool3 = nn.Sequential(*bakcbone[:17])
        self.pool4 = nn.Sequential(*bakcbone[17:24])
        self.pool5 = nn.Sequential(*bakcbone[24:])
        self.aux = aux
        if aux:
            self.auxlayer = FCNHead(512, num_classes)
        self.head = FCNHead(512, num_classes)
        self.score_pool3 = nn.Conv2d(256, num_classes, 1)
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)

    def forward(self, x):
        pool3 = self.pool3(x)
        pool4 = self.pool4(pool3)
        # pool5 = self.backbone(x)
        pool5 = self.pool5(pool4)

        outputs = []
        score_pool5 = self.head(pool5)
        score_pool4 = self.score_pool4(pool4)
        score_pool3 = self.score_pool3(pool3)

        upscore5 = F.interpolate(score_pool5, score_pool4.size()[-2:],
                                 mode="bilinear", align_corners=True)
        fuse_score4 = upscore5 + score_pool4

        upscore4 = F.interpolate(fuse_score4, score_pool3.size()[-2:],
                                 mode="bilinear", align_corners=True)
        fuse_score3 = upscore4 + score_pool3

        out = F.interpolate(fuse_score3, x.size()[-2:],
                            mode="bilinear", align_corners=True)
        outputs.append(out)
        if self.aux:
            aux_out = self.auxlayer(pool5)
            aux_out = F.interpolate(aux_out, x.size()[-2:],
                                   mode='bilinear', align_corners=True)
            outputs.append(aux_out)
        # equal -> return out, aux_out
        return tuple(outputs)

if __name__ == "__main__":
    backbone = torchvision.models.vgg.vgg16(pretrained=True).features
    model = FCN8s(num_classes=21, bakcbone=backbone)
    x = torch.rand(13,3,224,224)
    y = model(x)
    print(y[0].shape)

    classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'potted plant',
               'sheep', 'sofa', 'train', 'tv/monitor']

    # RGB color for each class
    colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
                [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
                [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
                [0, 192, 0], [128, 192, 0], [0, 64, 128]]

    cm2lbl = np.zeros(256 ** 3)  # 每个像素点有 0 ~ 255 的选择，RGB 三个通道
    for i, cm in enumerate(colormap):
        cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i  # 建立索引


    def image2label(im):
        data = np.array(im, dtype='int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(cm2lbl[idx], dtype='int64')  # 根据索引得到 label 矩阵