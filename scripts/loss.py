import torch
import os
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

def mask2label(masks, color_maps):
    size = masks.size()[-2:]
    idx = (masks[:,0,:,:] * size[-2] + masks[:,1,:,:]) * size[-1] + masks[:, 2, :, :]
    cm2lbl = torch.zeros(size**3, dtype=torch.long)
    for i, cm in enumerate(color_maps):
        cm2lbl[(cm[0] * size[-2] + cm[1]) * size[-1] + cm[2]] = i  # 建立索引
    labels = cm2lbl[idx.long()]
    return labels


class FcnLoss(nn.Module):
    def __init__(self):
        super(FcnLoss, self).__init__()
        pass
        # self.classes = classes
        # self.color_maps = color_maps
        # self.size = size
        # assert len(classes) == len(color_maps)
        # cm2lbl = torch.zeros(size**3, dtype=torch.long)
        # for i, cm in enumerate(color_maps):
        #     cm2lbl[(cm[0] * size + cm[1]) * size + cm[2]] = i  # 建立索引
        # self.cm2lbl = cm2lbl


    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        :param input:[batch, num_classes, H, W]
        :param target:[batch, H, W]
        :return:Tensor
        """
        # input = F.log_softmax(input, dim=1)
        return F.cross_entropy(input, target)


if __name__ == "__main__":
    # classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
    #        'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    #        'dog', 'horse', 'motorbike', 'person', 'potted plant',
    #        'sheep', 'sofa', 'train', 'monitor']
    #
    # base_path = r"C:\Users\12112\iPython\Jupyter\Deep Learning\data\Pascal VOC 2012\train"
    # img_path = os.path.join(base_path, "JPEGImages")
    # mask_path = os.path.join(base_path, "SegmentationClass")
    # img_sets = os.path.join(base_path, "ImageSets/Segmentation/train.txt")
    # with open(img_sets,'r') as f:
    #     img_sets = f.read().split('\n')
    # from data.pascal_voc import VOCSegmentation
    # import torchvision.transforms as transforms
    # from torch.utils.data import Dataset, DataLoader
    # # ToTensor() convert shape(H, W, C) array to shape(C, H, W) tensor,and divided 255.
    # transform = transforms.Compose([transforms.ToTensor(),# PIL or ndarray
    #                                 transforms.CenterCrop((256, 256)),  # tensor
    #                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #                                 ])
    # dataset = VOCSegmentation(img_path, mask_path, img_sets, transform)
    # dataloader = DataLoader(dataset,
    #                         shuffle=True,
    #                         batch_size=4)
    # # RGB color for each class
    colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
                [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
                [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
                [0, 192, 0], [128, 192, 0], [0, 64, 128]]
    # labels = []
    # imgs = []
    # for img, label in dataloader:
    #     imgs.append(img)
    #     labels.append(label)
    #     break
    # import matplotlib.pyplot as plt
    # plt.subplot(1,2,1)
    # tst_img = img[0].permute(1, 2, 0)
    # plt.imshow(tst_img/255.)
    # plt.subplot(1,2,2)
    # plt.imshow(label[0].permute(1, 2, 0))
    # plt.show()
    pass
