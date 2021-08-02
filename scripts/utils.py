import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def mIou(input, target,classNum):
    '''
    :param input: [b,h,w]
    :param target: [b,h,w]
    :param classNum: scalar
    :return:
    '''
    inputTmp = torch.zeros([input.shape[0],classNum,input.shape[1],input.shape[2]])#创建[b,c,h,w]大小的0矩阵
    targetTmp = torch.zeros([target.shape[0],classNum,target.shape[1],target.shape[2]])#同上
    input = input.unsqueeze(1)#将input维度扩充为[b,1,h,w]
    target = target.unsqueeze(1)#同上
    inputOht = inputTmp.scatter_(index=input,dim=1,value=1)#input作为索引，将0矩阵转换为onehot矩阵
    targetOht = targetTmp.scatter_(index=target,dim=1,value=1)#同上
    batchMious = []#为该batch中每张图像存储一个miou
    mul = inputOht * targetOht#乘法计算后，其中1的个数为intersection
    for i in range(input.shape[0]):#遍历图像
        ious = []
        for j in range(classNum):#遍历类别，包括背景
            intersection = torch.sum(mul[i][j])
            union = torch.sum(inputOht[i][j]) + torch.sum(targetOht[i][j]) - intersection + 1e-6
            iou = intersection / union
            ious.append(iou)
        miou = np.mean(ious)#计算该图像的miou
        batchMious.append(miou)
    return batchMious


def Pa(input, target):
    tmp = input == target

    return (torch.sum(tmp).float() / input.nelement())

if __name__ == "__main__":

    classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'potted plant',
               'sheep', 'sofa', 'train', 'monitor']

    # RGB color for each class
    colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
                [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
                [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
                [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
                [0, 192, 0], [128, 192, 0], [0, 64, 128]]


    cm2lbl = np.zeros(224 ** 3)  # 每个像素点有 0 ~ 255 的选择，RGB 三个通道
    for i, cm in enumerate(colormap):
        cm2lbl[(cm[0] * 224 + cm[1]) * 224 + cm[2]] = i  # 建立索引


    def image2label(im):
        data = np.array(im, dtype='int32')
        idx = (data[:,0,:, :] * 224 + data[:,1,:, :]) * 224 + data[:,2,:, :]
        return np.array(cm2lbl[idx], dtype='int64')  # 根据索引得到 label 矩阵



    label = torch.rand(4, 3, 224, 224) * 2
    label = label.long()
    lbl = image2label(label)
    # print(lbl.shape)
    # print(lbl)




