import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

"""
    The code is inspired by the image metrics.jpg in FCN. 
    TN: background pixel
    Pix accuracy: (TP + TN) / (TP + FP + TN + FN)
    iou: TP / (TP + FP + FN)
    mean iou: the mean of all categories iou including background.
    recall: TP / (TP + FN)
"""

#Todo optimized implementation
def confusion_matrix(input, target, num_classes):
    """
    input: torch.LongTensor:(H,W)
    target: torch.LongTensor:(H,W)
    num_classes: int
    results:Tensor
    """
    assert torch.max(input) < num_classes
    assert torch.max(target) < num_classes
    H, W = target.size()
    results = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for h in range(H):
        for w in range(W):
            results[target[h,w], input[h,w]] += 1
    return results

def pixel_accuracy(input, target):
    """
    input: torch.FloatTensor:(N,C,H,W)
    target: torch.LongTensor:(N,H,W)
    return: Tensor
    """
    assert len(input.size()) == 4
    assert len(target.size()) == 3
    N, H, W = target.size()
    input = F.softmax(input, dim=1)
    arg_max = torch.argmax(input, dim=1)
    # (TP + TN) / (TP + TN + FP + FN)
    return torch.sum(arg_max == target) / (H * W)

def mean_pixel_accuarcy(input, target):
    """
    input: torch.FloatTensor:(N,C,H,W)
    target: torch.LongTensor:(N,H,W)
    num_classes: int
    return: Tensor
    """
    assert len(input.size()) == 4
    assert len(target.size()) == 3
    N, num_classes, H, W = input.size()
    input = F.softmax(input, dim=1)
    arg_max = torch.argmax(input, dim=1)
    # matrix = confusion_matrix(arg_max, target, num_classes)
    result = 0
    for i in range(N):
        matrix = confusion_matrix(arg_max[i,:,:], target[i,:,:], num_classes)
        for k in range(num_classes):
            if matrix[k, k] == 0:
                continue
            else:
                result += (matrix[k, k] / torch.sum(matrix[k,:]))
    return result / num_classes


def iou(input, target):
    """
    input: torch.FloatTensor:(N,C,H,W)
    target: torch.LongTensor:(N,H,W)
    return: Tensor
    """
    assert len(input.size()) == 4
    assert len(target.size()) == 3
    N, H, W = target.size()
    input = F.softmax(input, dim=1)
    result = 0
    arg_max = torch.argmax(input, dim=1)
    for i in range(N):
        TN = torch.sum(arg_max[i, :, :] + target[i, :, :] == 0)
        # TP / (TP + FP + FN)
        result += (torch.sum(arg_max[i,:,:] == target[i,:,:]) - TN) / (H * W - TN)

    return result

def mean_iou(input, target):
    """
    input: torch.FloatTensor:(N,C,H,W)
    target: torch.LongTensor:(N,H,W)
    return: Tensor
    """
    assert len(input.size()) == 4
    assert len(target.size()) == 3
    N, num_classes, H, W = input.size()
    input = F.softmax(input, dim=1)
    arg_max = torch.argmax(input, dim=1)
    result = 0
    for i in range(N):
        matrix = confusion_matrix(arg_max[i,:,:], target[i,:, :], num_classes)
        for k in range(num_classes):
            nii = matrix[k, k]
            if nii == 0:
                continue
            else:
                ti, tj = torch.sum(matrix[k, :]), torch.sum(matrix[:, k])
                result += (nii / (ti + tj - nii))

    return result / num_classes

# Todo: Read more paper
def frequency_weighted_iou(input, target):
    """
    input: torch.FloatTensor:(N,C,H,W)
    target: torch.LongTensor:(N,H,W)
    return: Tensor
    """
    assert len(input.size()) == 4
    assert len(target.size()) == 3
    N, num_classes, H, W = input.size()
    input = F.softmax(input, dim=1)
    arg_max = torch.argmax(input, dim=1)
    # get confusion matrix
    result = 0
    for i in range(N):
        matrix = confusion_matrix(arg_max[i, :, :], target[i, :, :], num_classes)
        for k in range(num_classes):
            nii = matrix[k, k]
            if nii == 0:
                continue
            else:
                ti, tj = torch.sum(matrix[k, :]), torch.sum(matrix[:, k])
                result += (ti * nii / (ti + tj - nii))

        result = result / torch.sum(matrix)

    return result




