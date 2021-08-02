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
    input: torch.LongTensor:(N,H,W)
    target: torch.LongTensor:(N,H,W)
    num_classes: int
    results:Tensor
    """
    assert torch.max(input) < num_classes
    assert torch.max(target) < num_classes
    N, H, W = target.size()
    results = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for i in range(N):
        for h in range(H):
            for w in range(W):
                results[target[i,h,w], input[i,h,w]] += 1
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
    arg_max = torch.argmax(input, dim=1)
    # (TP + TN) / (TP + TN + FP + FN)
    return torch.sum(arg_max == target) / N * H * W

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
    arg_max = torch.argmax(input, dim=1)
    matrix = confusion_matrix(arg_max, target, num_classes)
    result = 0
    for i in range(num_classes):
        result += (matrix[i, i] / torch.sum(matrix[i, :]))

    return result / num_classes


def Iou(input, target):
    """
    input: torch.FloatTensor:(N,C,H,W)
    target: torch.LongTensor:(N,H,W)
    return: Tensor
    """
    assert len(input.size()) == 4
    assert len(target.size()) == 3
    N, H, W = target.size()
    # TP + TN
    arg_max = torch.argmax(input, dim=1)
    TN = torch.sum(input + target == 0)
    # TP / (TP + FP + FN)
    return (torch.sum(arg_max == target) - TN) / (N * H * W - TN)


def mean_iou(input, target):
    """
    input: torch.FloatTensor:(N,C,H,W)
    target: torch.LongTensor:(N,H,W)
    return: Tensor
    """
    assert len(input.size()) == 4
    assert len(target.size()) == 3
    N, num_classes, H, W = input.size()
    arg_max = torch.argmax(input, dim=1)
    # get confusion matrix
    matrix = confusion_matrix(arg_max, target, num_classes)
    result = 0
    for i in range(num_classes):
        nii = matrix[i, i]
        ti, tj = torch.sum(matrix[i, :]), torch.sum(matrix[:, i])
        result += (nii / (ti + tj - nii))

    return result / num_classes

# Todo Read more paper
def frequency_weighted_iou(input, target):
    """
    input: torch.FloatTensor:(N,C,H,W)
    target: torch.LongTensor:(N,H,W)
    return: Tensor
    """
    assert len(input.size()) == 4
    assert len(target.size()) == 3
    N, num_classes, H, W = input.size()
    arg_max = torch.argmax(input, dim=1)
    # get confusion matrix
    matrix = confusion_matrix(arg_max, target, num_classes)
    result = 0
    tk = 0
    for i in range(num_classes):
        nii = matrix[i, i]
        ti, tj = torch.sum(matrix[i, :]), torch.sum(matrix[:, i])
        tk += ti
        result += (ti * nii / (ti + tj - nii))

    return result / tk




