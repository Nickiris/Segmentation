import torch
from torch import nn
import torch.nn.functional as F

"""
    We can build a confusion matrix to calculate the metric result of one data. 
    When adding another data, we need to update the confusion matrix and calculate 
    the new metric result.The code is inspired by the image metrics.jpg in FCN. 
    TN: background pixel
    Pix accuracy: (TP + TN) / (TP + FP + TN + FN)
    iou: TP / (TP + FP + FN)
    mean iou: the mean of all categories iou including background.
    recall: TP / (TP + FN)
"""


def confusion_matrix(input, target, num_classes):
    """
    input: torch.LongTensor:(N, H, W)
    target: torch.LongTensor:(N, H, W)
    num_classes: int
    results:Tensor
    """
    assert torch.max(input) < num_classes
    assert torch.max(target) < num_classes
    H, W = target.size()[-2:]
    results = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for i, j in zip(target.flatten(), input.flatten()):
        results[i, j] += 1
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
    return torch.sum(arg_max == target) / (N * H * W)

def mean_pixel_accuarcy(input, target):
    """
    input: torch.FloatTensor:(N, C, H, W)
    target: torch.LongTensor:(N, H, W)
    return: Tensor
    """
    assert len(input.size()) == 4
    assert len(target.size()) == 3
    N, num_classes, H, W = input.size()
    input = F.softmax(input, dim=1)
    arg_max = torch.argmax(input, dim=1)
    result = 0
    confuse_matrix = confusion_matrix(arg_max, target, num_classes)
    for k in range(num_classes):
        # consider the case where the denominator is zero.
        if confuse_matrix[k, k] == 0:
            continue
        else:
            result += (confuse_matrix[k, k] / torch.sum(confuse_matrix[k,:]))
    return result / num_classes

# TODO: there are some problems.
def iou(input, target):
    """
    input: torch.FloatTensor:(N, C, H, W)
    target: torch.LongTensor:(N, H, W)
    return: Tensor
    """
    assert len(input.size()) == 4
    assert len(target.size()) == 3
    N, H, W = target.size()
    input = F.softmax(input, dim=1)
    result = 0
    arg_max = torch.argmax(input, dim=1)

    TN = torch.sum(arg_max + target == 0)
    # TP / (TP + FP + FN)
    result += (torch.sum(arg_max == target) - TN) / (N * H * W - TN)

    return result

def mean_iou(input, target):
    """
    input: torch.FloatTensor:(N, C, H, W)
    target: torch.LongTensor:(N, H, W)
    return: Tensor
    """
    assert len(input.size()) == 4
    assert len(target.size()) == 3
    N, num_classes, H, W = input.size()
    input = F.softmax(input, dim=1)
    arg_max = torch.argmax(input, dim=1)
    result = 0
    confuse_matrix = confusion_matrix(arg_max, target, num_classes)
    for k in range(num_classes):
        nii = confuse_matrix[k, k]
        # consider the case where the denominator is zero.
        if nii == 0:
            continue
        else:
            ti, tj = torch.sum(confuse_matrix[k, :]), torch.sum(confuse_matrix[:, k])
            result += (nii / (ti + tj - nii))

    return result / num_classes

# Todo: read more paper.
def frequency_weighted_iou(input, target):
    """
    input: torch.FloatTensor:(N, C, H, W)
    target: torch.LongTensor:(N, H, W)
    return: Tensor
    """
    assert len(input.size()) == 4
    assert len(target.size()) == 3
    N, num_classes, H, W = input.size()
    input = F.softmax(input, dim=1)
    arg_max = torch.argmax(input, dim=1)
    # get confusion matrix
    result = 0
    confuse_matrix = confusion_matrix(arg_max, target, num_classes)
    for k in range(num_classes):
        nii = confuse_matrix[k, k]
        if nii == 0:
            continue
        else:
            ti, tj = torch.sum(confuse_matrix[k, :]), torch.sum(confuse_matrix[:, k])
            result += (ti * nii / (ti + tj - nii))

    return result / torch.sum(confuse_matrix)




