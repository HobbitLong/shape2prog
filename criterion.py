from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
from misc import to_contiguous


class LSTMClassCriterion(nn.Module):
    def __init__(self):
        super(LSTMClassCriterion, self).__init__()

    def forward(self, pred, target, mask):
        # truncate to the same size
        pred = pred.clone()
        target = target.clone()
        mask = mask.clone()

        target = target[:, :pred.size(1)]
        mask = mask[:, :pred.size(1)]

        pred = to_contiguous(pred).view(-1, pred.size(2))
        target = to_contiguous(target).view(-1, 1)
        mask = to_contiguous(mask).view(-1, 1)

        # compute loss
        loss = - pred.gather(1, target) * mask
        loss = torch.sum(loss) / torch.sum(mask)

        # compute accuracy
        _, idx = torch.max(pred, dim=1)
        correct = idx.eq(torch.squeeze(target))
        correct = correct.float() * torch.squeeze(mask)
        accuracy = torch.sum(correct) / torch.sum(mask)
        return loss, accuracy

class LSTMRegressCriterion(nn.Module):
    def __init__(self):
        super(LSTMRegressCriterion, self).__init__()

    def forward(self, pred, target, mask):
        # truncate to the same size
        pred = pred.clone()
        target = target.clone()
        mask = mask.clone()
        target = target[:, :pred.size(1), :]
        mask = mask[:, :pred.size(1), :]
        # compute the loss
        diff = 0.5 * (pred - target) ** 2
        diff = diff * mask
        output = torch.sum(diff) / torch.sum(mask)
        return output


def BatchIoU(s1, s2):
    """
    :param s1: first sets of shapes
    :param s2: second sets of shapes
    :return: IoU
    """
    assert s1.shape[0] == s2.shape[0], "# (shapes1, shapes2) don't match"
    v1 = np.sum(s1 > 0.5, axis=(1, 2, 3))
    v2 = np.sum(s2 > 0.5, axis=(1, 2, 3))
    I = np.sum((s1 > 0.5) * (s2 > 0.5), axis=(1, 2, 3))
    U = v1 + v2 - I
    inds = U == 0
    U[inds] = 1
    I[inds] = 1
    IoU = I.astype(np.float32) / U.astype(np.float32)

    return IoU


def SingleIoU(s1, s2):
    """
    :param s1: shape 1
    :param s2: shape 2
    :return: Iou
    """
    v1 = np.sum(s1 > 0.5)
    v2 = np.sum(s2 > 0.5)
    I = np.sum((s1 > 0.5) * (s2 > 0.5))
    U = v1 + v2 - I
    if U == 0:
        IoU = 1
    else:
        IoU = float(I) / float(U)

    return IoU
