import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CELoss(nn.Module):
    def __init__(self, ignore_label: int = None, weight: np.ndarray = None):
        '''
        :param ignore_label: label to ignore
        :param weight: possible weights for weighted CE Loss
        '''
        super().__init__()
        if weight is not None:
            weight = torch.from_numpy(weight).float()
            print(f'----->Using weighted CE Loss weights: {weight}')

        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_label, weight=weight)
        self.ignored_label = ignore_label

    def forward(self, preds: torch.Tensor, gt: torch.Tensor):

        loss = self.loss(preds, gt)
        return loss


class DICELoss(nn.Module):

    def __init__(self, ignore_label=None, powerize=True, use_tmask=True):
        super(DICELoss, self).__init__()

        if ignore_label is not None:
            self.ignore_label = torch.tensor(ignore_label)
        else:
            self.ignore_label = ignore_label

        self.powerize = powerize
        self.use_tmask = use_tmask

    def forward(self, output, target):
        input_device = output.device
        # temporal solution to avoid nan
        output = output.cpu()
        target = target.cpu()

        if self.ignore_label is not None:
            valid_idx = torch.logical_not(target == self.ignore_label)
            target = target[valid_idx]
            output = output[valid_idx, :]

        target = F.one_hot(target, num_classes=output.shape[1])
        output = F.softmax(output, dim=-1)

        intersection = (output * target).sum(dim=0)
        if self.powerize:
            union = (output.pow(2).sum(dim=0) + target.sum(dim=0)) + 1e-12
        else:
            union = (output.sum(dim=0) + target.sum(dim=0)) + 1e-12
        if self.use_tmask:
            tmask = (target.sum(dim=0) > 0).int()
        else:
            tmask = torch.ones(target.shape[1]).int()

        iou = (tmask * 2 * intersection / union).sum(dim=0) / (tmask.sum(dim=0) + 1e-12)

        dice_loss = 1 - iou.mean()

        return dice_loss.to(input_device)


def get_soft(t_vector, eps=0.25):

    max_val = 1 - eps
    min_val = eps / (t_vector.shape[-1] - 1)

    t_soft = torch.empty(t_vector.shape)
    t_soft[t_vector == 0] = min_val
    t_soft[t_vector == 1] = max_val

    return t_soft


def get_kitti_soft(t_vector, labels, eps=0.25):

    max_val = 1 - eps
    min_val = eps / (t_vector.shape[-1] - 1)

    t_soft = torch.empty(t_vector.shape)
    t_soft[t_vector == 0] = min_val
    t_soft[t_vector == 1] = max_val

    searched_idx = torch.logical_or(labels == 6, labels == 1)
    if searched_idx.sum() > 0:
        t_soft[searched_idx, 1] = max_val/2
        t_soft[searched_idx, 6] = max_val/2

    return t_soft


class SoftDICELoss(nn.Module):

    def __init__(self, ignore_label=None, powerize=True, use_tmask=True,
                 neg_range=False, eps=0.05, is_kitti=False):
        super(SoftDICELoss, self).__init__()

        if ignore_label is not None:
            self.ignore_label = torch.tensor(ignore_label)
        else:
            self.ignore_label = ignore_label
        self.powerize = powerize
        self.use_tmask = use_tmask
        self.neg_range = neg_range
        self.eps = eps
        self.is_kitti = is_kitti

    def forward(self, output, target, return_class=False, is_kitti=False):
        input_device = output.device
        # temporal solution to avoid nan
        output = output.cpu()
        target = target.cpu()

        if self.ignore_label is not None:
            valid_idx = torch.logical_not(target == self.ignore_label)
            target = target[valid_idx]
            output = output[valid_idx, :]

        target_onehot = F.one_hot(target, num_classes=output.shape[1])
        if not self.is_kitti and not is_kitti:
            target_soft = get_soft(target_onehot, eps=self.eps)
        else:
            target_soft = get_kitti_soft(target_onehot, target, eps=self.eps)

        output = F.softmax(output, dim=-1)

        intersection = (output * target_soft).sum(dim=0)

        if self.powerize:
            union = (output.pow(2).sum(dim=0) + target_soft.sum(dim=0)) + 1e-12
        else:
            union = (output.sum(dim=0) + target_soft.sum(dim=0)) + 1e-12
        if self.use_tmask:
            tmask = (target_onehot.sum(dim=0) > 0).int()
        else:
            tmask = torch.ones(target_onehot.shape[1]).int()

        iou = (tmask * 2 * intersection / union).sum(dim=0) / (tmask.sum(dim=0) + 1e-12)
        iou_class = tmask * 2 * intersection / union

        if self.neg_range:
            dice_loss = -iou.mean()
            dice_class = -iou_class
        else:
            dice_loss = 1 - iou.mean()
            dice_class = 1 - iou_class
        if return_class:
            return dice_loss.to(input_device), dice_class
        else:
            return dice_loss.to(input_device)

