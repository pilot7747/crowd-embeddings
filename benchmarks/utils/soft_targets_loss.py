import torch
from torch import nn
import torch.nn.functional as F


def soft_targets_loss(logits, target, reduction='mean'):
    losses = torch.sum(F.log_softmax(logits, dim=-1) * target, dim=1)
    if reduction == 'mean':
        return -torch.mean(losses)
    elif reduction == 'sum':
        return -torch.sum(losses)
    else:
        raise ValueError('Reduction "{}" is not supported.'.format(reduction))


class SoftTargetsLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(SoftTargetsLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        return soft_targets_loss(input, target, self.reduction)
