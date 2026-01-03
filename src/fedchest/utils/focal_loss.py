import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.):
        super().__init__()
        assert gamma >= 0
        self.register_buffer('weight', torch.tensor(weight))
        self.register_buffer('gamma', torch.tensor(gamma))

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(
            input, target, reduction='none', weight=self.weight
        )
        p = torch.exp(-bce)
        loss = (1 - p) ** self.gamma * bce
        return loss.mean()