import torch
import torch.nn as nn

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)  
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss
        return focal_loss.mean()