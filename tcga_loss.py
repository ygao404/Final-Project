import torch
import torch.nn as nn

class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss function optimized for hard examples
    gamma: Focusing parameter. Larger gamma makes the model focus more on hard examples (default recommended: 2.0)
    """
    def __init__(self, gamma=2.0, pos_weight=None):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    def forward(self, inputs, targets):
        # Compute base BCE Loss (retains class balancing effect of pos_weight)
        bce_loss = self.bce(inputs, targets)
        # Compute model's prediction confidence for true labels (pt)
        pt = torch.exp(-bce_loss)  
        # Dynamically modulate weights: more accurate predictions get lower weights; worse predictions get exponentially higher weights
        focal_loss = ((1 - pt) ** self.gamma) * bce_loss
        return focal_loss.mean()