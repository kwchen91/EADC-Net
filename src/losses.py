import torch
import torch.nn.functional as F

def dice_loss(logits, targets, eps=1e-6):
    prob = torch.sigmoid(logits)
    inter = (prob * targets).sum(dim=(1,2,3))
    den = prob.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + eps
    return 1 - (2 * inter / den).mean()

def bce_dice_loss(logits, targets, bce_w=0.5, dice_w=0.5):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    d = dice_loss(logits, targets)
    return bce_w * bce + dice_w * d
