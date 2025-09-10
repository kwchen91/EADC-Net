import torch

def dice_coeff(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = (pred * target).sum(dim=(1,2,3))
    den = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + eps
    return (2.0 * inter / den).mean()

def hd95_placeholder(*args, **kwargs) -> float:
    return float("inf")
