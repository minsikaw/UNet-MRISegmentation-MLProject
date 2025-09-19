import torch
import torch.nn.functional as F


def dice_loss(pred, target, smooth: float = 1.0):
    """Computes Dice loss.

    Args:
        pred: Predicted mask tensor with values in [0, 1].
        target: Ground truth mask tensor.
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        Dice loss value.
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    dice = (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1.0 - dice


def bce_dice_loss(pred, target, bce_weight: float = 0.5):
    """Binary cross-entropy combined with Dice loss.

    Args:
        pred: Predicted mask tensor.
        target: Ground truth mask tensor.
        bce_weight: Weight for the BCE component (between 0 and 1).

    Returns:
        Weighted combination of BCE and Dice loss.
    """
    bce = F.binary_cross_entropy(pred, target)
    dl = dice_loss(pred, target)
    return bce_weight * bce + (1.0 - bce_weight) * dl