import torch


def dice_coef(pred, target, threshold: float = 0.5, smooth: float = 1.0):
    """Computes the Dice coefficient.

    Args:
        pred: Predicted mask tensor.
        target: Ground truth mask tensor.
        threshold: Threshold to binarize predictions.
        smooth: Smoothing factor to avoid division by zero.

    Returns:
        Dice coefficient (between 0 and 1).
    """
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()

    pred_bin = pred_bin.contiguous().view(-1)
    target_bin = target_bin.contiguous().view(-1)

    intersection = (pred_bin * target_bin).sum()
    return (2.0 * intersection + smooth) / (pred_bin.sum() + target_bin.sum() + smooth)