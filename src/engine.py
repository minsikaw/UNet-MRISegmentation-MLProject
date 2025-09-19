import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(model, loader: DataLoader, optimizer, loss_fn, device):
    """Runs one epoch of training.

    Args:
        model: The neural network model to train.
        loader: DataLoader providing training data.
        optimizer: Optimizer for updating model weights.
        loss_fn: Loss function.
        device: Device to perform computations on.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader: DataLoader, metric_fn, device):
    """Evaluates the model on a validation set.

    Args:
        model: The neural network model.
        loader: DataLoader providing validation data.
        metric_fn: Function that computes a metric (e.g., Dice coefficient).
        device: Device to perform computations on.

    Returns:
        Average metric value across the validation set.
    """
    model.eval()
    metric_total = 0.0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            metric_value = metric_fn(outputs, masks)
            metric_total += metric_value * images.size(0)
    return metric_total / len(loader.dataset)