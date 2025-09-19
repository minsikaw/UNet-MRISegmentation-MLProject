import argparse
import yaml
import os
import torch
from torch.utils.data import DataLoader

from src.data.dataset import BrainMRIDataset
from src.models.unet import UNet
from src.losses import dice_loss, bce_dice_loss
from src.metrics import dice_coef
from src.engine import train_one_epoch, evaluate
from src.utils.misc import set_seed


def load_config(config_path):
    """Loads a YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_loss_fn(cfg):
    """Returns the appropriate loss function based on the config."""
    loss_cfg = cfg['loss']
    if loss_cfg['type'] == 'dice':
        return dice_loss
    elif loss_cfg['type'] == 'bce-dice':
        bce_weight = loss_cfg.get('bce_weight', 0.5)
        return lambda pred, target: bce_dice_loss(pred, target, bce_weight)
    else:
        raise ValueError(f"Unsupported loss type: {loss_cfg['type']}")


def main(args):
    cfg = load_config(args.config)
    set_seed(cfg.get('seed', 42))

    device = torch.device(
        cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    )

    # Datasets and loaders
    train_dataset = BrainMRIDataset(
        cfg['data']['train_images_dir'],
        cfg['data']['train_masks_dir'],
        image_size=cfg['data'].get('image_size', 256)
    )
    val_dataset = BrainMRIDataset(
        cfg['data']['val_images_dir'],
        cfg['data']['val_masks_dir'],
        image_size=cfg['data'].get('image_size', 256)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['data']['batch_size'],
        shuffle=True,
        num_workers=cfg['data'].get('num_workers', 0)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['data']['batch_size'],
        shuffle=False,
        num_workers=cfg['data'].get('num_workers', 0)
    )

    # Model, loss, optimizer
    model = UNet(
        in_channels=cfg['model'].get('in_channels', 1),
        out_channels=cfg['model'].get('out_channels', 1)
    ).to(device)

    loss_fn = get_loss_fn(cfg)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg['train'].get('learning_rate', 1e-3)
    )

    num_epochs = cfg['train'].get('num_epochs', 50)
    best_metric = 0.0

    checkpoint_dir = cfg['train'].get('checkpoint_dir', 'runs')
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metric = evaluate(
            model, val_loader, lambda pred, target: dice_coef(pred, target), device
        )

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Dice: {val_metric:.4f}")

        if val_metric > best_metric:
            best_metric = val_metric
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved new best model to {checkpoint_path}")

    print("Training finished. Best validation Dice coefficient:", best_metric)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a U-Net on brain MRI data")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')
    args = parser.parse_args()
    main(args)