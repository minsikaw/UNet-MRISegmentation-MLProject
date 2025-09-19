import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from src.data.dataset import BrainMRIDataset
from src.models.unet import UNet
from src.metrics import dice_coef


def load_config(config_path):
    """Loads a YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main(args):
    cfg = load_config(args.config)
    device = torch.device(
        cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    )

    # Validation dataset
    val_dataset = BrainMRIDataset(
        cfg['data']['val_images_dir'],
        cfg['data']['val_masks_dir'],
        image_size=cfg['data'].get('image_size', 256)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['data']['batch_size'],
        shuffle=False,
        num_workers=cfg['data'].get('num_workers', 0)
    )

    model = UNet(
        in_channels=cfg['model'].get('in_channels', 1),
        out_channels=cfg['model'].get('out_channels', 1)
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    total_metric = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            metric = dice_coef(outputs, masks)
            total_metric += metric * images.size(0)

    avg_metric = total_metric / len(val_loader.dataset)
    print(f"Validation Dice coefficient: {avg_metric:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a U-Net model on validation data")
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth) file')
    args = parser.parse_args()
    main(args)