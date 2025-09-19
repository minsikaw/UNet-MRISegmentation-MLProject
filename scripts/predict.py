import argparse
import yaml
import torch
import cv2
import numpy as np
from pathlib import Path
from src.models.unet import UNet

VALID_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

def resolve_image_path(path_str: str) -> str:
    p = Path(path_str)
    if p.is_file():
        return str(p)
    # If the given path doesn't exist, try to resolve by basename across common extensions
    parent = p.parent if p.parent.as_posix() not in (".", "") else Path(".")
    base = p.stem if p.suffix else p.name
    for ext in VALID_EXTS:
        candidate = parent / f"{base}{ext}"
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        f"Could not find image for base '{base}' in '{parent}'. "
        f"Tried extensions: {', '.join(VALID_EXTS)}"
    )

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

def main(args):
    cfg = load_config(args.config)
    device = torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    model = UNet(
        in_channels=cfg['model'].get('in_channels', 1),
        out_channels=cfg['model'].get('out_channels', 1)
    )
    # safer load (future-proof)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    # resolve the image path (handle basename or wrong extension)
    img_path = resolve_image_path(args.image_path)

    # Load image
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {img_path}. Check path/extension.")

    size = cfg['data'].get('image_size', 256)
    image = cv2.resize(image, (size, size))
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=(0, 1))  # (1,1,H,W)

    image_tensor = torch.tensor(image, device=device)

    with torch.no_grad():
        output = model(image_tensor)
        mask = (output[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255

    # Ensure output dir exists
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_path), mask)
    print(f"Segmentation mask saved to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict mask for a single image using a trained U-Net model")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True, help="Path or basename of the image")
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    main(args)
