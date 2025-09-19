import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from src.models.unet import UNet
from src.metrics import dice_coef

VALID_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

def resolve_by_basename(root_dir: Path, basename_or_path: str, subfolder: str) -> Path:
    """
    Resolve an image path inside root_dir/subfolder when user provides either:
    - a full path, or
    - a basename without extension (e.g., TCGA_CS_4941_19960909_4)
    """
    p = Path(basename_or_path)
    # full existing path?
    if p.is_file():
        return p

    # resolve within root_dir/subfolder by trying ext variants
    parent = root_dir / subfolder
    for ext in VALID_EXTS:
        cand = parent / f"{p.stem if p.suffix else p.name}{ext}"
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"Could not resolve file for '{basename_or_path}' under '{parent}'. Tried {', '.join(VALID_EXTS)}.")

def to_tensor_gray(img_np: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    img = cv2.resize(img_np, (size, size), interpolation=cv2.INTER_AREA)
    img = (img.astype(np.float32) / 255.0)[None, None, ...]  # (1,1,H,W)
    return torch.from_numpy(img).to(device)

def binarize(mask_np: np.ndarray, thr: int = 127) -> np.ndarray:
    _, b = cv2.threshold(mask_np, thr, 255, cv2.THRESH_BINARY)
    return b

def draw_contours_on(image_gray: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    gt_bin = binarize(gt_mask)
    pred_bin = binarize(pred_mask)
    overlay = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

    gt_contours, _ = cv2.findContours(gt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pr_contours, _ = cv2.findContours(pred_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(overlay, gt_contours, -1, (0, 255, 0), 2)   # green = GT
    cv2.drawContours(overlay, pr_contours, -1, (0, 0, 255), 2)   # red = prediction
    return overlay

def main():
    ap = argparse.ArgumentParser(description="Predict & overlay tumor contours for a single image")
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--checkpoint", required=True, type=str)
    ap.add_argument("--image", required=True, type=str, help="Path or basename (e.g., TCGA_CS_4941_19960909_4)")
    ap.add_argument("--out_mask", default="outputs/pred_mask.png", type=str)
    ap.add_argument("--out_overlay", default="outputs/pred_overlay.png", type=str)
    args = ap.parse_args()

    cfg = load_config(args.config)

    # device selection: respect cfg if CUDA available, else CPU
    cfg_device = cfg.get("device", None)
    if torch.cuda.is_available() and (cfg_device in (None, "cuda")):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    size = int(cfg["data"].get("image_size", 256))

    # make output dirs
    Path(args.out_mask).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_overlay).parent.mkdir(parents=True, exist_ok=True)

    # locate image & GT mask (match by basename)
    project_root = Path(".").resolve()
    img_path = resolve_by_basename(project_root, args.image, "data/val/images")
    msk_path = resolve_by_basename(project_root, args.image, "data/val/masks")

    # load grayscale
    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    gt_gray = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    if gt_gray is None:
        raise FileNotFoundError(f"Could not read mask:  {msk_path}")

    # build model & load weights
    model = UNet(
        in_channels=cfg["model"].get("in_channels", 1),
        out_channels=cfg["model"].get("out_channels", 1),
    ).to(device)
    # safer future-proof load
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # tensorize input
    x = to_tensor_gray(img_gray, size, device)

    # predict
    with torch.no_grad():
        y = model(x)  # (1,1,H,W), sigmoid probs
        y_np = (y[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255

    # save predicted mask at network resolution
    cv2.imwrite(str(Path(args.out_mask)), y_np)

    # create overlay for visualization at original image scale
    pred_up = cv2.resize(y_np, (img_gray.shape[1], img_gray.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = draw_contours_on(img_gray, gt_gray, pred_up)
    cv2.imwrite(str(Path(args.out_overlay)), overlay)

    # report single-image Dice (at upsampled/original resolution)
    gt_bin = (gt_gray > 127).astype(np.float32)
    pr_bin = (pred_up > 127).astype(np.float32)
    # reuse same dice formula as codebase (quick)
    inter = (gt_bin * pr_bin).sum()
    dice = (2 * inter + 1.0) / (gt_bin.sum() + pr_bin.sum() + 1.0)
    print(f"Saved mask  -> {args.out_mask}")
    print(f"Saved overlay (GT green, Pred red) -> {args.out_overlay}")
    print(f"Single-image Dice: {dice:.4f}")

if __name__ == "__main__":
    main()
