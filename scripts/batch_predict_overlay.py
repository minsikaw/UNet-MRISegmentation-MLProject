# scripts/batch_predict_overlay.py

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

# --- Make the project root importable when running this file directly ---
project_root = Path(__file__).resolve().parents[1]  # .../unet_mri_project
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.unet import UNet  # noqa: E402

# -----------------------------------------------------------------------

VALID_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def list_image_mask_pairs(val_images_dir: Path, val_masks_dir: Path):
    """
    Return list of (basename, image_path, mask_path) where both exist.
    """
    imgs = [
        p for p in val_images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    ]
    mask_index = {
        m.stem: m for m in val_masks_dir.iterdir()
        if m.is_file() and m.suffix.lower() in VALID_EXTS
    }

    pairs = []
    for img in imgs:
        base = img.stem
        m = mask_index.get(base)
        if m is not None and m.exists():
            pairs.append((base, img, m))
    return pairs


def to_tensor_gray(img_np: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    img = cv2.resize(img_np, (size, size), interpolation=cv2.INTER_AREA)
    img = (img.astype(np.float32) / 255.0)[None, None, ...]  # (1,1,H,W)
    return torch.from_numpy(img).to(device)


def dice_np(gt_bin: np.ndarray, pr_bin: np.ndarray, smooth: float = 1.0) -> float:
    inter = (gt_bin * pr_bin).sum()
    denom = gt_bin.sum() + pr_bin.sum()
    return float((2.0 * inter + smooth) / (denom + smooth)) if denom > 0 else 1.0


def draw_contours_on(image_gray: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    def binarize(x):
        _, b = cv2.threshold(x, 127, 255, cv2.THRESH_BINARY)
        return b

    gt_bin = binarize(gt_mask)
    pr_bin = binarize(pred_mask)

    overlay = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    gt_contours, _ = cv2.findContours(gt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pr_contours, _ = cv2.findContours(pr_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(overlay, gt_contours, -1, (0, 255, 0), 1)  # green = GT
    cv2.drawContours(overlay, pr_contours, -1, (0, 0, 255), 1)  # red   = Pred
    return overlay


def next_trial_dir(out_root: Path) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    k = 1
    while True:
        cand = out_root / f"trial_{k}"
        if not cand.exists():
            (cand / "masks").mkdir(parents=True, exist_ok=True)
            (cand / "overlays").mkdir(parents=True, exist_ok=True)
            return cand
        k += 1


def main():
    ap = argparse.ArgumentParser(description="Run random validation predictions & overlays as a trial.")
    ap.add_argument("--config", required=True, type=str, help="Path to YAML config")
    ap.add_argument("--checkpoint", required=True, type=str, help="Path to best_model.pth")
    ap.add_argument("--n", type=int, default=10, help="How many random validation samples")
    ap.add_argument("--out_dir", type=str, default="outputs", help="Root output folder (trial_X will be created inside)")
    ap.add_argument("--seed", type=int, default=None, help="Optional seed for reproducibility")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    cfg = load_config(args.config)
    size = int(cfg["data"].get("image_size", 256))

    # Device: honor config if CUDA is available; otherwise fall back to CPU
    cfg_dev = cfg.get("device", None)
    if torch.cuda.is_available() and (cfg_dev in (None, "cuda")):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # Resolve validation paths (relative to project root or absolute)
    val_images_dir = (project_root / cfg["data"]["val_images_dir"]).resolve() if not Path(cfg["data"]["val_images_dir"]).is_absolute() else Path(cfg["data"]["val_images_dir"])
    val_masks_dir = (project_root / cfg["data"]["val_masks_dir"]).resolve() if not Path(cfg["data"]["val_masks_dir"]).is_absolute() else Path(cfg["data"]["val_masks_dir"])

    if not val_images_dir.exists() or not val_masks_dir.exists():
        raise FileNotFoundError(f"Validation paths not found:\n  images: {val_images_dir}\n  masks : {val_masks_dir}")

    # Build list of pairs and sample
    pairs = list_image_mask_pairs(val_images_dir, val_masks_dir)
    if not pairs:
        raise RuntimeError("No image-mask pairs found in validation directories.")
    sample = random.sample(pairs, k=min(args.n, len(pairs)))

    # Create trial folder
    out_root = (project_root / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    trial_dir = next_trial_dir(out_root)
    masks_dir = trial_dir / "masks"
    overlays_dir = trial_dir / "overlays"
    metrics_csv = trial_dir / "metrics.csv"
    summary_txt = trial_dir / "summary.txt"

    # Build model + load weights (future-proof weights_only=True if available)
    model = UNet(
        in_channels=cfg["model"].get("in_channels", 1),
        out_channels=cfg["model"].get("out_channels", 1),
    ).to(device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        state = torch.load(str(ckpt_path), map_location=device, weights_only=True)  # PyTorch >= 2.4 warning-safe
    except TypeError:
        state = torch.load(str(ckpt_path), map_location=device)  # fallback for older PyTorch
    model.load_state_dict(state)
    model.eval()

    # Process samples
    rows = [("basename", "dice")]
    dice_vals = []

    for base, img_path, msk_path in sample:
        # read grayscale
        img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        gt_gray = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None or gt_gray is None:
            print(f"[WARN] Skipping {base}: could not read image or mask.")
            continue

        # forward pass
        x = to_tensor_gray(img_gray, size, device)
        with torch.no_grad():
            y = model(x)
            y_np = (y[0, 0].detach().cpu().numpy() > 0.5).astype(np.uint8) * 255

        # upsample prediction back to original resolution
        pred_up = cv2.resize(y_np, (img_gray.shape[1], img_gray.shape[0]), interpolation=cv2.INTER_NEAREST)

        # per-image Dice
        gt_bin = (gt_gray > 127).astype(np.float32)
        pr_bin = (pred_up > 127).astype(np.float32)
        d = dice_np(gt_bin, pr_bin)
        dice_vals.append(d)
        rows.append((base, f"{d:.6f}"))

        # save outputs
        cv2.imwrite(str(masks_dir / f"{base}.png"), pred_up)
        overlay = draw_contours_on(img_gray, gt_gray, pred_up)
        cv2.imwrite(str(overlays_dir / f"{base}.png"), overlay)

    # write metrics & summary
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    mean_dice = float(np.mean(dice_vals)) if dice_vals else 0.0
    with open(summary_txt, "w") as f:
        f.write(f"Images: {len(dice_vals)}\n")
        f.write(f"Mean Dice: {mean_dice:.6f}\n")

    print(f"\nTrial folder: {trial_dir}")
    print(f"  Masks   -> {masks_dir}")
    print(f"  Overlays-> {overlays_dir}")
    print(f"  Metrics -> {metrics_csv}")
    print(f"  Summary -> {summary_txt}")
    print(f"Mean Dice on sampled {len(dice_vals)} images: {mean_dice:.4f}")


if __name__ == "__main__":
    main()
