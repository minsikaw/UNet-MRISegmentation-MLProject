# scripts/gui_predictor.py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import sys
import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
import yaml
import random
import csv

# ---- Make project root importable even if launched from elsewhere ----
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.unet import UNet  # noqa: E402

VALID_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

def safe_load_yaml(path: Path):
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def to_tensor_gray(img_np: np.ndarray, size: int, device: torch.device) -> torch.Tensor:
    img = cv2.resize(img_np, (size, size), interpolation=cv2.INTER_AREA)
    img = (img.astype(np.float32) / 255.0)[None, None, ...]  # (1,1,H,W)
    return torch.from_numpy(img).to(device)

def draw_contours(overlay_src_gray: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray, thickness: int) -> np.ndarray:
    def binarize(x):
        _, b = cv2.threshold(x, 127, 255, cv2.THRESH_BINARY)
        return b
    gt_bin = binarize(gt_mask)
    pr_bin = binarize(pred_mask)

    overlay = cv2.cvtColor(overlay_src_gray, cv2.COLOR_GRAY2BGR)
    gt_contours, _ = cv2.findContours(gt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pr_contours, _ = cv2.findContours(pr_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, gt_contours, -1, (0, 255, 0), thickness)  # green = GT
    cv2.drawContours(overlay, pr_contours, -1, (0, 0, 255), thickness)  # red   = Pred
    return overlay

def dice_np(gt_bin: np.ndarray, pr_bin: np.ndarray, smooth: float = 1.0) -> float:
    inter = (gt_bin * pr_bin).sum()
    denom = gt_bin.sum() + pr_bin.sum()
    return float((2.0 * inter + smooth) / (denom + smooth)) if denom > 0 else 1.0

def list_pairs(images_dir: Path, masks_dir: Path):
    imgs = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    mask_index = {m.stem: m for m in masks_dir.iterdir() if m.is_file() and m.suffix.lower() in VALID_EXTS}
    pairs = []
    for img in imgs:
        m = mask_index.get(img.stem)
        if m is not None and m.exists():
            pairs.append((img.stem, img, m))
    return pairs

def next_trial_dir(out_root: Path) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    k = 1
    while True:
        d = out_root / f"trial_{k}"
        if not d.exists():
            (d / "masks").mkdir(parents=True, exist_ok=True)
            (d / "overlays").mkdir(parents=True, exist_ok=True)
            return d
        k += 1

class UNetGUI(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        master.title("U-Net MRI Segmentation — GUI")
        self.pack(fill="both", expand=True)

        # state
        self.images_dir: Path | None = None
        self.masks_dir: Path | None = None
        self.checkpoint: Path | None = None
        self.out_root: Path = project_root / "outputs_gui"
        self.n_samples: int = 10
        self.line_thickness: int = 1
        self.seed: int | None = 42

        self.cfg = safe_load_yaml(project_root / "configs" / "baseline.yaml")
        self.image_size = int(self.cfg.get("data", {}).get("image_size", 256))

        if torch.cuda.is_available() and self.cfg.get("device", "cuda") == "cuda":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model: UNet | None = None
        self.outputs = []  # list of (overlay_path, mask_path)
        self.cur_idx = 0

        self._build_ui()

    def _build_ui(self):
        # controls
        row = 0
        frm = ttk.Frame(self)
        frm.grid(row=row, column=0, sticky="w", padx=8, pady=6)

        ttk.Button(frm, text="Validation Images…", command=self.pick_images).grid(row=0, column=0, padx=4)
        ttk.Button(frm, text="Validation Masks…", command=self.pick_masks).grid(row=0, column=1, padx=4)
        ttk.Button(frm, text="Checkpoint (.pth)…", command=self.pick_ckpt).grid(row=0, column=2, padx=4)
        ttk.Button(frm, text="Output Root…", command=self.pick_out_root).grid(row=0, column=3, padx=4)

        row += 1
        frm2 = ttk.Frame(self)
        frm2.grid(row=row, column=0, sticky="w", padx=8)

        ttk.Label(frm2, text="# Images:").grid(row=0, column=0, padx=4)
        self.n_var = tk.IntVar(value=self.n_samples)
        ttk.Spinbox(frm2, from_=1, to=200, textvariable=self.n_var, width=6).grid(row=0, column=1, padx=4)

        ttk.Label(frm2, text="Line thickness:").grid(row=0, column=2, padx=12)
        self.th_var = tk.IntVar(value=self.line_thickness)
        ttk.Spinbox(frm2, from_=1, to=6, textvariable=self.th_var, width=6).grid(row=0, column=3, padx=4)

        ttk.Label(frm2, text="Seed:").grid(row=0, column=4, padx=12)
        self.seed_var = tk.IntVar(value=self.seed if self.seed is not None else 0)
        ttk.Spinbox(frm2, from_=0, to=10_000, textvariable=self.seed_var, width=8).grid(row=0, column=5, padx=4)

        ttk.Button(frm2, text="Run", command=self.run_trial).grid(row=0, column=6, padx=12)

        # preview
        row += 1
        self.preview = ttk.Label(self)
        self.preview.grid(row=row, column=0, padx=8, pady=8)

        # navigation
        row += 1
        nav = ttk.Frame(self)
        nav.grid(row=row, column=0)
        ttk.Button(nav, text="◀ Prev", command=self.prev_img).grid(row=0, column=0, padx=10, pady=6)
        ttk.Button(nav, text="Next ▶", command=self.next_img).grid(row=0, column=1, padx=10, pady=6)

        # status
        row += 1
        self.status = ttk.Label(self, text="Select folders and checkpoint, then click Run.")
        self.status.grid(row=row, column=0, sticky="w", padx=8, pady=6)

    # pickers
    def pick_images(self):
        d = filedialog.askdirectory(title="Select validation IMAGES folder")
        if d:
            self.images_dir = Path(d)
            messagebox.showinfo("Selected", f"Images: {self.images_dir}")

    def pick_masks(self):
        d = filedialog.askdirectory(title="Select validation MASKS folder")
        if d:
            self.masks_dir = Path(d)
            messagebox.showinfo("Selected", f"Masks: {self.masks_dir}")

    def pick_ckpt(self):
        f = filedialog.askopenfilename(title="Select checkpoint (.pth)", filetypes=[("PyTorch checkpoint", "*.pth")])
        if f:
            self.checkpoint = Path(f)
            messagebox.showinfo("Selected", f"Checkpoint: {self.checkpoint}")

    def pick_out_root(self):
        d = filedialog.askdirectory(title="Select output root folder")
        if d:
            self.out_root = Path(d)
            messagebox.showinfo("Selected", f"Output root: {self.out_root}")

    # trial
    def run_trial(self):
        try:
            self._run_trial_inner()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _run_trial_inner(self):
        if self.images_dir is None or self.masks_dir is None or self.checkpoint is None:
            raise RuntimeError("Please select validation images, masks, and a checkpoint first.")

        if not self.images_dir.exists() or not self.masks_dir.exists():
            raise FileNotFoundError("Images or masks folder does not exist.")

        n = int(self.n_var.get())
        thickness = int(self.th_var.get())
        seed = int(self.seed_var.get())
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        # load model
        self.status.config(text="Loading model…")
        self.model = UNet(
            in_channels=self.cfg.get("model", {}).get("in_channels", 1),
            out_channels=self.cfg.get("model", {}).get("out_channels", 1),
        ).to(self.device)
        try:
            state = torch.load(self.checkpoint, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(self.checkpoint, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        # build pairs and sample
        pairs = list_pairs(self.images_dir, self.masks_dir)
        if not pairs:
            raise RuntimeError("No image-mask pairs with matching basenames were found.")
        sample = random.sample(pairs, k=min(n, len(pairs)))

        trial_dir = next_trial_dir(self.out_root)
        masks_dir = trial_dir / "masks"
        overlays_dir = trial_dir / "overlays"
        metrics_csv = trial_dir / "metrics.csv"
        summary_txt = trial_dir / "summary.txt"

        size = self.image_size
        dice_vals = []
        rows = [("basename", "dice")]

        self.outputs.clear()
        self.cur_idx = 0

        for base, img_path, msk_path in sample:
            img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            gt_gray  = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
            if img_gray is None or gt_gray is None:
                continue

            x = to_tensor_gray(img_gray, size, self.device)
            with torch.no_grad():
                y = self.model(x)
                y_np = (y[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255

            # upsample to original size
            pred_up = cv2.resize(y_np, (img_gray.shape[1], img_gray.shape[0]), interpolation=cv2.INTER_NEAREST)

            # dice
            d = dice_np((gt_gray > 127).astype(np.float32), (pred_up > 127).astype(np.float32))
            dice_vals.append(d)
            rows.append((base, f"{d:.6f}"))

            # save outputs
            out_mask = masks_dir / f"{base}.png"
            out_overlay = overlays_dir / f"{base}.png"
            cv2.imwrite(str(out_mask), pred_up)
            overlay = draw_contours(img_gray, gt_gray, pred_up, thickness=thickness)
            cv2.imwrite(str(out_overlay), overlay)

            self.outputs.append(out_overlay)

        # metrics
        with open(metrics_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        mean_dice = float(np.mean(dice_vals)) if dice_vals else 0.0
        with open(summary_txt, "w") as f:
            f.write(f"Images: {len(dice_vals)}\n")
            f.write(f"Mean Dice: {mean_dice:.6f}\n")

        self.status.config(text=f"Trial saved to: {trial_dir} | Mean Dice: {mean_dice:.4f}")
        if self.outputs:
            self.show_image(self.outputs[0])

    # viewer
    def show_image(self, path: Path):
        img = Image.open(path)
        # fit reasonably into the window:
        img.thumbnail((700, 700), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(img)
        self.preview.configure(image=imgtk)
        self.preview.image = imgtk  # keep reference

    def prev_img(self):
        if not self.outputs: return
        self.cur_idx = (self.cur_idx - 1) % len(self.outputs)
        self.show_image(self.outputs[self.cur_idx])

    def next_img(self):
        if not self.outputs: return
        self.cur_idx = (self.cur_idx + 1) % len(self.outputs)
        self.show_image(self.outputs[self.cur_idx])


if __name__ == "__main__":
    root = tk.Tk()
    app = UNetGUI(root)
    root.mainloop()
