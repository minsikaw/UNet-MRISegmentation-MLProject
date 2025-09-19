# scripts/gui_predictor.py
# GUI predictor with in-window legend: Green = Ground Truth, Red = Prediction

import os
import sys
from pathlib import Path
import threading

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import cv2
from PIL import Image, ImageTk

import torch
import torch.nn as nn

# --- Ensure we can import src/... when running from project root ---
# If this file is in scripts/, project root = parent of this file's folder
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import your U-Net model
try:
    from src.models.unet import UNet  # expects UNet(in_channels=1,out_channels=1)
except Exception as e:
    raise ImportError(
        "Could not import src.models.unet.UNet. "
        "Make sure your project structure has src/models/unet.py "
        "and you run this from project root."
    ) from e


# --------- Configuration (adjust if needed) ----------
INPUT_SIZE = 256        # model input H=W
THRESHOLD = 0.5         # default probability threshold
PRED_COLOR = (0, 0, 255)   # BGR: red for prediction
GT_COLOR = (0, 255, 0)     # BGR: green for ground truth
LINE_THICKNESS = 1
# ----------------------------------------------------


def load_model(checkpoint_path: Path, device: torch.device):
    model = UNet(in_channels=1, out_channels=1)
    model.to(device)
    model.eval()
    # Safe load: try weights_only=True (PyTorch 2.5+), else fallback
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
        # Accept either pure state_dict or dict with key
        if "model_state_dict" in state:
            state = state["model_state_dict"]
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    return model


def preprocess_image(img_path: Path):
    """Read grayscale, resize to INPUT_SIZE, normalize to [0,1], return np and tensor."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    h0, w0 = img.shape[:2]
    img_resized = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    img_norm = img_resized.astype(np.float32) / 255.0
    ten = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return img, img_resized, ten


def read_mask(mask_path: Path, target_size=(INPUT_SIZE, INPUT_SIZE)):
    """Read GT mask if available; return resized 0/1 np.uint8 mask or None."""
    if mask_path is None or not Path(mask_path).exists():
        return None
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    m = cv2.resize(m, target_size, interpolation=cv2.INTER_NEAREST)
    m_bin = (m > 127).astype(np.uint8)
    return m_bin


def predict_mask(model: nn.Module, ten: torch.Tensor, device: torch.device, threshold=THRESHOLD):
    with torch.no_grad():
        ten = ten.to(device)
        logits = model(ten)  # [1,1,H,W]
        prob = torch.sigmoid(logits)
        pred = (prob > threshold).float()
    pred_mask = pred.squeeze().cpu().numpy().astype(np.uint8)  # [H,W] 0/1
    prob_map = prob.squeeze().cpu().numpy().astype(np.float32)  # for potential use
    return pred_mask, prob_map


def draw_contours_on_image(gray_resized: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray | None):
    """Return a BGR overlay image with red (pred) and green (GT) contours."""
    # Base RGB image for display: convert grayscale to BGR
    vis = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2BGR)

    # Find contours for prediction
    contours_pred, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_pred:
        cv2.drawContours(vis, contours_pred, -1, PRED_COLOR, thickness=LINE_THICKNESS)

    # GT contours
    if gt_mask is not None:
        contours_gt, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_gt:
            cv2.drawContours(vis, contours_gt, -1, GT_COLOR, thickness=LINE_THICKNESS)

    return vis


def to_tk_image(bgr_img: np.ndarray, max_display=512):
    """Fit image into a square canvas preserving aspect; convert to Tk image."""
    h, w = bgr_img.shape[:2]
    scale = min(max_display / max(h, w), 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    disp = cv2.resize(bgr_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return ImageTk.PhotoImage(pil)


class GUIPredictor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("U-Net MRI Segmentation — GUI Predictor")
        self.geometry("1100x700")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        # Paths
        self.img_dir = tk.StringVar()
        self.mask_dir = tk.StringVar()
        self.ckpt_path = tk.StringVar()
        self.out_dir = tk.StringVar(value=str(ROOT / "outputs_gui"))

        # UI layout
        self._build_top_controls()
        self._build_center()
        self._build_bottom_legend()  # <-- Legend here

        # Data
        self.image_list = []
        self.current_preview = None

    def _build_top_controls(self):
        frm = ttk.Frame(self, padding=8)
        frm.pack(side=tk.TOP, fill=tk.X)

        # Image folder
        ttk.Label(frm, text="Images:").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.img_dir, width=60).grid(row=0, column=1, sticky="we", padx=4)
        ttk.Button(frm, text="Browse", command=self._browse_images).grid(row=0, column=2, padx=4)

        # Mask folder
        ttk.Label(frm, text="Masks:").grid(row=1, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.mask_dir, width=60).grid(row=1, column=1, sticky="we", padx=4)
        ttk.Button(frm, text="Browse", command=self._browse_masks).grid(row=1, column=2, padx=4)

        # Checkpoint
        ttk.Label(frm, text="Checkpoint (.pth):").grid(row=2, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.ckpt_path, width=60).grid(row=2, column=1, sticky="we", padx=4)
        ttk.Button(frm, text="Select", command=self._browse_ckpt).grid(row=2, column=2, padx=4)

        # Output folder
        ttk.Label(frm, text="Output:").grid(row=3, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.out_dir, width=60).grid(row=3, column=1, sticky="we", padx=4)
        ttk.Button(frm, text="Browse", command=self._browse_out).grid(row=3, column=2, padx=4)

        # Load model button
        ttk.Button(frm, text=f"Load Model ({self.device})", command=self._load_model_clicked).grid(row=0, column=3, rowspan=2, padx=8)

        # Refresh list
        ttk.Button(frm, text="Refresh Images", command=self._refresh_list).grid(row=2, column=3, padx=8)

        # Predict buttons
        ttk.Button(frm, text="Predict Selected", command=self._predict_selected).grid(row=3, column=3, padx=8)

        for c in range(4):
            frm.grid_columnconfigure(c, weight=1)

    def _build_center(self):
        body = ttk.Frame(self, padding=8)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left: listbox of images
        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(left, text="Images").pack(anchor="w")
        self.listbox = tk.Listbox(left, width=50, height=28)
        self.listbox.pack(fill=tk.Y, expand=False)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        # Right: preview area
        right = ttk.Frame(body)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.img_canvas = tk.Label(right, background="#202020")
        self.img_canvas.pack(fill=tk.BOTH, expand=True)

    def _build_bottom_legend(self):
        # Legend bar at bottom (inside GUI, not stamped on images)
        leg = ttk.Frame(self, padding=8)
        leg.pack(side=tk.BOTTOM, fill=tk.X)

        # Create colored squares using small Canvas widgets
        pred_box = tk.Canvas(leg, width=18, height=18, highlightthickness=0)
        pred_box.create_rectangle(0, 0, 18, 18, fill="#FF0000", outline="")
        pred_box.pack(side=tk.LEFT)
        ttk.Label(leg, text="Prediction (Red)").pack(side=tk.LEFT, padx=(4, 16))

        gt_box = tk.Canvas(leg, width=18, height=18, highlightthickness=0)
        gt_box.create_rectangle(0, 0, 18, 18, fill="#00FF00", outline="")
        gt_box.pack(side=tk.LEFT)
        ttk.Label(leg, text="Ground Truth (Green)").pack(side=tk.LEFT, padx=4)

        ttk.Label(leg, text="   |   Thin contours").pack(side=tk.LEFT, padx=8)

    # ---- Browse & data loading ----
    def _browse_images(self):
        d = filedialog.askdirectory(title="Select Images Folder")
        if d:
            self.img_dir.set(d)
            self._refresh_list()

    def _browse_masks(self):
        d = filedialog.askdirectory(title="Select Masks Folder")
        if d:
            self.mask_dir.set(d)

    def _browse_ckpt(self):
        f = filedialog.askopenfilename(title="Select Checkpoint", filetypes=[("PyTorch", "*.pth *.pt"), ("All", "*.*")])
        if f:
            self.ckpt_path.set(f)

    def _browse_out(self):
        d = filedialog.askdirectory(title="Select Output Folder")
        if d:
            self.out_dir.set(d)

    def _refresh_list(self):
        img_dir = Path(self.img_dir.get())
        self.listbox.delete(0, tk.END)
        self.image_list = []
        if img_dir.exists():
            for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
                for p in sorted(img_dir.glob(f"*{ext}")):
                    self.image_list.append(p)
            for p in self.image_list:
                self.listbox.insert(tk.END, p.name)

    def _load_model_clicked(self):
        ckpt = Path(self.ckpt_path.get())
        if not ckpt.exists():
            messagebox.showerror("Error", "Please select a valid checkpoint (.pth).")
            return
        try:
            self.model = load_model(ckpt, self.device)
            messagebox.showinfo("Model", f"Loaded model from:\n{ckpt}\nDevice: {self.device}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{e}")

    def _on_select(self, event=None):
        idxs = self.listbox.curselection()
        if not idxs:
            return
        idx = idxs[0]
        img_path = self.image_list[idx]
        self._run_preview(img_path)

    # ---- Prediction & preview ----
    def _predict_selected(self):
        idxs = self.listbox.curselection()
        if not idxs:
            messagebox.showwarning("Select", "Please select an image in the list.")
            return
        if self.model is None:
            messagebox.showwarning("Model", "Load a model checkpoint first.")
            return
        img_path = self.image_list[idxs[0]]
        self._run_save(img_path)

    def _run_preview(self, img_path: Path):
        """Threaded preview to keep UI responsive"""
        threading.Thread(target=self._preview_task, args=(img_path,), daemon=True).start()

    def _preview_task(self, img_path: Path):
        try:
            # Read & preprocess
            gray_orig, gray_resized, ten = preprocess_image(img_path)
            # Find GT mask path (same stem) if available
            mdir = Path(self.mask_dir.get()) if self.mask_dir.get() else None
            gt = None
            if mdir and mdir.exists():
                for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
                    cand = mdir / (img_path.stem + ext)
                    if cand.exists():
                        gt = read_mask(cand, (INPUT_SIZE, INPUT_SIZE))
                        break

            # If no model loaded, just show the original image
            if self.model is None:
                vis = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2BGR)
            else:
                pred_mask, _ = predict_mask(self.model, ten, self.device, THRESHOLD)
                vis = draw_contours_on_image(gray_resized, pred_mask, gt)

            tk_img = to_tk_image(vis, max_display=700)

            def _update():
                self.img_canvas.configure(image=tk_img)
                self.img_canvas.image = tk_img
                self.current_preview = (img_path, vis)
            self.after(0, _update)

        except Exception as e:
            messagebox.showerror("Preview error", str(e))

    def _run_save(self, img_path: Path):
        threading.Thread(target=self._save_task, args=(img_path,), daemon=True).start()

    def _save_task(self, img_path: Path):
        try:
            if self.model is None:
                messagebox.showwarning("Model", "Load a model first.")
                return

            out_dir = Path(self.out_dir.get())
            (out_dir / "masks").mkdir(parents=True, exist_ok=True)
            (out_dir / "overlays").mkdir(parents=True, exist_ok=True)

            # Prepare
            gray_orig, gray_resized, ten = preprocess_image(img_path)
            pred_mask, prob = predict_mask(self.model, ten, self.device, THRESHOLD)

            # Save predicted mask at input-size resolution
            pred_u8 = (pred_mask * 255).astype(np.uint8)
            mask_out = out_dir / "masks" / f"{img_path.stem}_pred.png"
            cv2.imwrite(str(mask_out), pred_u8)

            # Try load GT for overlay (optional)
            gt = None
            mdir = Path(self.mask_dir.get()) if self.mask_dir.get() else None
            if mdir and mdir.exists():
                for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
                    cand = mdir / (img_path.stem + ext)
                    if cand.exists():
                        gt = read_mask(cand, (INPUT_SIZE, INPUT_SIZE))
                        break

            overlay = draw_contours_on_image(gray_resized, pred_mask, gt)
            over_out = out_dir / "overlays" / f"{img_path.stem}_overlay.png"
            cv2.imwrite(str(over_out), overlay)

            messagebox.showinfo("Saved", f"Saved:\n{mask_out}\n{over_out}")

        except Exception as e:
            messagebox.showerror("Save error", str(e))


if __name__ == "__main__":
    app = GUIPredictor()
    app.mainloop()
