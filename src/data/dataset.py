import os
from torch.utils.data import Dataset
import cv2
import numpy as np
import torch

class BrainMRIDataset(Dataset):
    """
    Minimal dataset without Albumentations:
    - loads grayscale image & mask
    - resizes with OpenCV
    - normalizes to [0,1]
    - returns tensors shaped [1, H, W]
    """
    def __init__(self, images_dir, masks_dir, image_size=256):
        self.images_dir = images_dir
        self.masks_dir = masks_dir

        self.image_paths = sorted([
            os.path.join(images_dir, f)
            for f in os.listdir(images_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))
        ])
        self.mask_paths = sorted([
            os.path.join(masks_dir, f)
            for f in os.listdir(masks_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))
        ])
        assert len(self.image_paths) == len(self.mask_paths), \
            "Mismatch between number of images and masks"
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def _load_gray(self, path):
        x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if x is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        x = cv2.resize(x, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        x = x.astype(np.float32) / 255.0
        return x

    def __getitem__(self, idx):
        img = self._load_gray(self.image_paths[idx])
        msk = self._load_gray(self.mask_paths[idx])

        img = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]
        msk = torch.from_numpy(msk).unsqueeze(0)  # [1, H, W]

        msk = (msk > 0.5).float()
        return img, msk
