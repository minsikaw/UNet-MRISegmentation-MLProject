# Brain MRI Segmentation U-Net Project

This repository provides a clean, modular Python codebase for training and evaluating a U‑Net model on brain MRI segmentation tasks. It is designed to replace a Jupyter notebook prototype with a structured, maintainable project that can be versioned and extended easily.

## Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Dataset Structure

Your dataset should be organized with separate folders for training and validation images and masks. Update the paths in `configs/baseline.yaml` to match your dataset location. A typical structure might look like:

```
data/
  train/
    images/
      img1.png
      img2.png
      ...
    masks/
      img1_mask.png
      img2_mask.png
      ...
  val/
    images/
      ...
    masks/
      ...
```

## Running Training

Use the following command to start training the U‑Net model:

```bash
python scripts/train.py --config configs/baseline.yaml
```

This reads hyperparameters and paths from the YAML config file and saves the best model checkpoint into the directory specified under `train.checkpoint_dir`.

## Evaluating a Model

To evaluate a saved checkpoint on the validation set:

```bash
python scripts/evaluate.py --config configs/baseline.yaml --checkpoint runs/best_model.pth
```

## Inference on New Images

Run inference on a single image and save the predicted mask:

```bash
python scripts/predict.py --config configs/baseline.yaml --checkpoint runs/best_model.pth --image_path path/to/your/image.png --output_path path/to/save_mask.png
```

## File Descriptions

- **configs/baseline.yaml**: Defines dataset paths, training hyperparameters, and model settings.
- **src/data/dataset.py**: Dataset class loading images and masks, applying transforms.
- **src/models/unet.py**: Implementation of a U‑Net architecture.
- **src/losses.py**: Loss functions (Dice loss and BCE+Dice).
- **src/metrics.py**: Dice coefficient metric.
- **src/engine.py**: Training and evaluation loops.
- **src/utils/misc.py**: Utility functions (e.g., reproducible seeding).
- **scripts/train.py**: CLI entrypoint for training.
- **scripts/evaluate.py**: CLI entrypoint for validation.
- **scripts/predict.py**: CLI entrypoint for inference on new images.