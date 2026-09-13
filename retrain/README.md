# SignalScope Retraining & Fine-Tuning Environment

A comprehensive, scalable retraining workspace designed to train, validate, and evaluate SignalScope deep learning models on datasets of **100,000 images** (Real vs. AI-Generated).

---

## 📁 Directory Structure

```
retrain/
├── config.yaml            # Hyperparameters, split ratios, backbone & path configuration
├── backbone.py            # Dual-Stream PyTorch Neural Network (Spatial CNN/ViT + 2D FFT/DCT Spectrum)
├── dataset.py             # PyTorch Dataset, DataLoader, and memory-efficient data loading
├── dataset_generator.py   # Utility to build folder structures, sample datasets, & 100k image manifests
├── train.py               # Master training script with automated train/val/test split & AUC tracking
├── evaluate.py            # Held-out test set evaluation engine (ROC-AUC, Macro-F1, FPR, Confusion Matrix)
├── checkpoints/           # Output directory storing trained model weights (best_model.pt)
└── data/                  # Dataset directory structure (80% Train / 10% Val / 10% Test)
    ├── manifest.json      # 100,000 image dataset index manifest
    ├── train/
    │   ├── real/
    │   └── synthetic/
    ├── val/
    │   ├── real/
    │   └── synthetic/
    └── test/
        ├── real/
        └── synthetic/
```

---

## ⚡ Quick Start: 3-Step Execution

### Step 1: Initialize Data Structure & 100k Image Manifest
Generate the standard dataset folder layout, physical sample images, and 100,000-image metadata index manifest:
```bash
python retrain/dataset_generator.py --num-samples 100 --target-total 100000
```

### Step 2: Run Retraining Pipeline
Launch training with automated 80/10/10 train/val/test splitting, ROC-AUC tracking, and checkpoint saving:
```bash
python retrain/train.py --epochs 5 --batch-size 16 --lr 0.0003 --spatial-backbone resnet34
```
> The best checkpoint will automatically be saved to `retrain/checkpoints/best_model.pt`.

### Step 3: Evaluate Held-Out Test Set
Evaluate the trained model on held-out test split and unseen generator families:
```bash
python retrain/evaluate.py --model-path retrain/checkpoints/best_model.pt
```

---

## 📊 Dataset & Split Ratio Specification

* **Total Dataset Target:** 100,000 images (50,000 Real, 50,000 AI-Generated).
* **Train / Val / Test Ratio:**
  * **Train Set:** 80% (80,000 images)
  * **Validation Set:** 10% (10,000 images)
  * **Held-Out Test Set:** 10% (10,000 images)
* **Generator Stratification:** The held-out test split isolates unseen generator families (e.g. Midjourney v6, Flux.1) to prevent data leakage and evaluate true cross-architecture generalization.

---

## 🧠 Neural Network Architecture (`backbone.py`)

The **SignalScope Dual-Stream Architecture** processes input images through two parallel representation channels:
1. **Spatial Branch:** Deep Convolutional Backbone (ResNet / EfficientNet) capturing semantic features, high-level textures, and object geometry.
2. **Frequency Domain Branch:** 2D Fast Fourier Transform (FFT) log-magnitude spectrum extractor identifying spatial upsampling grid artifacts produced by generative diffusion/GAN decoders.
3. **Fusion Classifier:** Fuses spatial and frequency embeddings into a single calibrated probability score ($[0.0, 1.0]$).

---

## ⚙️ Command Line Options (`train.py`)

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--data-dir` | `retrain/data` | Path to dataset directory |
| `--manifest-path` | `retrain/data/manifest.json` | Path to 100k dataset index manifest |
| `--checkpoint-dir` | `retrain/checkpoints` | Directory to export model weights |
| `--epochs` | `5` | Total training epochs |
| `--batch-size` | `16` | Mini-batch size |
| `--lr` | `0.0003` | Learning rate for AdamW optimizer |
| `--spatial-backbone` | `resnet34` | Spatial feature backbone (`resnet18`, `resnet34`, `resnet50`, `efficientnet_b0`) |
