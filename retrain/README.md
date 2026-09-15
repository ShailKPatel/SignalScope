# SignalScope Retraining & Fine-Tuning Environment

A retraining workspace to train, validate, and evaluate SignalScope detectors on **[CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)**: 120,000 images, Real vs. AI-Generated.

---

## 📁 Directory Structure

```
retrain/
├── config.yaml                               # Hyperparameters, CIFAKE split settings, backbone & paths
├── backbone.py                               # Dual-Stream PyTorch Neural Network (Spatial CNN + 2D FFT Spectrum)
├── dataset.py                                # Loads CIFAKE train/test folders, carves validation out of train
├── dataset_generator.py                      # Seeds a CIFAKE-layout folder with placeholder images for dry runs
├── train.py                                  # Local training script with ROC-AUC tracking & checkpointing
├── evaluate.py                               # CIFAKE test-split evaluation (ROC-AUC, Macro-F1, FPR, Confusion Matrix)
├── eval_majority_vote.py                     # 5-member majority-vote evaluation on the CIFAKE test split
├── kaggle_train_cifake.py / .ipynb           # Kaggle GPU training run (produces best_model.pt + metrics.json)
├── kaggle_majority_baseline.py / .ipynb      # Frozen-member majority vote on the CIFAKE test split
├── kaggle_ensemble_vs_majority.py / .ipynb   # Learned combiner vs majority vote on the CIFAKE test split
├── kaggle_train_ensemble.ipynb               # Logistic-regression combiner fitted on CIFAKE train
├── checkpoints/                              # Trained model weights (best_model.pt)
└── data/                                     # CIFAKE, as downloaded from Kaggle
    ├── train/
    │   ├── REAL/                             # 50,000 CIFAR-10 images
    │   └── FAKE/                             # 50,000 Stable Diffusion v1.4 images
    └── test/
        ├── REAL/                             # 10,000 CIFAR-10 images
        └── FAKE/                             # 10,000 Stable Diffusion v1.4 images
```

---

## ⚡ Quick Start: 3-Step Execution

### Step 1: Download CIFAKE
```bash
kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images -p retrain/data --unzip
```
Check that you end up with `retrain/data/train/REAL`, `retrain/data/train/FAKE`, `retrain/data/test/REAL` and `retrain/data/test/FAKE`.

For a dry run without the download, seed the same layout with placeholder images:
```bash
python retrain/dataset_generator.py --num-samples 120
```

### Step 2: Run Retraining Pipeline
Trains on CIFAKE train (with 10% held back for validation), tracks ROC-AUC, and saves the best checkpoint:
```bash
python retrain/train.py --epochs 5 --batch-size 16 --lr 0.0003 --spatial-backbone resnet34
```
> The best checkpoint will automatically be saved to `retrain/checkpoints/best_model.pt`.

### Step 3: Evaluate on the CIFAKE Test Split
```bash
python retrain/evaluate.py --model-path retrain/checkpoints/best_model.pt
```

### Kaggle GPU Run
Attach `birdy654/cifake-real-and-ai-generated-synthetic-images` via **Add Data**, enable a GPU, and run `kaggle_train_cifake.ipynb` top to bottom. It writes `best_model.pt` and `metrics.json` to `/kaggle/working/`.

---

## 📊 Dataset & Split Specification

* **Dataset:** CIFAKE. REAL images come from CIFAR-10; FAKE images were generated with Stable Diffusion v1.4.
* **Labels:** `REAL` = 0, `FAKE` = 1 (AI-generated).
* **Train Split (`train/`):** 100,000 images (50,000 REAL, 50,000 FAKE)
  * **Training Set:** 90% per class (90,000 images)
  * **Validation Set:** 10% per class (10,000 images), used for epoch selection and temperature calibration
* **Held-Out Test Set (`test/`):** 20,000 images (10,000 REAL, 10,000 FAKE). Used only for reported metrics.
* **Resolution:** Every image is 32x32. Models train at that native size, and checkpoints record `native_size: 32` so inference resizes inputs to match.
* **Limitation:** CIFAKE has a single generator, so there is no unseen-generator split. Test-split metrics do not measure generalization to other generators.

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
| `--data-dir` | `retrain/data` | CIFAKE root folder (contains `train/` and `test/`) |
| `--manifest-path` | none | Optional manifest JSON; when omitted, the CIFAKE folders are scanned |
| `--checkpoint-dir` | `retrain/checkpoints` | Directory to export model weights |
| `--epochs` | `5` | Total training epochs |
| `--batch-size` | `16` | Mini-batch size |
| `--lr` | `0.0003` | Learning rate for AdamW optimizer |
| `--spatial-backbone` | `resnet34` | Spatial feature backbone (`resnet18`, `resnet34`, `resnet50`, `efficientnet_b0`) |
| `--max-samples` | none | Random subset size for quick dry runs |
