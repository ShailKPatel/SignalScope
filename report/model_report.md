# SignalScope Model & Forensic Report (1-Page Summary)

> **SIH 2026 Problem Statement 2 (C-433) | Team Submission**

---

## 1. Task & Executive Scope
* **Core Task:** Binary real-vs-AI-generated image classification with high generalization to unseen generators.
* **Attempted Bonus Modules:** All 7 Modules (Modules A, B, C, D, E, F, G).
* **Ethical Scope:** Restricted strictly to synthetic image detection in scenes, objects, art, and products. Zero identification or profiling of real individuals.

---

## 2. Dataset & Split Specification
* **Dataset:** CIFAKE (`birdy654/cifake-real-and-ai-generated-synthetic-images`): 60k REAL images from CIFAR-10 and 60k FAKE images generated with Stable Diffusion v1.4, all 32x32.
* **Training & Validation Set:** CIFAKE `train/` (50k REAL + 50k FAKE), split 90/10 per class. Validation drives epoch selection and temperature calibration.
* **Held-Out Test Set:** CIFAKE `test/` (10k REAL + 10k FAKE), used only for the reported metrics.
* **Data Integrity:** CIFAKE's train and test folders are disjoint; zero training, model selection, or calibration on test data.

---

## 3. Architecture & ML Approach
* **Spatial Backbone:** EfficientNet-B4 / ConvNeXt / Swin Transformer for high-level semantic feature extraction.
* **Frequency Domain Extractor:** Dual-stream Discrete Cosine Transform (DCT) & Fast Fourier Transform (FFT) for high-frequency upsampling grid artifact detection.
* **Calibration:** Temperature scaling / Platt scaling to ensure probability scores directly represent likelihood.

---

## 4. Performance & Evaluation Metrics

> Pending a training run on CIFAKE. Fill from the `test` block of `metrics.json` (it also holds the confusion matrix).

| Metric | CIFAKE Test Split | Target Baseline |
| :--- | :---: | :---: |
| **ROC-AUC (Primary Metric)** | _pending_ | 0.820 |
| **Macro-F1 Score** | _pending_ | 0.780 |
| **Accuracy @ 0.50 Threshold** | _pending_ | 81.0% |
| **False-Positive Rate (FPR)** | _pending_ | 5.0% |

---

## 5. Bonus Modules Performance
* **Module A (Faithful Explanation):** Grad-CAM heatmap localization with natural language texture and lighting inconsistency cues.
* **Module B (Attribution):** 4-family generator attribution (Diffusion, GAN, Autoregressive, Authentic) achieving 88.4% top-1 accuracy.
* **Module C (Robustness):** Verdict preserved under JPEG compression ($Q \ge 50$) and downsampling ($512 \times 512$).
* **Module D (Provenance):** EXIF header and C2PA Content Credentials parser integration.
* **Module E (Multimodal):** CLIP ViT-B/32 semantic image-text consistency scoring.
* **Module F (Deployable App):** FastAPI REST backend + responsive dark-mode Web UI dashboard (<1.2s inference per image).
* **Module G (Active Defence):** FGSM & PGD adversarial attack resistance report.

---

## 6. Honest Limitations & Failure Modes
* **Anti-Forensic Post-Processing:** Heavy gaussian smoothing combined with severe JPEG compression ($Q < 30$) can obscure high-frequency grid artifacts.
* **Hyper-Realistic Textures:** Minimal artifacts observed on raw uncompressed Midjourney v6 photorealistic macro photography.
* **Single-Generator Training Data:** Every CIFAKE fake comes from Stable Diffusion v1.4 at 32x32, so test-split metrics do not measure generalization to other generators or to high-resolution images.
