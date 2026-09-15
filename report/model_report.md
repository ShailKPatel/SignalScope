# SignalScope Model Report (1-Page Summary)

> **SIH 2026 Problem Statement 2 (C-433) | Team Submission**

## 1. Task & Scope
* **Core:** binary real-vs-AI image classification with a calibrated score.
* **Modules built:** A (explanation, limited), C (robustness, measured), D (metadata/provenance), F (web app + API). **Not built:** B, E, G. The app reports them as not built.
* **Ethics:** general synthetic-image detection only. No identification of real individuals; no person images in any data.

## 2. Data & Split
* **CIFAKE** (Kaggle `birdy654/...`): REAL = CIFAR-10, FAKE = Stable Diffusion v1.4, 32×32.
* **Train:** 90% of CIFAKE `train/` (90,000). **Validation:** remaining 10% (10,000). Used for epoch selection, temperature scaling, and the stacker fit (balanced 4,000 sample, 5-fold out-of-fold).
* **Test:** CIFAKE `test/` (20,000), used only for the numbers below. No test data touches training, selection, or calibration.

## 3. Approach
* **Level 1:** explicit generator signatures in EXIF/XMP/PNG text, plus C2PA manifest reading.
* **Level 2 members:** (1) ViT-Base `dima806/deepfake_vs_real_image_detection`, (2) Swin `Organika/sdxl-detector`, both frozen; (3) our dual-stream model, a ResNet34 spatial branch plus a 2D-FFT magnitude branch, trained on CIFAKE at native 32×32 so resizing does not erase frequency artifacts.
* **Fusion:** ridge logistic regression on member logits (C = 31.6). Learned weights on standardized logits: ViT −0.29, Swin +0.17, dual-stream +7.53.
* **Calibration:** dual-stream temperature scaling (T = 1.066) on validation. Threshold 0.5, plus a recorded ≈5%-validation-FPR threshold.

## 4. Results (CIFAKE test, n = 20,000)

| Model | AUC | Macro-F1 | Acc | FPR | TN / FP / FN / TP |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Stacked @ 0.5 (deployed)** | **0.9976** | **0.9780** | **97.80%** | **2.35%** | 9765 / 235 / 205 / 9795 |
| Stacked @ 5%-val-FPR thr. | 0.9976 | 0.9674 | 96.74% | 5.71% | 9429 / 571 / 81 / 9919 |
| Dual-stream alone | 0.9976 | 0.9774 | 97.74% | 2.63% | 9737 / 263 / 189 / 9811 |
| Majority vote (baseline) | 0.9101 | 0.5906 | 63.82% | 2.06% | 9794 / 206 / 7029 / 2971 |
| Swin alone | 0.6454 | 0.5330 | 57.43% | 12.84% | 8716 / 1284 / 7231 / 2769 |
| ViT alone | 0.4142 | 0.3582 | 48.33% | 7.54% | 9246 / 754 / 9581 / 419 |

**Takeaways.** Stacking beats majority voting by 34 accuracy points. It does this by learning to ignore the two off-the-shelf detectors, which fail on 32×32 CIFAKE. Its gain over the dual-stream model alone is small (+0.06 pp accuracy, −0.28 pp FPR). **Unseen-generator AUC** cannot be measured on CIFAKE (one generator); organisers compute it.

## 5. Modules
* **A:** overlay = ViT member's last-layer attention. Cues = each member's P(AI), its additive share of the stacked log-odds, and high-frequency spectral energy (reported for inspection, not used in the decision). No cue is templated text. Samples: `report/explanation_samples/`. **Faithfulness caveat:** the decision is driven by the dual-stream member, which has no map, so the overlay is context rather than evidence.
* **C:** the image is actually re-encoded (JPEG Q90/70/50/30) and downscaled (75/50/25%), and every variant is re-scored. Example: a CIFAKE fake held its verdict at every JPEG level, but dropped to 0.18 when downscaled to 24×24.
* **D:** signature and C2PA parsing. A found signature short-circuits to "likely AI-generated (metadata)". A missing signature is never read as authenticity.
* **F:** FastAPI REST + batch + WebSocket, web dashboard. Returns HTTP 503 rather than a made-up score when no detector loads.

## 6. Limitations & Failure Modes
* Single generator (SD v1.4) at 32×32, so no evidence of generalisation to other generators or to high-resolution or compressed real-world images. A prior prototype on different data fell from 0.93 to 0.70 AUC on held-out generators.
* Sensitive to small rescaling of 32×32 inputs (Module C example above).
* Module A overlay is not faithful to the deciding model (see §5).
* Metadata signatures are trivially stripped or forged.
* CPU latency ≈10 s per image, dominated by Module C re-scoring.
