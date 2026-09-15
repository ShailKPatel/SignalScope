# SIGNAL SCOPE: Comprehensive Project Requirements & Implementation Specification

> **SIH - 2026 [Internal Hackathon] | Problem Statement 2 (C-433)**
> **Topic:** Telling Real From Synthetic in the Age of Generative Media
> **Target:** 100/100 Points Across Core + ALL Bonus Modules (Modules A – G)

---

## 1. Executive Summary & Critical Constraints

### 1.1 Core Mission
Build an end-to-end, media authenticity verification system (**SignalScope**) that:
1. **Classifies** an input image as **Real** or **AI-Generated** (synthetic).
2. **Generalizes** effectively to **unseen generator architectures** (e.g., Midjourney, SDXL, Flux, DALL-E 3, Imagen, etc.).
3. **Explains** its verdict with faithful, localized visual cues (heatmaps) and natural language explanations.
4. **Extends** into a comprehensive forensic suite delivering **all 7 optional bonus modules**.

### 1.2 Mandatory Ethics & Scope Boundaries (Strict Disqualification Rules)
> [!CAUTION]
> **DISQUALIFICATION WARNING:** Submissions that violate ethics rules will receive 0 points / disqualification.
* **In-Scope:** Detection of synthetic imagery in general (scenes, objects, art, architecture, product shots).
* **Out-of-Scope (FORBIDDEN):**
  * Face-swap deepfakes of real, identifiable individual people.
  * Political claims or real-world news event adjudication.
  * Scraping images of identifiable individuals.
* **Verdict Framing Rule:** Always frame outputs as **likelihood assessments** (e.g., *"likely AI-generated (88% confidence)"*), **NEVER** absolute accusations.

---

## 2. Detailed Technical Breakdown: Core + ALL Bonus Modules

### 2.1 Mandatory Core Task (Required)
* **Input:** Single image (JPEG/PNG/WebP, arbitrary resolution).
* **Output:**
  * Binary Label: `Real` vs `AI-generated` (with probability score $[0.0, 1.0]$).
  * Operating Point Metrics: Accuracy and False-Positive Rate (FPR) at a calibrated threshold.
* **Evaluation Metrics:**
  * **Primary Metric:** **ROC-AUC on Held-Out Test Set** (Weighted heavily on the **Unseen-Generator Split**).
  * **Secondary Metrics:** Overall ROC-AUC, Macro-F1 score, Confusion Matrix.
* **Minimum Bar:**
  * Transfer learning backbone (CNN or ViT).
  * Honest train/val/test split (no data leakage).
  * Predict interface accessible via Web App / Notebook / CLI.

---

### 2.2 Bonus Module A: Faithful Explanation (Headline Bonus)
* **Objective:** Explain *why* an image is flagged as synthetic using human-interpretable cues.
* **Key Components:**
  1. **Visual Heatmap / Saliency Map:** Grad-CAM / Layer-CAM / Attention map highlighting the precise anomalous regions (e.g., warped text, anatomical flaws, irregular lighting/reflections).
  2. **Grounded Textual Cues:** Specific natural language points covering:
     * Implausible textures / frequency artifacts.
     * Warped / unreadable text.
     * Lighting and shadow inconsistencies.
     * Anatomical or geometrical errors.
* **Scoring Rubric (15 Points):**
  * **Correctness:** Cited cues correspond to real artifacts.
  * **Localisation:** Heatmap points to genuinely anomalous areas, not whole image.
  * **Usefulness:** Understandable to non-experts; honest uncertainty hedging.
  * **No Over-claiming:** Avoids fabricated certainty.

---

### 2.3 Bonus Module B: Generator Attribution
* **Objective:** Go beyond binary classification to identify the likely generator family or model type.
* **Key Components:**
  * **Multi-class Classifier:** Identifies architecture families:
    * Diffusion-based (Stable Diffusion, Midjourney, DALL-E, Flux)
    * GAN-based (StyleGAN, ProGAN, BigGAN)
    * Autoregressive / Transformer-based (Parti, Imagen)
  * **Metrics:** Multi-class confusion matrix, top-1 accuracy, and macro-F1 per generator family.

---

### 2.4 Bonus Module C: Robustness to Degradation
* **Objective:** Ensure detector accuracy remains resilient under real-world image degradations.
* **Supported Degradation Vectors:**
  1. **JPEG Compression:** Quality factors ($Q \in [30, 50, 70, 90]$).
  2. **Resizing / Downsampling:** Scaling down to $256 \times 256$, $512 \times 512$.
  3. **Screenshotting / Re-encoding:** Artifacts introduced by social media re-uploads.
  4. **Light Editing / Blur / Gaussian Noise.**
* **Deliverable:** Degradation-vs-Accuracy curve plots and automated robustness test suite.

---

### 2.5 Bonus Module D: Provenance & Metadata Analysis
* **Objective:** Combine metadata signals with visual model verdicts for holistic authenticity checks.
* **Key Components:**
  * **EXIF Metadata Parser:** Extracts camera specs, software history, edit timestamps.
  * **C2PA / Content Credentials Parser:** Reads signed cryptographic manifests (C2PA / JUMBF headers).
  * **Fusion Engine:** Combines metadata evidence (e.g., C2PA signature status, missing EXIF camera info) with ML confidence score.

---

### 2.6 Bonus Module E: Multimodal (Image + Text) Consistency
* **Objective:** Assess semantic consistency between an image and its accompanying caption/claim.
* **Key Components:**
  * **Cross-Modal Embedding Alignment:** Use CLIP / BLIP-2 / ViT-L text-image matching score.
  * **Mismatch Signal:** Detect when synthetic images are paired with misleading generic descriptive captions.

---

### 2.7 Bonus Module F: Real-Time / Deployable Application
* **Objective:** Deliver a production-ready, highly responsive user interface with low latency.
* **Key Features:**
  * **Web Dashboard:** Sleek, modern drag-and-drop web UI (Vite + React / Web UI) with instant visual feedback.
  * **Batch Scanning:** Upload & evaluate multiple images simultaneously.
  * **Browser Extension Mockup:** Chrome extension concept for single-click image verification on web pages.
  * **Latency Target:** $< 3$ seconds per prediction on standard GPU/CPU.

---

### 2.8 Bonus Module G: Active Defence & Failure Analysis
* **Objective:** Test detector vulnerability against adversarial perturbations and document failure cases honestly.
* **Key Components:**
  * **Adversarial Attack Testing:** Evaluate against FGSM, PGD, and frequency domain noise injection.
  * **Mitigation Strategies:** Test blur defense, JPEG re-compression defense, and adversarial training.
  * **Failure Analysis Report:** Honest documentation of edge cases where the detector fails.

---

## 3. Evaluation Criteria & Scoring Matrix (100 Points Total)

| Criteria | Weight | Focus Areas & Scoring Anchors |
| :--- | :---: | :--- |
| **AI/ML Implementation** | **25 pts** | Held-out AUC (unseen-generator weighted heaviest), honest split, calibration, correct metric reporting. |
| **Technical Implementation** | **20 pts** | Reproducibility (runs in <10m from README), code quality, robustness engineering, deployment. |
| **Innovation & Creativity** | **15 pts** | Novel feature extraction (frequency domain, spatial noise, multi-model ensemble). |
| **Explanation & Trust Impact** | **15 pts** | Module A score (faithfulness, Grad-CAM localisation, useful natural language explanations). |
| **User Experience** | **10 pts** | Sleek UI, responsible "likely" framing, drag-and-drop, batch scanning. |
| **Problem Understanding** | **10 pts** | Grasp of unseen generalisation, honest limitation reporting. |
| **Presentation & Demo** | **5 pts** | 3-5 minute clear demo video. |
| **TOTAL** | **100 pts** | |

### Tie-Break Hierarchy
1. **Unseen-generator-split AUC** on held-out test set (Higher wins).
2. **Overall held-out AUC**.
3. **Reproducibility** (Clean single-command run from README).
4. **Explanation faithfulness score** & depth of bonus modules.

---

## 4. Required Repository Structure (Submission Contract)

```
SIGNALSCOPE/
├── .gitignore                      # Exhaustive gitignore for ML, web, & temp files
├── README.md                       # Main entry point with reproduction guide
├── REQUIREMENTS_AND_SPECIFICATIONS.md # Technical specification & rubric breakdown
├── requirements.txt                # Python dependencies
├── src/                            # Core application source code
│   ├── app/                        # Web dashboard / frontend code
│   └── api/                        # Backend REST / FastAPI services
├── model/                          # Model definition & prediction interface
│   ├── predict.py                  # Standardized inference entrypoint
│   ├── backbone.py                 # Neural network architecture definition
│   ├── train.py                    # Training & validation scripts
│   ├── explainability.py           # Grad-CAM & explanation generator (Module A)
│   ├── attribution.py              # Generator classification (Module B)
│   ├── robustness.py               # Degradation testing (Module C)
│   ├── metadata.py                 # EXIF / C2PA parser (Module D)
│   ├── multimodal.py               # CLIP text-image consistency (Module E)
│   └── active_defense.py           # Adversarial attack testing (Module G)
└── report/                         # Deliverables directory
    ├── model_report.md             # Standardized 1-page model report
    └── explanation_samples/        # Sample heatmaps and outputs for judging
```

---

## 5. Next Execution Steps
1. **Data Pipeline:** Set up CIFAKE (`birdy654/cifake-real-and-ai-generated-synthetic-images`) ingestion: train on its `train/` split (10% held back for validation) and report on its `test/` split.
2. **Model Training Pipeline:** Implement EfficientNet / ViT baseline + frequency domain feature extraction (FFT / DCT artifacts).
3. **Bonus Modules Engine:** Implement Modules A through G in `model/`.
4. **Web UI & API:** Build a high-performance web dashboard with drag-and-drop, batch processing, and visual heatmaps.
5. **Evaluation Suite:** Build automated evaluation scripts for held-out metrics, ROC-AUC, confusion matrix, and degradation curves.
