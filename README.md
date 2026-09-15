# SignalScope: Telling Real From Synthetic in the Age of Generative Media

> **SIH 2026 [Internal Hackathon] | Problem Statement 2 (C-433)**
> **Domain:** AI / Media Forensics / Trust & Safety

SignalScope classifies an image as *likely real* or *likely AI-generated* with a calibrated score. It pairs that verdict with a metadata/provenance check, a saliency overlay with measured cues, and a measured degradation test, all served through a web dashboard and REST/WebSocket API.

▶️ **[Watch the demo video](https://drive.google.com/file/d/1rLY3w_Dz0JCrzayGLjvlkKtZDi9RmhhZ/view?usp=sharing)**

---

## 1. Built Modules

| Module | Status | What it actually does |
| :--- | :---: | :--- |
| **Core** real-vs-AI classifier | ✅ Built | 3-member stacked ensemble; score, verdict, operating point from the CIFAKE test split |
| **A** Explanation | ✅ Built | SmoothGrad saliency overlay from the dual-stream member (the one that drives the decision), a per-image deletion check (do salient pixels move the score more than random ones?), and cues from measured signals (member P(AI), each member's share of the stacked log-odds, spectral energy) |
| **B** Generator attribution | ❌ Not built | Reports a generator only when file metadata declares one (via Module D) |
| **C** Robustness to degradation | ✅ Built | Re-encodes the uploaded image at JPEG Q90/70/50/30 and downscales to 75/50/25%, re-scores every variant, reports whether the verdict holds |
| **D** Metadata & provenance | ✅ Built | EXIF / XMP / PNG text-chunk generator signatures, C2PA manifest read via `c2pa-python` |
| **E** Multimodal consistency | ❌ Not built | Caption is recorded; no score is computed |
| **F** Deployable app | ✅ Built | FastAPI REST (`/api/predict`, `/api/predict-batch`), WebSocket (`/ws/analyze`), web dashboard |
| **G** Active defence | ❌ Not built | No adversarial evaluation is run |

---

## 2. Setup & Run (about 10 minutes, CPU is enough)

Needs Python 3.10+ and internet on the first run: about 750 MB of model weights download once, automatically.

### Step 1: Clone and create the virtual environment

Linux / macOS:
```bash
git clone https://github.com/ShailKPatel/SignalScope.git
cd SignalScope
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Recommended without an NVIDIA GPU: CPU-only torch is a far smaller download
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Windows (PowerShell):
```powershell
git clone https://github.com/ShailKPatel/SignalScope.git
cd SignalScope
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Step 2: Start the web app
```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
On the first start the server downloads all model weights before it accepts requests. The terminal shows progress and then `SignalScope: models ready (stacked ensemble)`. What it downloads:
* Dual-stream checkpoint `best_model.pt` (87 MB) from the [v1.0 GitHub Release](https://github.com/ShailKPatel/SignalScope/releases/tag/v1.0), saved to `retrain/checkpoints/` and checksum-verified.
* ViT and Swin members (about 330 MB each) from Hugging Face, cached in `~/.cache/huggingface`.

Then open:
* Dashboard: `http://localhost:8000`
* API docs (Swagger): `http://localhost:8000/docs`

If the automatic download is blocked (offline machine, proxy), download `best_model.pt` from the release page and place it at `retrain/checkpoints/best_model.pt`. Without it the app still runs, but in majority-vote fallback mode, where the reported metrics do not apply.

### Step 3 (optional): Prediction from the command line
```bash
python -c "
from model.predict import predict_image
for p in ['test_images/cifake_real_0.png', 'test_images/cifake_fake_0.png']:
    v = predict_image(p)['verdict']
    print(p, '->', v['label'], v['confidence_score'], '|', v['ensemble_breakdown']['ensemble_strategy'])
"
```
Expected: `cifake_real_0.png -> likely real`, `cifake_fake_0.png -> likely AI-generated`, strategy `Stacked Generalization (ridge logistic meta-learner)`.

### Step 4 (optional): Smoke tests
```bash
python test_ensemble_system.py   # ensemble, stacker math, REST + WebSocket
python test_live_system.py       # Level 1 metadata path + Level 2 model path
```

---

## 3. Datasets & Licences

| Dataset | Use | Source / licence |
| :--- | :--- | :--- |
| **CIFAKE** (Bird & Lotfi, 2024) | Train, validation, test | [Kaggle `birdy654/cifake-real-and-ai-generated-synthetic-images`](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images). Licence "Other" per the dataset page: REAL = CIFAR-10 (Krizhevsky, 2009), FAKE = Stable Diffusion v1.4 generations |
| 4 CIFAKE test images in `test_images/cifake_*.png` | Smoke tests only | Hugging Face mirror [`dragonintelligence/CIFAKE-image-dataset`](https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset), test split |
| `test_images/*_sample.png` (Gemini, Grok, ComfyUI, DALL-E) | Level 1 metadata tests only | Flat-colour images generated locally with injected metadata (`test_images/create_metadata_test_samples.py`); not real generator outputs |

| Split | REAL | FAKE | Use |
| :--- | :---: | :---: | :--- |
| CIFAKE `train/` 90% | 45,000 | 45,000 | Dual-stream training |
| CIFAKE `train/` 10% (validation) | 5,000 | 5,000 | Epoch selection, temperature scaling, stacker fit (balanced 4,000-image sample, 5-fold CV) |
| CIFAKE `test/` | 10,000 | 10,000 | Reported metrics only |

No person images are used anywhere (CIFAR-10 has no person class).

---

## 4. Reported Metrics (CIFAKE test split, 20,000 images)

| Model | ROC-AUC | Macro-F1 | Accuracy | FPR | Confusion (TN / FP / FN / TP) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Stacked ensemble @ 0.5 (deployed)** | **0.9976** | **0.9780** | **97.80%** | **2.35%** | 9765 / 235 / 205 / 9795 |
| Stacked ensemble @ 5%-val-FPR threshold (0.165) | 0.9976 | 0.9674 | 96.74% | 5.71% | 9429 / 571 / 81 / 9919 |
| Dual-stream alone (ResNet34 + FFT) | 0.9976 | 0.9774 | 97.74% | 2.63% | 9737 / 263 / 189 / 9811 |
| Majority vote of 3 members (baseline) | 0.9101 | 0.5906 | 63.82% | 2.06% | 9794 / 206 / 7029 / 2971 |
| Swin `Organika/sdxl-detector` alone | 0.6454 | 0.5330 | 57.43% | 12.84% | 8716 / 1284 / 7231 / 2769 |
| ViT `dima806/deepfake_vs_real_image_detection` alone | 0.4142 | 0.3582 | 48.33% | 7.54% | 9246 / 754 / 9581 / 419 |

**Unseen-generator-split AUC:** not measurable by us. Every CIFAKE fake comes from Stable Diffusion v1.4, so no held-out generator exists in the provided data. The organisers' unseen-generator evaluation is the only measure of this.

Source files: `retrain/checkpoints/metrics.json` (dual-stream) and `retrain/checkpoints/stacking_metalearner.json` (`test_metrics`, `test_metrics_at_5pct_fpr`, `test_baselines`). Produced by `retrain/kaggle_one_shot.py`.

---

## 5. Architecture, Calibration & Robustness

```
Image (+ optional caption)
  │
  ├─ Level 1: Metadata & provenance (model/metadata.py)
  │     explicit generator signature found? → verdict "likely AI-generated (metadata)"
  │
  └─ Level 2: Stacked ensemble (model/ensemble.py)
        ├─ ViT-Base   dima806/deepfake_vs_real_image_detection   → P(AI)
        ├─ Swin       Organika/sdxl-detector                     → P(AI)
        ├─ Dual-stream: ResNet34 spatial + 2D FFT magnitude branch, trained on CIFAKE at 32×32 → P(AI)
        └─ Meta-learner: ridge logistic regression on logit(P(AI)) of the 3 members
              (weights: ViT −0.29, Swin +0.17, dual-stream +7.53 on standardized logits)
  │
  └─ Modules: A overlay + measured cues · C degradation re-scoring · D metadata report
```

* **Calibration:** dual-stream temperature scaling (T = 1.066) fit on validation. The meta-learner is itself a logistic model fit on validation-only out-of-fold predictions, and its decision threshold is 0.5. A low-FPR threshold (≈5% FPR on validation) is also recorded.
* **Robustness (Module C):** per image, measured rather than simulated. On the bundled samples the verdict held across all JPEG levels. Downscaling a 32×32 fake to 24×24 dropped its score to 0.18, a real failure the curve exposes.
* **Code map:** `model/predict.py` (entry point `predict_image`), `model/ensemble.py`, `model/explainability.py`, `model/robustness.py`, `model/metadata.py`, `retrain/` (training), `src/api/main.py`, `src/app/`.

---

## 6. Known Limitations

* **The ensemble adds little over the dual-stream model.** On CIFAKE the ViT scores below chance (AUC 0.41) and the Swin is weak (0.65). Both were fine-tuned on high-resolution data and see CIFAKE's 32×32 images upscaled. The meta-learner correctly assigns them near-zero weight.
* **Explanation faithfulness is partial.** The overlay is SmoothGrad on the dual-stream member at its 32×32 input, so it is coarse. The deletion check shows salient pixels moving the score more than random pixels on 3 of the 4 bundled samples, not all; each result reports its own check. Cues list measured numbers and do not claim to localize artifacts. Samples: `report/explanation_samples/`.
* **Single generator, low resolution.** Training data is SD v1.4 at 32×32, so performance on other generators, high-resolution photos, or real-world JPEGs is unmeasured. An earlier prototype on a different dataset (`trained-v1/metrics.json`) fell from 0.93 validation AUC to 0.70 on held-out generators. Expect a similar drop.
* **Level 1 trusts self-declared metadata.** It is trivially stripped or forged. Absence of a signature is never treated as evidence of authenticity.
* **CPU latency:** about 10 s per image, because Module C re-scores 7 variants.

---

## 7. Ethics & Responsible Use
* In scope: synthetic imagery in general (scenes, objects, art). No features identify, profile, or adjudicate claims about real individuals, and no test data contains people.
* Every output is a likelihood assessment ("likely AI-generated"), never an accusation.

---

## 8. Demo Video & Deployment
* **Demo video (3–5 min):** [Watch on Google Drive](https://drive.google.com/file/d/1rLY3w_Dz0JCrzayGLjvlkKtZDi9RmhhZ/view?usp=sharing)
* **Deployed app:** none; run locally per Section 2.

---

## 9. Originality Declaration

All code in this repository was written by the team between 10 and 15 September 2026. No public real-vs-fake notebook was copied. Third-party components used:

| Component | Use | Licence |
| :--- | :--- | :--- |
| [`dima806/deepfake_vs_real_image_detection`](https://huggingface.co/dima806/deepfake_vs_real_image_detection) | Frozen ensemble member (ViT) | Apache-2.0 |
| [`Organika/sdxl-detector`](https://huggingface.co/Organika/sdxl-detector) | Frozen ensemble member (Swin) | CC-BY-NC-3.0 (non-commercial) |
| torchvision ResNet34 ImageNet weights | Dual-stream spatial backbone initialisation | BSD-3-Clause |
| PyTorch, Hugging Face Transformers, scikit-learn, FastAPI, Pillow, NumPy, c2pa-python | Libraries | Respective open-source licences |
| CIFAKE dataset (Bird & Lotfi, 2024) | Training / evaluation data | See Section 3 |

AI coding assistants were used during development, as the rules permit.
