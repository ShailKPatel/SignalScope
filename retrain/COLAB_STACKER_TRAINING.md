# Training the Stacking Meta-Learner on Google Colab

This guide trains the ensemble's stacking meta-learner (`retrain/train_stacker.py`) on Google Colab using the **CIFAKE test split**, then brings the result back into the project.

The paths inside the zip must match the project layout, because the code imports `model.*` and `retrain.*` and looks for the dual-stream checkpoint at `retrain/checkpoints/best_model.pt`.

> **Note on speed:** the ensemble code never moves the models to the GPU. In Colab they run on the CPU even on a GPU runtime, so scoring 4,000 images through 3 models will be slow. The score cache means a disconnect won't lose progress.

## Why the CIFAKE test split

The meta-learner must see honest, out-of-sample member scores. The dual-stream member is trained on CIFAKE `train/` (by `retrain/kaggle_train_cifake.ipynb`), so stacking on `train/` would make it look better than it is. Stack on CIFAKE `test/` instead. `train_stacker.py` still holds back 20% of the sampled images (`--test-size`) for its own held-out comparison.

## What to upload

| Upload | Why |
|---|---|
| `model/` (whole folder, skip `__pycache__`) | `model/__init__.py` imports `predict.py`, which imports every other module in the folder |
| `retrain/train_stacker.py` | the training script |
| `retrain/eval_majority_vote.py` | the training script uses its CIFAKE loader and metrics |
| `retrain/backbone.py` | defines the dual-stream model |
| `retrain/checkpoints/best_model.pt` | the CIFAKE-trained dual-stream weights, downloaded from the `kaggle_train_cifake.ipynb` output |

**Skip:**
- `trained-v1/best_model.zip`: it was trained on ArtiFact, not CIFAKE. The loader tries `trained-v1` **before** `retrain/checkpoints`, so leaving it out makes the CIFAKE checkpoint the one that loads.
- The two Hugging Face models: Colab downloads them automatically.
- CIFAKE itself: Colab downloads it in Step 2.

## Step 1: Build the zip (PowerShell, in the project root)

```powershell
$s = "$env:TEMP\signalscope_colab"
Remove-Item $s -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory "$s\retrain\checkpoints" -Force | Out-Null
Copy-Item model "$s\model" -Recurse
Remove-Item "$s\model\__pycache__" -Recurse -Force -ErrorAction SilentlyContinue
Copy-Item retrain\backbone.py, retrain\eval_majority_vote.py, retrain\train_stacker.py "$s\retrain"
Copy-Item retrain\checkpoints\best_model.pt "$s\retrain\checkpoints"
Compress-Archive "$s\*" signalscope_colab.zip -Force
```

Upload `signalscope_colab.zip` to the top level of **My Drive** in Google Drive. That's faster than Colab's upload button, and the files survive a disconnect.

## Step 2: Set up Colab (one cell each)

**Cell 1: mount Drive, unzip, install packages**

```python
from google.colab import drive
drive.mount('/content/drive')
!unzip -q -o /content/drive/MyDrive/signalscope_colab.zip -d /content/SIGNALSCOPE
%cd /content/SIGNALSCOPE
!pip install -q scikit-learn pandas transformers kagglehub
```

**Cell 2: download CIFAKE (about 100 MB)**

```python
import kagglehub, os
CIFAKE = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
os.environ["CIFAKE_TEST"] = os.path.join(CIFAKE, "test")
print(os.environ["CIFAKE_TEST"], os.listdir(os.environ["CIFAKE_TEST"]))  # expect ['FAKE', 'REAL'] in some order
```

If `kagglehub` asks for credentials, run `kagglehub.login()` first and paste your Kaggle API token.

**Cell 3: check all 3 members load (should print `True` 3 times)**

```python
!python -c "from model.ensemble import *; print('dual_stream', load_dual_stream() is not None); [print(m['id'], load_ensemble_model(m['name'])[0] is not None) for m in ENSEMBLE_MEMBERS if m['id'] != 'dual_stream_freq']"
```

The dual-stream line should also report loading `retrain/checkpoints/best_model.pt`. If it names `trained-v1`, remove that folder from the zip.

## Step 3: Train

```python
!mkdir -p /content/drive/MyDrive/signalscope_stacker
!python retrain/train_stacker.py \
  --data-dir "$CIFAKE_TEST" \
  --n 4000 --target-fpr 0.02 \
  --cache /content/drive/MyDrive/signalscope_stacker/scores.npz \
  --out   /content/drive/MyDrive/signalscope_stacker/stacking_metalearner.json
```

- **`--data-dir`:** the CIFAKE `test/` folder. A balanced sample of `--n` images (half REAL, half FAKE) is drawn from it; the split has 10,000 per class, so `--n` can go up to 20,000.
- **`--target-fpr 0.02`:** sets the threshold so about 2% of real images get flagged, which matches the `operating_point_fpr: 0.02` the app reports. Leave it out to use 0.5.
- **If Colab disconnects:** re-run cells 1–3, then this command with exactly the same arguments. The cache is keyed on the data folder, `--n` and `--seed`, so scoring resumes where it stopped.

When it finishes, the output shows the held-out comparison: each member, the majority vote, and the stacked meta-learner.

## Step 4: Bring the model back

1. Download `MyDrive/signalscope_stacker/stacking_metalearner.json`.
2. Put it in your local project at `retrain/checkpoints/stacking_metalearner.json`.
3. Restart the app. Nothing else needs to change: the `ensemble_breakdown` in the API response should now show `"Stacked Generalization (ridge logistic meta-learner)"` instead of the majority-vote fallback.

> Locally the app still prefers `trained-v1/best_model.zip` when it exists. The meta-learner's weights were fitted on the CIFAKE checkpoint's scores, so move `trained-v1/best_model.zip` aside (or replace it) to keep inference consistent with what the stacker learned.
