"""
SignalScope - one-shot Kaggle pipeline on CIFAKE.

A single GPU session that produces every file the 3-member stacked ensemble needs:
  1. Dual-stream member (ResNet34 + 2D FFT) trained on CIFAKE train/
     -> best_model.pt, metrics.json
  2. Ridge-logistic stacking meta-learner over
     [dima806 ViT, Organika Swin, dual-stream], fitted on the validation split
     and scored on the full CIFAKE test split
     -> stacking_metalearner.json
All three are bundled into /kaggle/working/signalscope_outputs.zip.

Splits (no leakage): train 90k -> dual-stream weights; validation 10k -> epoch
selection, temperature, stacker fit; test 20k -> reported metrics only.

Self-contained on purpose: the repo is private, so nothing is cloned. The model
definition mirrors retrain/backbone.py, the label mapping mirrors model/labels.py,
and the stacker fit and JSON format mirror retrain/train_stacker.py, so the
outputs load in model/ensemble.py unchanged.

Build the notebook with: python retrain/build_kaggle_notebook.py
Dry-run locally with SS_CIFAKE_ROOT, SS_WORK_DIR, SS_ALLOW_CPU, SS_WORKERS=0 and
the small SS_* sizes below.
"""

# %% [markdown]
# # SignalScope: CIFAKE training + stacking meta-learner (one run)
#
# **Before clicking Run All**
# 1. Dataset attached: `birdy654/cifake-real-and-ai-generated-synthetic-images` (right sidebar, Input).
# 2. Session options: **Accelerator = GPU** (T4 x2 or P100), **Internet = On**.
#
# The first cell stops with a clear message if the GPU, internet or dataset is missing.
# Expected run time: about 25-40 minutes. When the last cell prints `ALL DONE`,
# download `signalscope_outputs.zip` from the Output panel (`/kaggle/working`).
#
# Splits: train 90k (weights) / validation 10k (epoch, temperature, stacker fit) / test 20k (reported only).

# %%
import io
import json
import os
import random
import time
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification

T_START = time.time()
WORK = os.environ.get("SS_WORK_DIR", "/kaggle/working")
EPOCHS = int(os.environ.get("SS_EPOCHS", "6"))
TRAIN_BUDGET_MIN = float(os.environ.get("SS_TRAIN_BUDGET_MIN", "25"))  # stop adding epochs past this
STACK_N = int(os.environ.get("SS_STACK_N", "4000"))                  # balanced validation images for the stacker fit
BATCH = int(os.environ.get("SS_BATCH", "256"))
MAX_TRAIN = int(os.environ.get("SS_MAX_TRAIN", "0"))                 # 0 = full train split; >0 only for dry runs
SEED = 42
os.makedirs(WORK, exist_ok=True)


def elapsed_min():
    return (time.time() - T_START) / 60


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu" and not os.environ.get("SS_ALLOW_CPU"):
    raise RuntimeError("NO GPU. Right sidebar -> Session options -> Accelerator -> GPU, then Run All again.")
print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")


def find_cifake():
    if os.environ.get("SS_CIFAKE_ROOT"):
        return os.environ["SS_CIFAKE_ROOT"]
    # Kaggle has mounted attached datasets at several depths over time; search instead of guessing.
    for root, dirs, _ in os.walk("/kaggle/input"):
        if all(os.path.isdir(os.path.join(root, s, c)) for s in ("train", "test") for c in ("REAL", "FAKE")):
            return root
        dirs[:] = [d for d in dirs if d not in ("REAL", "FAKE")] if root.count(os.sep) < 8 else []
    return None


CIFAKE_ROOT = find_cifake()
if CIFAKE_ROOT is None:
    raise RuntimeError("CIFAKE not found. Right sidebar -> Input -> Add Input -> search "
                       "'cifake-real-and-ai-generated-synthetic-images' (birdy654) -> Add, then Run All again.")
print("CIFAKE root:", CIFAKE_ROOT)

# Order must match ENSEMBLE_MEMBERS in model/ensemble.py.
HF_MEMBERS = [("vit_dima806", "dima806/deepfake_vs_real_image_detection"),
              ("swin_sdxl", "Organika/sdxl-detector")]
MEMBER_IDS = [mid for mid, _ in HF_MEMBERS] + ["dual_stream_freq"]

# Download everything that needs internet now, so a missing toggle fails in seconds, not after training.
try:
    for _, name in HF_MEMBERS:
        AutoImageProcessor.from_pretrained(name)
        AutoModelForImageClassification.from_pretrained(name)
    tv_models.resnet34(weights=tv_models.ResNet34_Weights.DEFAULT)
except Exception as e:
    raise RuntimeError(f"Model download failed ({e}). Session options -> Internet -> On, then Run All again.")
print(f"setup OK ({elapsed_min():.1f} min)")

# %% [markdown]
# ## Part 1 - Dual-stream member on CIFAKE train
# Validation (epoch selection + temperature) is a stratified 10% of train. The test split is only reported on.

# %%
def list_split(split):
    """One row per image in a CIFAKE split. label: 0 = REAL, 1 = FAKE."""
    rows = []
    for cls, label in (("REAL", 0), ("FAKE", 1)):
        d = os.path.join(CIFAKE_ROOT, split, cls)
        rows += [(os.path.join(d, f), label) for f in sorted(os.listdir(d))]
    return pd.DataFrame(rows, columns=["abspath", "label"])


train_pool = list_split("train")
test_df = list_split("test")
val_df = train_pool.groupby("label").sample(frac=0.10, random_state=SEED)
train_df = train_pool.drop(index=val_df.index).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
if MAX_TRAIN:
    train_df = train_df.head(MAX_TRAIN)
assert not set(train_df["abspath"]) & set(val_df["abspath"]), "train/val leakage"
print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")


class RandomJPEG:
    """REAL and FAKE went through different encoders; random re-compression hides that shortcut."""
    def __init__(self, qmin=50, qmax=95, p=0.5):
        self.qmin, self.qmax, self.p = qmin, qmax, p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(self.qmin, self.qmax))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


# Native 32x32: upsampling would blur away the high-frequency fingerprints the FFT branch uses.
IMG_SIZE = 32
NATIVE = 32
NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_tf = T.Compose([
    RandomJPEG(qmin=50, qmax=95, p=0.5),
    T.Resize(NATIVE),
    T.RandomCrop(IMG_SIZE, padding=4, padding_mode="reflect"),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    T.ToTensor(),
    NORM,
])
# Same as the inference transform model/ensemble.py builds from native_size/image_size.
eval_tf = T.Compose([T.Resize(NATIVE), T.CenterCrop(IMG_SIZE), T.ToTensor(), NORM])


class CIFAKEDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        try:
            img = Image.open(row["abspath"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (127, 127, 127))
        return self.transform(img), torch.tensor([float(row["label"])])


WORKERS = int(os.environ.get("SS_WORKERS", min(8, os.cpu_count() or 2)))
loader_kw = dict(batch_size=BATCH, num_workers=WORKERS, pin_memory=(DEVICE == "cuda"), persistent_workers=WORKERS > 0)
train_loader = DataLoader(CIFAKEDataset(train_df, train_tf), shuffle=True, drop_last=True, **loader_kw)
val_loader = DataLoader(CIFAKEDataset(val_df, eval_tf), shuffle=False, **loader_kw)
test_loader = DataLoader(CIFAKEDataset(test_df, eval_tf), shuffle=False, **loader_kw)


# Mirrors retrain/backbone.py exactly so the checkpoint loads locally without changes.
class FrequencyBranch(nn.Module):
    def __init__(self, in_channels=1, feature_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(128 * 4 * 4, feature_dim)

    def extract_fft_spectrum(self, x):
        if x.shape[1] == 3:
            gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
        else:
            gray = x
        fft_shift = torch.fft.fftshift(torch.fft.fft2(gray))
        log_spectrum = torch.log(torch.abs(fft_shift) + 1e-8)
        flat = log_spectrum.view(log_spectrum.size(0), -1)
        min_v = flat.min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        max_v = flat.max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        return (log_spectrum - min_v) / (max_v - min_v + 1e-8)

    def forward(self, x):
        feat = self.extract_fft_spectrum(x)
        feat = F.relu(self.bn1(self.conv1(feat)))
        feat = F.relu(self.bn2(self.conv2(feat)))
        feat = F.relu(self.bn3(self.conv3(feat)))
        feat = torch.flatten(self.adaptive_pool(feat), 1)
        return F.relu(self.fc(feat))


class SignalScopeDualStreamModel(nn.Module):
    def __init__(self, spatial_backbone="resnet34", pretrained=False, dropout_rate=0.3):
        super().__init__()
        base = tv_models.resnet34(weights=tv_models.ResNet34_Weights.DEFAULT if pretrained else None)
        num_spatial_features = base.fc.in_features
        base.fc = nn.Identity()
        self.spatial_stream = base
        self.freq_dim = 128
        self.frequency_stream = FrequencyBranch(in_channels=1, feature_dim=self.freq_dim)
        self.classifier = nn.Sequential(
            nn.Linear(num_spatial_features + self.freq_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        combined = torch.cat((self.spatial_stream(x), self.frequency_stream(x)), dim=1)
        return self.classifier(combined)


BACKBONE = "resnet34"
model = SignalScopeDualStreamModel(spatial_backbone=BACKBONE, pretrained=True).to(DEVICE)
print(sum(p.numel() for p in model.parameters()) / 1e6, "M params")

# %%
LR = 3e-4
CKPT = os.path.join(WORK, "best_model.pt")
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))


@torch.no_grad()
def evaluate(loader):
    model.eval()
    logits_all, labels_all = [], []
    for imgs, labels in loader:
        with torch.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
            logits = model(imgs.to(DEVICE, non_blocking=True))
        logits_all.append(logits.float().cpu())
        labels_all.append(labels)
    logits_all = torch.cat(logits_all).squeeze(1)
    labels_all = torch.cat(labels_all).squeeze(1)
    return labels_all.numpy(), torch.sigmoid(logits_all).numpy(), logits_all


best_auc, epochs_run, epoch_secs = -1.0, 0, []
t_train = time.time()
for epoch in range(1, EPOCHS + 1):
    t_epoch = time.time()
    model.train()
    running, seen = 0.0, 0
    for step, (imgs, labels) in enumerate(train_loader, 1):
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=(DEVICE == "cuda")):
            loss = criterion(model(imgs), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss.item() * imgs.size(0)
        seen += imgs.size(0)
        if step % 100 == 0:
            print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {running / seen:.4f}")
    scheduler.step()

    y_val, p_val, _ = evaluate(val_loader)
    val_auc = roc_auc_score(y_val, p_val)
    epochs_run = epoch
    epoch_secs.append(time.time() - t_epoch)
    print(f"epoch {epoch}: train_loss={running / max(seen, 1):.4f}  val_auc={val_auc:.4f}  "
          f"({epoch_secs[-1] / 60:.1f} min, total {elapsed_min():.1f} min)")

    # Select on validation only - the CIFAKE test split stays untouched until the final report.
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save({"model_state_dict": model.state_dict(), "spatial_backbone": BACKBONE, "val_auc": val_auc}, CKPT)
        print(f"  saved new best (val_auc {val_auc:.4f})")

    if epoch < EPOCHS and (time.time() - t_train + max(epoch_secs)) / 60 > TRAIN_BUDGET_MIN:
        print(f"time budget of {TRAIN_BUDGET_MIN} min reached; stopping after epoch {epoch}")
        break

del train_loader

# %% [markdown]
# ## Calibration + metrics
# Temperature scaling on validation makes confidence honest. A second threshold targets ~5% FPR on validation reals.

# %%
model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=False)["model_state_dict"])

y_val, _, val_logits = evaluate(val_loader)
log_t = torch.zeros(1, requires_grad=True)
opt_t = torch.optim.LBFGS([log_t], lr=0.1, max_iter=60)
val_targets = torch.tensor(y_val, dtype=torch.float32)


def _closure():
    opt_t.zero_grad()
    loss = F.binary_cross_entropy_with_logits(val_logits / torch.exp(log_t), val_targets)
    loss.backward()
    return loss


opt_t.step(_closure)
TEMPERATURE = float(torch.exp(log_t).item())
print("temperature:", round(TEMPERATURE, 4))


def report(name, y, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    auc = roc_auc_score(y, probs)
    macro_f1 = f1_score(y, preds, average="macro")
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    acc = (tp + tn) / (tn + fp + fn + tp)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    print(f"\n=== {name} ===")
    print(f"ROC-AUC   : {auc:.4f}")
    print(f"Macro-F1  : {macro_f1:.4f}")
    print(f"Accuracy  : {acc * 100:.2f}%   FPR: {fpr * 100:.2f}%  @ threshold {threshold:.4f}")
    print(f"Confusion : TN={tn} FP={fp} FN={fn} TP={tp}")
    return {"auc": float(auc), "macro_f1": float(macro_f1), "accuracy": float(acc), "fpr": float(fpr),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


y_v, _, lg_v = evaluate(val_loader)
p_v_cal = torch.sigmoid(lg_v / TEMPERATURE).numpy()
m_val = report("Validation (10% of CIFAKE train)", y_v, p_v_cal)
LOW_FPR_THRESHOLD = float(np.quantile(p_v_cal[y_v == 0], 0.95))
m_val_lowfpr = report("Validation @ 5% FPR operating point", y_v, p_v_cal, LOW_FPR_THRESHOLD)

y_t, _, lg_t = evaluate(test_loader)
p_t_cal = torch.sigmoid(lg_t / TEMPERATURE).numpy()
m_test = report("CIFAKE test split", y_t, p_t_cal)
m_test_lowfpr = report("CIFAKE test split @ 5% FPR operating point", y_t, p_t_cal, LOW_FPR_THRESHOLD)

metrics_json = {
    "dataset": "CIFAKE (birdy654/cifake-real-and-ai-generated-synthetic-images)",
    "backbone": BACKBONE,
    "image_size": IMG_SIZE,
    "epochs": epochs_run,
    "epochs_planned": EPOCHS,
    "best_val_auc": float(best_auc),
    "temperature": TEMPERATURE,
    "low_fpr_threshold": LOW_FPR_THRESHOLD,
    "train_size": len(train_df),
    "val_size": len(val_df),
    "test_size": len(test_df),
    "validation": m_val,
    "validation_at_5pct_fpr": m_val_lowfpr,
    "test": m_test,
    "test_at_5pct_fpr": m_test_lowfpr,
}
torch.save(
    {"model_state_dict": model.state_dict(), "spatial_backbone": BACKBONE,
     "temperature": TEMPERATURE, "image_size": IMG_SIZE, "native_size": NATIVE,
     "low_fpr_threshold": LOW_FPR_THRESHOLD, "metrics": metrics_json},
    CKPT,
)
with open(os.path.join(WORK, "metrics.json"), "w") as f:
    json.dump(metrics_json, f, indent=2)
print(f"\nsaved best_model.pt + metrics.json ({elapsed_min():.1f} min)")

# %% [markdown]
# ## Part 2 - Stacking meta-learner + ensemble test metrics
# No leakage: the stacker is fitted on a balanced sample of the **validation** split (no member trained on it),
# and the stacked ensemble is scored on the **full CIFAKE test split**, which nothing has been fitted on.
# One cell, relying only on Part 1's variables, so it can be pasted and re-run on its own in the same session.
# Same fit and JSON format as retrain/train_stacker.py.

# %%
import re
from concurrent.futures import ThreadPoolExecutor

_EXACT = {
    "fake": True, "deepfake": True, "artificial": True, "ai": True, "aiartdata": True,
    "synthetic": True, "generated": True, "aigenerated": True,
    "real": False, "realism": False, "human": False, "realart": False,
    "authentic": False, "natural": False,
}


def ai_class_index(id2label):
    """Mirrors model/labels.py: the logit index meaning AI-generated. Raises instead of guessing."""
    def means_ai(label):
        key = re.sub(r"[^a-z]", "", str(label).lower())
        if key in _EXACT:
            return _EXACT[key]
        if any(t in key for t in ("fake", "artificial", "synthetic", "generated")):
            return True
        if any(t in key for t in ("real", "human", "authentic")):
            return False
        return None

    ai = [int(i) for i, lbl in id2label.items() if means_ai(lbl) is True]
    real = [int(i) for i, lbl in id2label.items() if means_ai(lbl) is False]
    if len(ai) == 1 and len(real) == len(id2label) - 1:
        return ai[0]
    raise ValueError(f"Cannot tell which class means AI-generated from id2label={id2label}")


def score_hf(name, paths, batch=128, chunk=8):
    """P(AI) per path. The member's own processor does the 224 resize, as at inference; fp16 on GPU for speed."""
    proc = AutoImageProcessor.from_pretrained(name)
    mdl = AutoModelForImageClassification.from_pretrained(name).to(DEVICE).eval()
    ai_idx = ai_class_index(mdl.config.id2label)

    def prep(batch_paths):
        return proc(images=[Image.open(p).convert("RGB") for p in batch_paths], return_tensors="pt")["pixel_values"]

    batches = [paths[s:s + batch] for s in range(0, len(paths), batch)]
    out = []
    with torch.no_grad(), ThreadPoolExecutor(4) as pool:
        for k in range(0, len(batches), chunk):  # bounded prefetch keeps RAM flat
            for x in pool.map(prep, batches[k:k + chunk]):
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
                    logits = mdl(pixel_values=x.to(DEVICE, non_blocking=True)).logits
                out.append(torch.softmax(logits.float(), dim=-1)[:, ai_idx].cpu().numpy())
    print(f"  {name}: AI index {ai_idx} of {mdl.config.id2label}")
    del mdl
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(out)


def logit_features(P, eps=1e-6):
    P = np.clip(np.asarray(P, dtype=np.float64), eps, 1.0 - eps)
    return np.log(P) - np.log1p(-P)


def stack_metrics(y, preds, scores=None):
    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
    m = {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "macro_f1": float(f1_score(y, preds, average="macro", zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if scores is not None and len(set(y)) == 2:
        m["auc"] = float(roc_auc_score(y, scores))
    return m


# Fit set: balanced sample of validation. Eval set: all of CIFAKE test.
per_class = min(STACK_N // 2, int(val_df["label"].value_counts().min()))
fit_idx = (val_df.groupby("label").sample(n=per_class, random_state=SEED)
           .sample(frac=1.0, random_state=SEED).index.to_numpy())
fit_paths, y_fit = val_df.loc[fit_idx, "abspath"].tolist(), val_df.loc[fit_idx, "label"].to_numpy()
test_paths, y_test = test_df["abspath"].tolist(), test_df["label"].to_numpy()
# Part 1's calibrated dual-stream probabilities are row-aligned with val_df / test_df; reuse them.
assert np.array_equal(y_v[fit_idx], y_fit) and np.array_equal(y_t, y_test), "row order mismatch with Part 1"
print(f"stacker fit: {len(y_fit)} validation images   ensemble eval: {len(y_test)} test images")

P_fit = np.zeros((len(fit_paths), len(MEMBER_IDS)))
P_test = np.zeros((len(test_paths), len(MEMBER_IDS)))
for j, (mid, name) in enumerate(HF_MEMBERS):
    t_m = time.time()
    scores = score_hf(name, fit_paths + test_paths)
    P_fit[:, j], P_test[:, j] = scores[:len(fit_paths)], scores[len(fit_paths):]
    print(f"scored {mid} in {time.time() - t_m:.0f}s")
P_fit[:, -1], P_test[:, -1] = p_v_cal[fit_idx], p_t_cal

FOLDS = 5
X_fit, X_test = logit_features(P_fit), logit_features(P_test)
cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
search = make_pipeline(StandardScaler(), LogisticRegressionCV(
    Cs=list(np.logspace(-4, 2, 13)), cv=cv, penalty="l2", solver="lbfgs", scoring="neg_log_loss", max_iter=5000))
search.fit(X_fit, y_fit)
C = float(search.named_steps["logisticregressioncv"].C_[0])
print(f"selected C = {C:.4g}")

pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=5000))
p_oof = cross_val_predict(clone(pipe), X_fit, y_fit, cv=cv, method="predict_proba")[:, 1]
threshold = 0.5
cv_metrics = stack_metrics(y_fit, (p_oof >= threshold).astype(int), p_oof)
stack_low_fpr = float(np.quantile(p_oof[y_fit == 0], 0.95))  # ~5% FPR, chosen on validation only

pipe.fit(X_fit, y_fit)
p_stack = pipe.predict_proba(X_test)[:, 1]
test_metrics = stack_metrics(y_test, (p_stack >= threshold).astype(int), p_stack)
test_metrics_lowfpr = stack_metrics(y_test, (p_stack >= stack_low_fpr).astype(int), p_stack)

baseline = {n: stack_metrics(y_test, (P_test[:, j] >= 0.5).astype(int), P_test[:, j]) for j, n in enumerate(MEMBER_IDS)}
votes = (P_test >= 0.5).sum(axis=1)  # ties resolve to real, as in model/ensemble.py majority_vote
baseline["majority_vote"] = stack_metrics(y_test, (votes > len(MEMBER_IDS) / 2).astype(int), votes / len(MEMBER_IDS))

header = f"{'model':34s} {'acc':>7s} {'prec':>7s} {'recall':>7s} {'f1':>7s} {'auc':>7s} {'fpr':>7s}"


def show(name, m):
    print(f"{name:34s} {m['accuracy']:.4f} {m['precision']:.4f} {m['recall']:.4f} "
          f"{m['f1']:.4f} {m.get('auc', float('nan')):7.4f} {m['fpr']:.4f}")


print(f"\nOUT-OF-FOLD on validation fit set ({len(y_fit)} images)\n{header}")
show("STACKED meta-learner (OOF)", cv_metrics)
print(f"\nCIFAKE TEST SPLIT ({len(y_test)} images)\n{header}")
for n in MEMBER_IDS:
    show(n, baseline[n])
show("MAJORITY VOTE", baseline["majority_vote"])
print("-" * len(header))
show("STACKED meta-learner @ 0.5", test_metrics)
show(f"STACKED @ 5% val FPR ({stack_low_fpr:.3f})", test_metrics_lowfpr)

scaler_step, lr_step = pipe.named_steps["standardscaler"], pipe.named_steps["logisticregression"]
print("\nmeta-learner weights (standardized member logits):")
for n, w in zip(MEMBER_IDS, lr_step.coef_[0]):
    print(f"  {n:28s} {w:+.4f}")
print(f"  {'intercept':28s} {lr_step.intercept_[0]:+.4f}")

spec = {
    "type": "stacking_logistic_regression",
    "penalty": "l2",
    "C": C,
    "features": "logit(clip(P(AI), eps, 1 - eps)) per member, standardized",
    "eps": 1e-6,
    "members": MEMBER_IDS,
    "scaler_mean": scaler_step.mean_.tolist(),
    "scaler_scale": scaler_step.scale_.tolist(),
    "coefficients": lr_step.coef_[0].tolist(),
    "intercept": float(lr_step.intercept_[0]),
    "threshold": threshold,
    "threshold_rule": "0.5",
    "low_fpr_threshold": stack_low_fpr,
    "training": {
        "source": {"dataset": "CIFAKE", "fit_split": "validation (10% of CIFAKE train), balanced sample",
                   "eval_split": "CIFAKE test (full)", "n": len(y_fit)},
        "n_meta_train": len(y_fit),
        "n_test": len(y_test),
        "cv_folds": FOLDS,
        "seed": SEED,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    },
    "oof_cv_metrics": cv_metrics,
    "test_metrics": test_metrics,
    "test_metrics_at_5pct_fpr": test_metrics_lowfpr,
    "test_baselines": baseline,
}
with open(os.path.join(WORK, "stacking_metalearner.json"), "w") as f:
    json.dump(spec, f, indent=2)

bundle = os.path.join(WORK, "signalscope_outputs.zip")
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_STORED) as z:
    for name in ("best_model.pt", "metrics.json", "stacking_metalearner.json"):
        z.write(os.path.join(WORK, name), name)
print(f"\nCIFAKE test ROC-AUC  dual-stream: {m_test['auc']:.4f}   stacked ensemble: {test_metrics['auc']:.4f}")
print(f"ALL DONE (stacker v2: fit on validation, scored on full test) in {elapsed_min():.1f} min. "
      f"Download {bundle} from the Output panel.")
