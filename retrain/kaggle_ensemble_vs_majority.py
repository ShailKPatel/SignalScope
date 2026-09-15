"""
SignalScope - ensemble combiner vs majority-vote baseline.

No classifier is trained or retrained. Every member is loaded frozen. The only
thing fitted is the fusion layer, and it is compared against an untrained
majority vote over the same members on the same images.

Kaggle setup:
  Add Data -> birdy654/cifake-real-and-ai-generated-synthetic-images
  Add Data -> Notebook Output -> the training run holding best_model.pt
  Settings -> GPU + Internet ON
"""

# %% [markdown]
# # Ensemble Combiner vs Majority-Vote Baseline
#
# Members are frozen - nothing is trained except the fusion layer.
#
# * **Baseline:** every member votes AI if p >= 0.5, majority wins. No fitting.
# * **Learned:** a logistic regression over the members' probabilities.
#
# Both are scored on the same sample of the **CIFAKE test split**, which the
# combiner never sees during fitting.

# %%
import os
import json
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

N_FIT = 4000
N_TEST = 4000
BATCH = 32

# %% [markdown]
# ## 1. Splits
# `fit` is sampled from CIFAKE train and `test` from the official CIFAKE test
# split, so the two sets never share an image.

# %%
CIFAKE_SLUG = "cifake-real-and-ai-generated-synthetic-images"
CIFAKE_CANDIDATES = [f"/kaggle/input/datasets/birdy654/{CIFAKE_SLUG}", f"/kaggle/input/{CIFAKE_SLUG}"]
CIFAKE_ROOT = next((p for p in CIFAKE_CANDIDATES if os.path.isdir(os.path.join(p, "train", "REAL"))), None)
if CIFAKE_ROOT is None:
    raise SystemExit(f"CIFAKE not found under {CIFAKE_CANDIDATES}. Confirm the dataset is attached via Add Data.")
print("CIFAKE root:", CIFAKE_ROOT)


def list_split(split):
    """One row per image in a CIFAKE split. label: 0 = REAL, 1 = FAKE."""
    rows = []
    for cls, label in (("REAL", 0), ("FAKE", 1)):
        d = os.path.join(CIFAKE_ROOT, split, cls)
        rows += [(os.path.join(d, f), label) for f in sorted(os.listdir(d))]
    return pd.DataFrame(rows, columns=["abspath", "label"])


def balanced(df, n):
    k = min(n // 2, *df["label"].value_counts().tolist())
    return df.groupby("label").sample(k, random_state=SEED).sample(frac=1.0, random_state=SEED).reset_index(drop=True)


fit_df = balanced(list_split("train"), N_FIT)
test_df = balanced(list_split("test"), N_TEST)
print(f"fit set={len(fit_df)} (fake {fit_df.label.sum()})   test set={len(test_df)} (fake {test_df.label.sum()})")

# %% [markdown]
# ## 2. Frozen members

# %%
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
        gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3] if x.shape[1] == 3 else x
        spec = torch.log(torch.abs(torch.fft.fftshift(torch.fft.fft2(gray))) + 1e-8)
        flat = spec.view(spec.size(0), -1)
        mn = flat.min(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        mx = flat.max(dim=1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        return (spec - mn) / (mx - mn + 1e-8)

    def forward(self, x):
        f = self.extract_fft_spectrum(x)
        f = F.relu(self.bn1(self.conv1(f)))
        f = F.relu(self.bn2(self.conv2(f)))
        f = F.relu(self.bn3(self.conv3(f)))
        return F.relu(self.fc(torch.flatten(self.adaptive_pool(f), 1)))


class SignalScopeDualStreamModel(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        base = tv_models.resnet34(weights=None)
        num_spatial = base.fc.in_features
        base.fc = nn.Identity()
        self.spatial_stream = base
        self.frequency_stream = FrequencyBranch(1, 128)
        self.classifier = nn.Sequential(
            nn.Linear(num_spatial + 128, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(dropout_rate), nn.Linear(256, 64), nn.ReLU(),
            nn.Dropout(dropout_rate / 2), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.classifier(torch.cat((self.spatial_stream(x), self.frequency_stream(x)), dim=1))


def rezip_checkpoint(folder):
    """Kaggle auto-extracts uploaded .pt files (they are zip archives) into a
    folder. Rebuild the archive so torch.load can read it. /kaggle/input is
    read-only, so the rebuilt file goes to /kaggle/working."""
    import zipfile
    prefix = os.path.basename(folder.rstrip("/"))
    out = "/kaggle/working/best_model.pt"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:
        zf.write(os.path.join(folder, "data.pkl"), f"{prefix}/data.pkl")
        for root, _, files in os.walk(folder):
            for fn in files:
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, folder).replace(os.sep, "/")
                if rel != "data.pkl":
                    zf.write(full, f"{prefix}/{rel}")
    return out


# Walk /kaggle/input but prune the CIFAKE tree - no need to list its 120k images.
ckpt_paths = []
for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
    dirnames[:] = [d for d in dirnames if d != CIFAKE_SLUG]
    if "best_model.pt" in filenames:
        ckpt_paths.append(os.path.join(dirpath, "best_model.pt"))
        break
    if "data.pkl" in filenames and "version" in filenames:
        print("found extracted checkpoint folder, rebuilding archive:", dirpath)
        ckpt_paths.append(rezip_checkpoint(dirpath))
        break

if not ckpt_paths:
    print("Mounted inputs (excluding CIFAKE):")
    for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
        dirnames[:] = [d for d in dirnames if d != CIFAKE_SLUG]
        depth = dirpath.count(os.sep) - 2
        if depth <= 4:
            print("  " * depth + dirpath, filenames[:5])
    raise SystemExit("best_model.pt not found - attach the training notebook's output via Add Data.")
print("dual-stream checkpoint:", ckpt_paths[0])

ckpt = torch.load(ckpt_paths[0], map_location=DEVICE, weights_only=False)
dual = SignalScopeDualStreamModel().to(DEVICE)
dual.load_state_dict(ckpt["model_state_dict"])
dual.eval()
DUAL_TEMPERATURE = float(ckpt.get("temperature", 1.0))

NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
if "image_size" in ckpt:
    dual_tf = T.Compose([T.Resize(ckpt.get("native_size", 200)), T.CenterCrop(ckpt["image_size"]), T.ToTensor(), NORM])
else:
    # Checkpoints without a recorded size came from the plain 224-resize pipeline.
    dual_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), NORM])
print("dual-stream transform:", dual_tf)

# %%
from transformers import AutoImageProcessor, AutoModelForImageClassification

HF_MEMBERS = [
    "dima806/deepfake_vs_real_image_detection",
    "umm-maybe/AI-image-detector",
    "Organika/sdxl-detector",
]
AI_TOKENS = ("fake", "artificial", "synthetic", "generated")

hf_models = {}
for name in HF_MEMBERS:
    proc = AutoImageProcessor.from_pretrained(name)
    mdl = AutoModelForImageClassification.from_pretrained(name).to(DEVICE).eval()
    idx = next((int(i) for i, lbl in mdl.config.id2label.items()
                if any(tok in lbl.lower() for tok in AI_TOKENS)), 1)
    print(f"{name}: id2label={mdl.config.id2label} -> AI index {idx}")
    hf_models[name] = (proc, mdl, idx)

MEMBER_NAMES = HF_MEMBERS + ["signalscope_dual_stream"]

# %% [markdown]
# ## 3. Score every member once
# Forward passes only. This is the slow part, so it runs a single time and both
# methods reuse the result - which also guarantees they see identical images.

# %%
@torch.no_grad()
def hf_probs(name, df):
    proc, mdl, idx = hf_models[name]
    paths = df["abspath"].tolist()
    out = []
    for start in range(0, len(paths), BATCH):
        imgs = [Image.open(p).convert("RGB") for p in paths[start:start + BATCH]]
        batch = proc(images=imgs, return_tensors="pt").to(DEVICE)
        out.append(torch.softmax(mdl(**batch).logits, dim=1)[:, idx].float().cpu().numpy())
        if start % (BATCH * 25) == 0:
            print(f"   {name} {start}/{len(paths)}")
    return np.concatenate(out)


@torch.no_grad()
def dual_probs(df):
    paths = df["abspath"].tolist()
    out = []
    for start in range(0, len(paths), BATCH):
        imgs = torch.stack([dual_tf(Image.open(p).convert("RGB")) for p in paths[start:start + BATCH]]).to(DEVICE)
        out.append(torch.sigmoid(dual(imgs) / DUAL_TEMPERATURE).squeeze(1).float().cpu().numpy())
    return np.concatenate(out)


def member_matrix(df, tag):
    print(f"\n--- scoring {tag} ({len(df)} images) ---")
    cols = [hf_probs(n, df) for n in HF_MEMBERS]
    cols.append(dual_probs(df))
    return np.column_stack(cols)


X_fit = member_matrix(fit_df, "fit set")
y_fit = fit_df["label"].values
X_test = member_matrix(test_df, "CIFAKE test set")
y_test = test_df["label"].values

# A member whose label mapping is reversed scores below chance; correct it so it
# is not voting backwards.
for j, name in enumerate(MEMBER_NAMES):
    auc = roc_auc_score(y_fit, X_fit[:, j])
    if auc < 0.5:
        print(f"flipping inverted member {name} (fit AUC {auc:.3f})")
        X_fit[:, j] = 1.0 - X_fit[:, j]
        X_test[:, j] = 1.0 - X_test[:, j]

# %% [markdown]
# ## 4. Evaluate

# %%
def evaluate(name, y, preds, scores=None):
    cm = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()
    m = {
        "accuracy": float((tp + tn) / cm.sum()),
        "macro_f1": float(f1_score(y, preds, average="macro")),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "recall_on_fakes": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if scores is not None:
        m["auc"] = float(roc_auc_score(y, scores))
    auc_txt = f"AUC={m['auc']:.4f}  " if "auc" in m else " " * 11
    print(f"{name:44s} {auc_txt}acc={m['accuracy']*100:5.1f}%  F1={m['macro_f1']:.4f}  "
          f"FPR={m['fpr']*100:5.1f}%  recall={m['recall_on_fakes']*100:5.1f}%")
    return m


# --- individual members, for reference -------------------------------------
print("\n=== individual members on the CIFAKE test split ===")
member_metrics = {
    n: evaluate(n, y_test, (X_test[:, j] >= 0.5).astype(int), X_test[:, j])
    for j, n in enumerate(MEMBER_NAMES)
}

# --- baseline: untrained majority vote -------------------------------------
def majority(X):
    """Ties resolve to real - wrongly flagging a genuine photo is the costly error."""
    votes = (X >= 0.5).astype(int)
    counts = votes.sum(axis=1)
    return (counts > X.shape[1] / 2).astype(int), counts / X.shape[1]


print("\n=== BASELINE: majority vote (nothing trained) ===")
preds_maj_fit, frac_maj_fit = majority(X_fit)
m_maj_fit = evaluate("majority vote - fit set", y_fit, preds_maj_fit, frac_maj_fit)
preds_maj, frac_maj = majority(X_test)
m_maj_test = evaluate("majority vote - CIFAKE test split", y_test, preds_maj, frac_maj)
preds_hf, frac_hf = majority(X_test[:, :3])
m_maj_hf = evaluate("majority of 3 HF only - CIFAKE test split", y_test, preds_hf, frac_hf)

# --- learned combiner -------------------------------------------------------
combiner = LogisticRegression(max_iter=1000, C=1.0)
combiner.fit(X_fit, y_fit)
p_fit = combiner.predict_proba(X_fit)[:, 1]
p_test = combiner.predict_proba(X_test)[:, 1]

print("\n=== LEARNED: logistic-regression combiner ===")
m_lr_fit = evaluate("learned combiner - fit set (in-sample)", y_fit, (p_fit >= 0.5).astype(int), p_fit)
m_lr_test = evaluate("learned combiner - CIFAKE test split", y_test, (p_test >= 0.5).astype(int), p_test)

LOW_FPR_THRESHOLD = float(np.quantile(p_fit[y_fit == 0], 0.95))
m_lr_lowfpr = evaluate(f"learned combiner @ 5% FPR (t={LOW_FPR_THRESHOLD:.3f})",
                       y_test, (p_test >= LOW_FPR_THRESHOLD).astype(int), p_test)

print("\nlearned weights:")
for n, w in zip(MEMBER_NAMES, combiner.coef_[0]):
    print(f"  {n:42s} {w:+.4f}")
print(f"  {'intercept':42s} {combiner.intercept_[0]:+.4f}")

# --- side by side -----------------------------------------------------------
print("\n" + "=" * 78)
print("SIDE BY SIDE on the CIFAKE test split")
print("=" * 78)
print(f"{'method':44s} {'AUC':>7s} {'acc':>7s} {'F1':>7s} {'FPR':>7s}")
for label, m in [
    ("BASELINE majority vote (untrained)", m_maj_test),
    ("BASELINE majority of 3 HF (untrained)", m_maj_hf),
    ("LEARNED logistic combiner", m_lr_test),
    ("LEARNED combiner @ 5% FPR", m_lr_lowfpr),
]:
    print(f"{label:44s} {m.get('auc', float('nan')):7.4f} {m['accuracy']*100:6.1f}% "
          f"{m['macro_f1']:7.4f} {m['fpr']*100:6.1f}%")

# %% [markdown]
# ## 5. Save

# %%
spec = {
    "note": "No classifier trained or retrained. Members frozen; only the fusion layer is fitted.",
    "dataset": f"CIFAKE (birdy654/{CIFAKE_SLUG})",
    "members": MEMBER_NAMES,
    "dual_stream_checkpoint": os.path.basename(ckpt_paths[0]),
    "fit_size": int(len(fit_df)),
    "test_size": int(len(test_df)),
    "tie_rule": "even split resolves to real",
    "baseline_majority": {"fit": m_maj_fit, "test": m_maj_test, "test_3hf": m_maj_hf},
    "learned_combiner": {
        "coefficients": combiner.coef_[0].tolist(),
        "intercept": float(combiner.intercept_[0]),
        "low_fpr_threshold": LOW_FPR_THRESHOLD,
        "fit_in_sample": m_lr_fit,
        "test": m_lr_test,
        "test_at_5pct_fpr": m_lr_lowfpr,
    },
    "individual_members_test": member_metrics,
}

with open("/kaggle/working/ensemble_vs_majority.json", "w") as f:
    json.dump(spec, f, indent=2)

print("\nSaved /kaggle/working/ensemble_vs_majority.json")
