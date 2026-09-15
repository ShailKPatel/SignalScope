"""
SignalScope - majority-vote baseline.

Nothing is trained here. Every member is already trained and stays frozen; each
casts one real/AI vote and the majority wins. This establishes the reference
number that any learned fusion has to beat.

Kaggle setup:
  Add Data -> birdy654/cifake-real-and-ai-generated-synthetic-images
  Add Data -> Notebook Output -> the training run holding best_model.pt
  Settings -> GPU + Internet ON
"""

# %% [markdown]
# # Majority-Vote Baseline
# Frozen members, one vote each, majority wins.
#
# Evaluated on a balanced sample of the **CIFAKE test split**, which the
# dual-stream member never saw during training, plus a CIFAKE train sample for
# context and for catching members whose labels are inverted.

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
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

N_TEST = 4000
N_REF = 2000
BATCH = 32

# %% [markdown]
# ## 1. Sample CIFAKE

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


ref_df = balanced(list_split("train"), N_REF)
test_df = balanced(list_split("test"), N_TEST)
print(f"train sample={len(ref_df)} (fake {ref_df.label.sum()})   test sample={len(test_df)} (fake {test_df.label.sum()})")

# %% [markdown]
# ## 2. Load the frozen members

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


ckpt_paths = glob.glob("/kaggle/input/**/best_model.pt", recursive=True)
if not ckpt_paths:
    raise SystemExit("best_model.pt not found - add the training notebook's output as a data source.")
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

# %% [markdown]
# ## 3. Score every member (forward passes only)

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


MEMBER_NAMES = HF_MEMBERS + ["signalscope_dual_stream"]


def member_matrix(df, tag):
    print(f"\n--- scoring {tag} ({len(df)} images) ---")
    cols = [hf_probs(n, df) for n in HF_MEMBERS]
    cols.append(dual_probs(df))
    return np.column_stack(cols)


X_ref = member_matrix(ref_df, "CIFAKE train sample")
y_ref = ref_df["label"].values
X_test = member_matrix(test_df, "CIFAKE test sample")
y_test = test_df["label"].values

# A member whose label mapping is reversed scores below chance; correct it here
# so the vote is not being cast backwards. Decided on train so test stays untouched.
for j, name in enumerate(MEMBER_NAMES):
    auc = roc_auc_score(y_ref, X_ref[:, j])
    if auc < 0.5:
        print(f"flipping inverted member {name} (train-sample AUC {auc:.3f})")
        X_ref[:, j] = 1.0 - X_ref[:, j]
        X_test[:, j] = 1.0 - X_test[:, j]

# %% [markdown]
# ## 4. Majority vote
#
# Each member votes AI if its probability is >= 0.5. With an even number of
# members a 2-2 tie is possible; ties resolve to *real*, since wrongly flagging a
# genuine photo is the costly error.

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
    print(f"{name:42s} {auc_txt}acc={m['accuracy']*100:5.1f}%  F1={m['macro_f1']:.4f}  "
          f"FPR={m['fpr']*100:5.1f}%  recall={m['recall_on_fakes']*100:5.1f}%")
    return m


def majority(X, tie_breaks_real=True):
    votes = (X >= 0.5).astype(int)
    counts = votes.sum(axis=1)
    n = X.shape[1]
    preds = (counts > n / 2).astype(int)
    if not tie_breaks_real and n % 2 == 0:
        preds = np.where(counts == n / 2, 1, preds)
    return preds, counts / n


results = {}
for tag, X, y in [("CIFAKE train sample", X_ref, y_ref), ("CIFAKE test", X_test, y_test)]:
    print(f"\n=== {tag} ===")
    per_member = {}
    for j, name in enumerate(MEMBER_NAMES):
        per_member[name] = evaluate(name, y, (X[:, j] >= 0.5).astype(int), X[:, j])

    preds_all, frac_all = majority(X)
    m_all = evaluate("MAJORITY VOTE (4 members, tie=real)", y, preds_all, frac_all)

    preds_hf, frac_hf = majority(X[:, :3])
    m_hf = evaluate("MAJORITY VOTE (3 HF members, no ties)", y, preds_hf, frac_hf)

    m_mean = evaluate("mean-probability soft vote", y, (X.mean(axis=1) >= 0.5).astype(int), X.mean(axis=1))

    results[tag] = {
        "members": per_member,
        "majority_4_tie_real": m_all,
        "majority_3_hf": m_hf,
        "mean_soft_vote": m_mean,
    }

# %% [markdown]
# ## 5. Save the baseline

# %%
baseline = {
    "note": "No training performed. All members frozen; simple majority vote.",
    "dataset": f"CIFAKE (birdy654/{CIFAKE_SLUG})",
    "members": MEMBER_NAMES,
    "tie_rule": "even split resolves to real",
    "train_sample_size": int(len(ref_df)),
    "test_set_size": int(len(test_df)),
    "dual_stream_checkpoint": os.path.basename(ckpt_paths[0]),
    "results": results,
}

with open("/kaggle/working/majority_baseline.json", "w") as f:
    json.dump(baseline, f, indent=2)

print("\nSaved /kaggle/working/majority_baseline.json")
print("\nHeadline baseline (CIFAKE test split):",
      f"majority-vote accuracy {results['CIFAKE test']['majority_4_tie_real']['accuracy']*100:.1f}%")
