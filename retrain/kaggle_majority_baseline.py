"""
SignalScope - majority-vote baseline.

Nothing is trained here. Every member is already trained and stays frozen; each
casts one real/AI vote and the majority wins. This establishes the reference
number that any learned fusion has to beat.

Kaggle setup:
  Add Data -> awsaf49/artifact-dataset
  Add Data -> Notebook Output -> the training run holding best_model.pt
  Settings -> GPU + Internet ON
"""

# %% [markdown]
# # Majority-Vote Baseline
# Frozen members, one vote each, majority wins.
#
# Evaluated on the **unseen-generator split** (`stable_diffusion` + `glide`) that
# no member was trained on, plus a seen-generator sample for context.

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

N_UNSEEN = 4000
N_SEEN = 2000
BATCH = 32

# %% [markdown]
# ## 1. Rebuild the same splits

# %%
ARTIFACT_ROOT = "/kaggle/input/datasets/awsaf49/artifact-dataset"

folders = {}
for name in sorted(os.listdir(ARTIFACT_ROOT)):
    d = os.path.join(ARTIFACT_ROOT, name)
    if os.path.isdir(d) and os.path.exists(os.path.join(d, "metadata.csv")):
        folders[name] = d
print(f"found {len(folders)} source folders")

FACE_EXCLUDE = {
    "ffhq", "celebahq", "metfaces", "face_synthetics", "sfhq",
    "stylegan1", "stylegan2", "stylegan3", "star_gan", "mat",
}
FACE_PATH_TOKENS = ("face", "ffhq", "celeba")
UNSEEN_GENERATORS = ["stable_diffusion", "glide"]

rows = []
for name, path in sorted(folders.items()):
    if name.lower() in FACE_EXCLUDE:
        continue
    meta = pd.read_csv(os.path.join(path, "metadata.csv"))[["image_path", "target"]].copy()
    meta = meta[~meta["image_path"].str.lower().str.contains("|".join(FACE_PATH_TOKENS), regex=True, na=False)]
    meta["source"] = name
    meta["label"] = (meta["target"].astype(int) != 0).astype(int)
    meta["abspath"] = path.rstrip("/") + "/" + meta["image_path"].astype(str)
    rows.append(meta[["abspath", "label", "source"]])

catalog = pd.concat(rows, ignore_index=True)
print("total usable images:", len(catalog))

unseen_fake = catalog[catalog["source"].isin(UNSEEN_GENERATORS) & (catalog["label"] == 1)]
seen_fake = catalog[~catalog["source"].isin(UNSEEN_GENERATORS) & (catalog["label"] == 1)]
real_all = catalog[catalog["label"] == 0]

real_shuffled = real_all.sample(frac=1.0, random_state=SEED)
split = len(real_shuffled) // 2
real_a, real_b = real_shuffled.iloc[:split], real_shuffled.iloc[split:]


def balanced(fakes, reals, n):
    half = min(n // 2, len(fakes), len(reals))
    return pd.concat([
        fakes.sample(half, random_state=SEED),
        reals.sample(half, random_state=SEED),
    ], ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)


unseen_df = balanced(unseen_fake, real_a, N_UNSEEN)
seen_df = balanced(seen_fake, real_b, N_SEEN)
print(f"unseen set={len(unseen_df)} (fake {unseen_df.label.sum()})   seen set={len(seen_df)} (fake {seen_df.label.sum()})")

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


X_seen = member_matrix(seen_df, "seen generators")
y_seen = seen_df["label"].values
X_unseen = member_matrix(unseen_df, "UNSEEN generators")
y_unseen = unseen_df["label"].values

# A member whose label mapping is reversed scores below chance; correct it here
# so the vote is not being cast backwards.
for j, name in enumerate(MEMBER_NAMES):
    auc = roc_auc_score(y_seen, X_seen[:, j])
    if auc < 0.5:
        print(f"flipping inverted member {name} (seen AUC {auc:.3f})")
        X_seen[:, j] = 1.0 - X_seen[:, j]
        X_unseen[:, j] = 1.0 - X_unseen[:, j]

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
for tag, X, y in [("SEEN generators", X_seen, y_seen), ("UNSEEN generators", X_unseen, y_unseen)]:
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
    "members": MEMBER_NAMES,
    "tie_rule": "even split resolves to real",
    "held_out_generators": UNSEEN_GENERATORS,
    "seen_set_size": int(len(seen_df)),
    "unseen_set_size": int(len(unseen_df)),
    "dual_stream_checkpoint": os.path.basename(ckpt_paths[0]),
    "results": results,
}

with open("/kaggle/working/majority_baseline.json", "w") as f:
    json.dump(baseline, f, indent=2)

print("\nSaved /kaggle/working/majority_baseline.json")
print("\nHeadline baseline (unseen split):",
      f"majority-vote accuracy {results['UNSEEN generators']['majority_4_tie_real']['accuracy']*100:.1f}%")
