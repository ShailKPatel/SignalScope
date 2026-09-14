"""
SignalScope - ArtiFact training pipeline for Kaggle GPU notebooks.

Attach the dataset `awsaf49/artifact-dataset` and enable a GPU accelerator,
then run top to bottom. Produces retrain/checkpoints/best_model.pt in the
format model/predict.py expects.
"""

# %% [markdown]
# # SignalScope - Real vs AI-Generated Detector
# Dual-stream (spatial + FFT) training on the ArtiFact dataset.
#
# **Before running:** Add Data -> search `artifact-dataset` (awsaf49) -> Add.
# Then Settings -> Accelerator -> GPU T4 x2 or P100.

# %%
import os
import json
import random
import io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)
if DEVICE == "cuda":
    print("gpu:", torch.cuda.get_device_name(0))

# %% [markdown]
# ## 1. Discover the dataset layout
# ArtiFact ships one folder per real source / generator, each with a metadata.csv.
# Folder names vary, so we discover rather than hardcode.

# %%
ARTIFACT_ROOT = "/kaggle/input/datasets/awsaf49/artifact-dataset"

# Probe the 33 source directories directly. os.walk over /kaggle/input would
# enumerate all 2.5M images on a slow network mount and take tens of minutes.
folders = {}
for name in sorted(os.listdir(ARTIFACT_ROOT)):
    d = os.path.join(ARTIFACT_ROOT, name)
    if os.path.isdir(d) and os.path.exists(os.path.join(d, "metadata.csv")):
        folders[name] = d

print(f"found {len(folders)} source folders:")
for name in sorted(folders):
    print(" ", name)

if not folders:
    raise SystemExit(
        f"No metadata.csv found under {ARTIFACT_ROOT}. "
        "Confirm the artifact-dataset is attached via Add Data."
    )

# %% [markdown]
# ## 2. Ethics filter + split design
#
# The problem statement disqualifies work involving real, identifiable individuals,
# so every face-derived source and generator is dropped before sampling.
#
# The unseen-generator split holds two diffusion generators out of training
# entirely - that split is the primary ranking metric and the first tie-break.

# %%
# Sources that are entirely face data.
FACE_EXCLUDE = {
    "ffhq", "celebahq", "metfaces", "face_synthetics", "sfhq",
    "stylegan1", "stylegan2", "stylegan3", "star_gan", "mat",
}

# Face subsets also hide inside non-face folders - stable_diffusion ships
# "stable-face/...", taming_transformer ships "tt-ffhq/..." - so the same rule
# must run against every image path, not just the folder name.
FACE_PATH_TOKENS = ("face", "ffhq", "celeba")

TARGET_PER_CLASS = 50_000
VAL_FRACTION = 0.10

rows = []
for name, path in sorted(folders.items()):
    if name.lower() in FACE_EXCLUDE:
        print(f"excluded folder (faces): {name}")
        continue

    meta = pd.read_csv(os.path.join(path, "metadata.csv"))[["image_path", "target"]].copy()

    face_mask = meta["image_path"].str.lower().str.contains(
        "|".join(FACE_PATH_TOKENS), regex=True, na=False
    )
    dropped = int(face_mask.sum())
    meta = meta[~face_mask]

    meta["source"] = name
    # target is a generator-family code: 0 = real, each nonzero value a family.
    meta["generator"] = meta["target"].astype(int)
    meta["label"] = (meta["generator"] != 0).astype(int)
    meta["abspath"] = path.rstrip("/") + "/" + meta["image_path"].astype(str)

    rows.append(meta[["abspath", "label", "generator", "source"]])
    print(f"kept {name}: {len(meta)}" + (f"  (dropped {dropped} face images)" if dropped else ""))

catalog = pd.concat(rows, ignore_index=True)

print("\ntotal usable images:", len(catalog))
print("\nreal vs fake:")
print(catalog["label"].value_counts().rename({0: "real", 1: "fake"}).to_string())
print("\nper source (count, fake_ratio):")
print(catalog.groupby("source")["label"].agg(["count", "mean"]).to_string())

# The Dataset falls back to a blank image on read errors, which would silently
# mask a wrong path join across the whole run. Fail loudly here instead.
missing = [p for p in catalog.sample(30, random_state=SEED)["abspath"] if not os.path.exists(p)]
assert not missing, f"broken abspath join, examples: {missing[:3]}"
print("\npath spot-check OK")

# %%
# Hold out whole diffusion generators so the unseen-split AUC measures real
# generalisation. Chosen from what survived the face filter, since a folder can
# shrink drastically once its face subsets are removed.
PREFERRED_UNSEEN = ["stable_diffusion", "glide", "vq_diffusion", "latent_diffusion"]
MIN_UNSEEN_IMAGES = 2000

fake_counts = catalog[catalog["label"] == 1].groupby("source").size()
UNSEEN_GENERATORS = [g for g in PREFERRED_UNSEEN if fake_counts.get(g, 0) >= MIN_UNSEEN_IMAGES][:2]
if not UNSEEN_GENERATORS:
    raise SystemExit(f"No generator has >= {MIN_UNSEEN_IMAGES} images post-filter: {fake_counts.to_dict()}")
print("held-out generators:", UNSEEN_GENERATORS)

# A third generator, also never trained on, used purely to choose the best epoch.
# Selecting on the seen-generator validation set optimises for the thing that
# does not transfer, and the reported unseen split must stay untouched.
DEV_GENERATOR = next(
    (g for g in ["vq_diffusion", "latent_diffusion", "big_gan"]
     if g not in UNSEEN_GENERATORS and fake_counts.get(g, 0) >= MIN_UNSEEN_IMAGES),
    None,
)
print("model-selection generator:", DEV_GENERATOR)

catalog["unseen"] = catalog["source"].isin(UNSEEN_GENERATORS)
catalog["dev"] = catalog["source"] == DEV_GENERATOR

unseen_pool = catalog[catalog["unseen"] & (catalog["label"] == 1)]
dev_pool = catalog[catalog["dev"] & (catalog["label"] == 1)]
seen_pool = catalog[~catalog["unseen"] & ~catalog["dev"]]

real_pool = seen_pool[seen_pool["label"] == 0]
fake_pool = seen_pool[seen_pool["label"] == 1]

print(f"real available: {len(real_pool)}  fake available: {len(fake_pool)}  unseen fake: {len(unseen_pool)}")


def sample_balanced(pool, n):
    """Spread the sample evenly across sources so no single source dominates.

    Keeps the original catalog index so sampled rows can be excluded from the
    held-out split later - resetting it here would silently leak train images
    into the test set.
    """
    sources = pool["source"].unique()
    per_source = max(1, n // len(sources))
    picked = [
        grp.sample(min(per_source, len(grp)), random_state=SEED)
        for _, grp in pool.groupby("source")
    ]
    out = pd.concat(picked)
    if len(out) > n:
        out = out.sample(n, random_state=SEED)
    return out


real_sample = sample_balanced(real_pool, TARGET_PER_CLASS)
fake_sample = sample_balanced(fake_pool, TARGET_PER_CLASS)
print(f"sampled real={len(real_sample)} fake={len(fake_sample)}")

# Reserve held-out real images BEFORE shuffling, using the preserved index, then
# split them so the dev and unseen sets never share a real image either.
leftover_real = real_pool.drop(index=real_sample.index, errors="ignore")
assert len(leftover_real.index.intersection(real_sample.index)) == 0, "real leakage into held-out splits"

leftover_real = leftover_real.sample(frac=1.0, random_state=SEED)
half = len(leftover_real) // 2
dev_real, unseen_real = leftover_real.iloc[:half], leftover_real.iloc[half:]

trainval = pd.concat([real_sample, fake_sample]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

n_val = int(len(trainval) * VAL_FRACTION)
val_df = trainval.iloc[:n_val].reset_index(drop=True)
train_df = trainval.iloc[n_val:].reset_index(drop=True)


def build_holdout(fake_rows, real_rows, cap=5000):
    n = min(len(fake_rows), cap, len(real_rows))
    if n == 0:
        raise SystemExit("No held-out images available - check generator names.")
    return pd.concat([
        fake_rows.sample(n, random_state=SEED),
        real_rows.sample(n, random_state=SEED),
    ], ignore_index=True)


unseen_test = build_holdout(unseen_pool, unseen_real)
dev_test = build_holdout(dev_pool, dev_real, cap=3000)

print(f"train={len(train_df)}  val={len(val_df)}  dev={len(dev_test)}  unseen_test={len(unseen_test)}")
print("held-out generators:", sorted(unseen_pool["source"].unique()))

# %% [markdown]
# ## 3. Augmentation
#
# Every ArtiFact image is 200x200 and JPEG-compressed. Without randomising the
# compression and scale, the model learns that fingerprint instead of the
# generation artifacts and collapses on out-of-distribution inputs.

# %%
class RandomJPEG:
    def __init__(self, qmin=30, qmax=95, p=0.7):
        self.qmin, self.qmax, self.p = qmin, qmax, p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(self.qmin, self.qmax))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomRescale:
    def __init__(self, sizes=(128, 160, 200, 256, 320), p=0.5):
        self.sizes, self.p = sizes, p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        s = random.choice(self.sizes)
        return img.resize((s, s), Image.BICUBIC)


# Source images are 200x200. Upsampling them to 224 interpolates away the
# high-frequency generator fingerprints the FFT branch relies on, so crop at
# native resolution instead - Resize(200) is a no-op unless augmentation
# already rescaled the image.
IMG_SIZE = 192
NATIVE = 200
NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_tf = T.Compose([
    RandomJPEG(qmin=30, qmax=95, p=0.8),
    RandomRescale(sizes=(160, 200, 224, 256), p=0.4),
    T.Resize(NATIVE),
    T.RandomCrop(IMG_SIZE),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
    T.ToTensor(),
    T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.35),
    NORM,
])

eval_tf = T.Compose([
    T.Resize(NATIVE),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    NORM,
])


class ArtifactDataset(Dataset):
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


BATCH = 64
WORKERS = 2

train_loader = DataLoader(ArtifactDataset(train_df, train_tf), batch_size=BATCH,
                          shuffle=True, num_workers=WORKERS, pin_memory=True, drop_last=True)
val_loader = DataLoader(ArtifactDataset(val_df, eval_tf), batch_size=BATCH,
                        shuffle=False, num_workers=WORKERS, pin_memory=True)
dev_loader = DataLoader(ArtifactDataset(dev_test, eval_tf), batch_size=BATCH,
                        shuffle=False, num_workers=WORKERS, pin_memory=True)
unseen_loader = DataLoader(ArtifactDataset(unseen_test, eval_tf), batch_size=BATCH,
                           shuffle=False, num_workers=WORKERS, pin_memory=True)

# %% [markdown]
# ## 4. Model
# Mirrors retrain/backbone.py exactly so the checkpoint loads locally without changes.

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
        if spatial_backbone == "resnet18":
            base = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT if pretrained else None)
            num_spatial_features = base.fc.in_features
            base.fc = nn.Identity()
        elif spatial_backbone == "resnet50":
            base = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT if pretrained else None)
            num_spatial_features = base.fc.in_features
            base.fc = nn.Identity()
        elif spatial_backbone == "efficientnet_b0":
            base = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            num_spatial_features = base.classifier[1].in_features
            base.classifier = nn.Identity()
        else:
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

    def predict_probability(self, x):
        return torch.sigmoid(self.forward(x))


BACKBONE = "resnet34"
model = SignalScopeDualStreamModel(spatial_backbone=BACKBONE, pretrained=True).to(DEVICE)
print(sum(p.numel() for p in model.parameters()) / 1e6, "M params")

# %% [markdown]
# ## 5. Train

# %%
EPOCHS = 8
LR = 3e-4

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))


@torch.no_grad()
def evaluate(loader):
    model.eval()
    logits_all, labels_all = [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            logits = model(imgs)
        logits_all.append(logits.float().cpu())
        labels_all.append(labels)
    logits_all = torch.cat(logits_all).squeeze(1)
    labels_all = torch.cat(labels_all).squeeze(1)
    probs = torch.sigmoid(logits_all).numpy()
    y = labels_all.numpy()
    return y, probs, logits_all


CKPT_DIR = "/kaggle/working/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
best_auc = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    running, seen = 0.0, 0
    for step, (imgs, labels) in enumerate(train_loader, 1):
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            loss = criterion(model(imgs), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running += loss.item() * imgs.size(0)
        seen += imgs.size(0)
        if step % 200 == 0:
            print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {running/seen:.4f}")

    scheduler.step()
    y_val, p_val, _ = evaluate(val_loader)
    val_auc = roc_auc_score(y_val, p_val)
    y_dev, p_dev, _ = evaluate(dev_loader)
    dev_auc = roc_auc_score(y_dev, p_dev)
    print(f"epoch {epoch}: train_loss={running/seen:.4f}  val_auc={val_auc:.4f}  dev_auc={dev_auc:.4f}")

    # Select on the held-out generator, not the seen-generator validation set.
    if dev_auc > best_auc:
        best_auc = dev_auc
        torch.save(
            {"model_state_dict": model.state_dict(), "spatial_backbone": BACKBONE,
             "val_auc": val_auc, "dev_auc": dev_auc},
            os.path.join(CKPT_DIR, "best_model.pt"),
        )
        print(f"  saved new best (dev_auc {dev_auc:.4f})")

# %% [markdown]
# ## 6. Calibration
# A raw sigmoid is overconfident. Temperature scaling on the validation set makes
# the reported confidence honest, which the explanation rubric rewards.

# %%
ckpt = torch.load(os.path.join(CKPT_DIR, "best_model.pt"), map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])

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

# %% [markdown]
# ## 7. Metrics - these are the numbers that go in the README

# %%
def report(name, y, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    auc = roc_auc_score(y, probs)
    macro_f1 = f1_score(y, preds, average="macro")
    cm = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()
    acc = (tp + tn) / cm.sum()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    print(f"\n=== {name} ===")
    print(f"ROC-AUC   : {auc:.4f}")
    print(f"Macro-F1  : {macro_f1:.4f}")
    print(f"Accuracy  : {acc*100:.2f}%   FPR: {fpr*100:.2f}%  @ threshold {threshold}")
    print(f"Confusion : TN={tn} FP={fp} FN={fn} TP={tp}")
    return {"auc": auc, "macro_f1": macro_f1, "accuracy": acc, "fpr": fpr,
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


y_v, p_v, lg_v = evaluate(val_loader)
p_v_cal = torch.sigmoid(lg_v / TEMPERATURE).numpy()
m_val = report("Validation (seen generators)", y_v, p_v_cal)

# Wrongly flagging a real photo is the costly error (spec 4.2), so pick the
# operating point from the validation reals rather than defaulting to 0.5.
LOW_FPR_THRESHOLD = float(np.quantile(p_v_cal[y_v == 0], 0.95))
print(f"\nthreshold for ~5% FPR on validation: {LOW_FPR_THRESHOLD:.4f}")
m_val_lowfpr = report("Validation @ 5% FPR operating point", y_v, p_v_cal, LOW_FPR_THRESHOLD)

y_u, p_u, lg_u = evaluate(unseen_loader)
p_u_cal = torch.sigmoid(lg_u / TEMPERATURE).numpy()
m_unseen = report("Held-out UNSEEN-generator split", y_u, p_u_cal)
m_unseen_lowfpr = report("UNSEEN split @ 5% FPR operating point", y_u, p_u_cal, LOW_FPR_THRESHOLD)

metrics = {
    "backbone": BACKBONE,
    "image_size": IMG_SIZE,
    "epochs": EPOCHS,
    "temperature": TEMPERATURE,
    "low_fpr_threshold": LOW_FPR_THRESHOLD,
    "train_size": len(train_df),
    "val_size": len(val_df),
    "unseen_test_size": len(unseen_test),
    "held_out_generators": sorted(unseen_pool["source"].unique().tolist()),
    "model_selection_generator": DEV_GENERATOR,
    "excluded_face_sources": sorted(FACE_EXCLUDE),
    "validation": m_val,
    "validation_at_5pct_fpr": m_val_lowfpr,
    "unseen_generator_split": m_unseen,
    "unseen_at_5pct_fpr": m_unseen_lowfpr,
}

torch.save(
    {"model_state_dict": model.state_dict(), "spatial_backbone": BACKBONE,
     "temperature": TEMPERATURE, "image_size": IMG_SIZE, "native_size": NATIVE,
     "low_fpr_threshold": LOW_FPR_THRESHOLD, "metrics": metrics},
    os.path.join(CKPT_DIR, "best_model.pt"),
)

with open("/kaggle/working/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved best_model.pt and metrics.json to /kaggle/working/")
print("Download both from the notebook Output panel.")
