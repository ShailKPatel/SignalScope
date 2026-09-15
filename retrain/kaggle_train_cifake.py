"""
SignalScope - CIFAKE training pipeline for Kaggle GPU notebooks.

Attach the dataset `birdy654/cifake-real-and-ai-generated-synthetic-images` and
enable a GPU accelerator, then run top to bottom. Produces
retrain/checkpoints/best_model.pt in the format model/predict.py expects.
"""

# %% [markdown]
# # SignalScope - Real vs AI-Generated Detector
# Dual-stream (spatial + FFT) training on the CIFAKE dataset.
#
# **Before running:** Add Data -> search `cifake-real-and-ai-generated-synthetic-images` (birdy654) -> Add.
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
# ## 1. Locate the dataset
# CIFAKE ships `train/{REAL,FAKE}` (50k + 50k) and `test/{REAL,FAKE}` (10k + 10k).
# REAL images come from CIFAR-10; FAKE images were generated with Stable Diffusion v1.4.
# Every image is 32x32.

# %%
CIFAKE_SLUG = "cifake-real-and-ai-generated-synthetic-images"
# Kaggle mounts attached datasets under either path depending on the notebook version.
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


train_pool = list_split("train")
test_df = list_split("test")
for name, df in (("train", train_pool), ("test", test_df)):
    print(name, len(df), df["label"].value_counts().rename({0: "real", 1: "fake"}).to_dict())

# The Dataset falls back to a blank image on read errors, which would silently
# mask a wrong path join across the whole run. Fail loudly here instead.
missing = [p for p in train_pool.sample(30, random_state=SEED)["abspath"] if not os.path.exists(p)]
assert not missing, f"broken abspath join, examples: {missing[:3]}"
print("\npath spot-check OK")

# %% [markdown]
# ## 2. Split design
#
# The official CIFAKE test split is the held-out evaluation set. It is never used
# for training, epoch selection or calibration. Validation - which picks the best
# epoch and fits the temperature - is a stratified 10% slice of CIFAKE train.
#
# CIFAKE is built from CIFAR-10 objects and scenes, so there are no identifiable
# people to filter out.

# %%
VAL_FRACTION = 0.10

val_df = train_pool.groupby("label").sample(frac=VAL_FRACTION, random_state=SEED)
train_df = train_pool.drop(index=val_df.index).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

assert not set(train_df["abspath"]) & set(val_df["abspath"]), "train/val leakage"
print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

# %% [markdown]
# ## 3. Augmentation
#
# CIFAKE's REAL and FAKE images went through different encoding pipelines.
# Randomising JPEG compression keeps the model from learning that fingerprint
# instead of the generation artifacts.
#
# Training stays at the native 32x32: upsampling would interpolate away the
# high-frequency fingerprints the FFT branch relies on. The checkpoint records
# native_size so inference resizes inputs down to the same resolution.

# %%
class RandomJPEG:
    def __init__(self, qmin=50, qmax=95, p=0.5):
        self.qmin, self.qmax, self.p = qmin, qmax, p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=random.randint(self.qmin, self.qmax))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


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

eval_tf = T.Compose([
    T.Resize(NATIVE),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
    NORM,
])


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


BATCH = 256
WORKERS = 2

train_loader = DataLoader(CIFAKEDataset(train_df, train_tf), batch_size=BATCH,
                          shuffle=True, num_workers=WORKERS, pin_memory=True, drop_last=True)
val_loader = DataLoader(CIFAKEDataset(val_df, eval_tf), batch_size=BATCH,
                        shuffle=False, num_workers=WORKERS, pin_memory=True)
test_loader = DataLoader(CIFAKEDataset(test_df, eval_tf), batch_size=BATCH,
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
        if step % 100 == 0:
            print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {running/seen:.4f}")

    scheduler.step()
    y_val, p_val, _ = evaluate(val_loader)
    val_auc = roc_auc_score(y_val, p_val)
    print(f"epoch {epoch}: train_loss={running/seen:.4f}  val_auc={val_auc:.4f}")

    # Select on validation only - the CIFAKE test split stays untouched until the final report.
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(
            {"model_state_dict": model.state_dict(), "spatial_backbone": BACKBONE, "val_auc": val_auc},
            os.path.join(CKPT_DIR, "best_model.pt"),
        )
        print(f"  saved new best (val_auc {val_auc:.4f})")

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
m_val = report("Validation (10% of CIFAKE train)", y_v, p_v_cal)

# Wrongly flagging a real photo is the costly error (spec 4.2), so pick the
# operating point from the validation reals rather than defaulting to 0.5.
LOW_FPR_THRESHOLD = float(np.quantile(p_v_cal[y_v == 0], 0.95))
print(f"\nthreshold for ~5% FPR on validation: {LOW_FPR_THRESHOLD:.4f}")
m_val_lowfpr = report("Validation @ 5% FPR operating point", y_v, p_v_cal, LOW_FPR_THRESHOLD)

y_t, p_t, lg_t = evaluate(test_loader)
p_t_cal = torch.sigmoid(lg_t / TEMPERATURE).numpy()
m_test = report("CIFAKE test split", y_t, p_t_cal)
m_test_lowfpr = report("CIFAKE test split @ 5% FPR operating point", y_t, p_t_cal, LOW_FPR_THRESHOLD)

metrics = {
    "dataset": f"CIFAKE (birdy654/{CIFAKE_SLUG})",
    "backbone": BACKBONE,
    "image_size": IMG_SIZE,
    "epochs": EPOCHS,
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
     "low_fpr_threshold": LOW_FPR_THRESHOLD, "metrics": metrics},
    os.path.join(CKPT_DIR, "best_model.pt"),
)

with open("/kaggle/working/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved best_model.pt and metrics.json to /kaggle/working/")
print("Download both from the notebook Output panel.")
