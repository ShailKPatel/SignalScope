"""
SignalScope Dataset & Data Loader Module
Loads CIFAKE (Real vs. AI-Generated, 32x32) from its train/{REAL,FAKE} and
test/{REAL,FAKE} folders. Validation is carved out of train per class; the
official test split stays held out. Includes FFT frequency spectrum calculation.
"""

import os
import json
import random
from PIL import Image
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

CIFAKE_CLASSES = (("REAL", 0), ("FAKE", 1))
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def extract_fft_numpy(img_pil, target_size=(32, 32)):
    """
    Computes 2D FFT log-magnitude spectrum as a NumPy array [1, H, W].
    """
    img_gray = img_pil.convert("L").resize(target_size)
    arr = np.array(img_gray, dtype=np.float32) / 255.0

    # 2D FFT
    fft = np.fft.fft2(arr)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    log_spectrum = np.log(magnitude + 1e-8)

    # Normalize to [0, 1]
    min_v = log_spectrum.min()
    max_v = log_spectrum.max()
    norm_spectrum = (log_spectrum - min_v) / (max_v - min_v + 1e-8)

    return np.expand_dims(norm_spectrum, axis=0).astype(np.float32)


class SignalScopeImageDataset:
    """
    Core dataset class for indexing CIFAKE REAL and FAKE image samples.
    """
    def __init__(self, data_list, target_size=(32, 32), transform=None):
        """
        data_list: List of dicts [{"path": str, "label": int (0: REAL, 1: FAKE), "split": str, "generator": str}]
        """
        self.data_list = data_list
        self.target_size = target_size
        self.transform = transform

        if HAS_TORCH and self.transform is None:
            # Resize + center crop mirrors the native_size / image_size transform in model/predict.py.
            self.transform = transforms.Compose([
                transforms.Resize(self.target_size[0]),
                transforms.CenterCrop(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        img_path = item["path"]
        label = item["label"]
        generator = item.get("generator", "unknown")

        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
        else:
            # Fallback synthetic image if file missing (for manifest dry-runs)
            img = Image.new("RGB", self.target_size, color=(128, 128, 128))

        if HAS_TORCH:
            tensor_img = self.transform(img)
            label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
            return tensor_img, label_tensor, generator, img_path
        else:
            img_resized = img.resize(self.target_size)
            arr = np.array(img_resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
            return arr, label, generator, img_path


if HAS_TORCH:
    class PyTorchSignalScopeDataset(Dataset):
        def __init__(self, data_list, target_size=(32, 32), transform=None):
            self.base_ds = SignalScopeImageDataset(data_list, target_size=target_size, transform=transform)

        def __len__(self):
            return len(self.base_ds)

        def __getitem__(self, idx):
            return self.base_ds[idx]


def build_split_dataloaders(data_dir="retrain/data", manifest_path=None, batch_size=32, val_fraction=0.10, image_size=32, seed=42, max_samples=None):
    """
    Scans the CIFAKE folders under data_dir (or loads a manifest), holds out
    val_fraction of each train class for validation, keeps CIFAKE's test split
    as the held-out set, and returns DataLoaders for train, validation, and test.
    """
    random.seed(seed)
    items = []

    # 1. Load from manifest if one is given
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            items = json.load(f).get("samples", [])

    # 2. Otherwise scan the CIFAKE layout: data_dir/{train,test}/{REAL,FAKE}
    if not items and os.path.exists(data_dir):
        for split in ("train", "test"):
            for cls_name, cls_label in CIFAKE_CLASSES:
                cls_dir = os.path.join(data_dir, split, cls_name)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in sorted(os.listdir(cls_dir)):
                    if fname.lower().endswith(IMAGE_EXTS):
                        items.append({
                            "path": os.path.join(cls_dir, fname),
                            "label": cls_label,
                            "split": split,
                            "generator": "stable_diffusion_v1_4" if cls_label == 1 else "cifar10"
                        })

    if not items:
        print(f"Warning: No CIFAKE images found in {data_dir} (expected train/ and test/, each with REAL/ and FAKE/). "
              "Download CIFAKE there, or run retrain/dataset_generator.py for placeholder images.")
        return None, None, None, []

    if max_samples and max_samples > 0:
        # Folders are read class by class, so shuffle before slicing or the subset is single-class.
        random.shuffle(items)
        items = items[:max_samples]

    splits = {i.get("split") for i in items}
    if not splits & {"train", "test"}:
        # No split metadata at all: random partition, val_fraction each for val and test
        random.shuffle(items)
        n_hold = int(len(items) * val_fraction)
        for n, item in enumerate(items):
            item["split"] = "val" if n < n_hold else "test" if n < 2 * n_hold else "train"
    elif "val" not in splits:
        # CIFAKE ships train/test only: hold out val_fraction of each train class
        for label in (0, 1):
            cls_items = [i for i in items if i.get("split") == "train" and i["label"] == label]
            random.shuffle(cls_items)
            for item in cls_items[:int(len(cls_items) * val_fraction)]:
                item["split"] = "val"

    train_items = [i for i in items if i.get("split") == "train"]
    val_items = [i for i in items if i.get("split") == "val"]
    test_items = [i for i in items if i.get("split") == "test"]

    print(f"Dataset split loaded successfully:")
    print(f"  Train samples: {len(train_items)}")
    print(f"  Val samples:   {len(val_items)}")
    print(f"  Test samples:  {len(test_items)}")
    print(f"  Total samples: {len(train_items) + len(val_items) + len(test_items)}")

    if HAS_TORCH:
        target_size = (image_size, image_size)
        train_ds = PyTorchSignalScopeDataset(train_items, target_size=target_size)
        val_ds = PyTorchSignalScopeDataset(val_items, target_size=target_size)
        test_ds = PyTorchSignalScopeDataset(test_items, target_size=target_size)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        return train_loader, val_loader, test_loader, items
    else:
        return train_items, val_items, test_items, items


if __name__ == "__main__":
    print("Testing retrain/dataset.py...")
    tr, va, te, items = build_split_dataloaders("retrain/data")
    print("Dataset module test finished successfully!")
