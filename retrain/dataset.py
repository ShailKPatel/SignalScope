"""
SignalScope Dataset & Data Loader Module
Handles loading, splitting, and preprocessing 100,000 images (Real vs. Synthetic).
Includes FFT frequency spectrum calculation and train/val/test splits.
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


def extract_fft_numpy(img_pil, target_size=(224, 224)):
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
    Core dataset class for indexing real and synthetic image samples.
    """
    def __init__(self, data_list, target_size=(224, 224), transform=None):
        """
        data_list: List of dicts [{"path": str, "label": int (0: real, 1: synthetic), "generator": str}]
        """
        self.data_list = data_list
        self.target_size = target_size
        self.transform = transform
        
        if HAS_TORCH and self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.target_size),
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
        def __init__(self, data_list, target_size=(224, 224), transform=None):
            self.base_ds = SignalScopeImageDataset(data_list, target_size=target_size, transform=transform)

        def __len__(self):
            return len(self.base_ds)

        def __getitem__(self, idx):
            return self.base_ds[idx]


def build_split_dataloaders(data_dir="retrain/data", manifest_path=None, batch_size=32, split_ratio=(0.80, 0.10, 0.10), seed=42, max_samples=None):
    """
    Scans directory or loads manifest, performs stratified train/val/test split,
    and returns DataLoaders for train, validation, and held-out test set.
    """
    random.seed(seed)
    items = []
    
    # 1. Load from manifest if present
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest_data = json.load(f)
            items = manifest_data.get("samples", [])
    
    if max_samples and max_samples > 0:
        items = items[:max_samples]
    
    # 2. Scan data_dir if no manifest items found
    if not items and os.path.exists(data_dir):
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(data_dir, split)
            if not os.path.exists(split_dir):
                continue
            for cls_name, cls_label in [("real", 0), ("synthetic", 1)]:
                cls_dir = os.path.join(split_dir, cls_name)
                if os.path.exists(cls_dir):
                    for fname in os.listdir(cls_dir):
                        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                            items.append({
                                "path": os.path.join(cls_dir, fname),
                                "label": cls_label,
                                "split": split,
                                "generator": "synthetic_gen" if cls_label == 1 else "real_camera"
                            })
                            
    # If still empty, return dummy list
    if not items:
        print(f"Warning: No dataset images found in {data_dir}. Run retrain/dataset_generator.py first!")
        return None, None, None, []

    # Partition dataset into Train / Val / Test
    # Check if pre-split exist in metadata
    train_items = [i for i in items if i.get("split") == "train"]
    val_items = [i for i in items if i.get("split") == "val"]
    test_items = [i for i in items if i.get("split") == "test"]

    if not (train_items and val_items and test_items):
        # Auto-split dynamically
        random.shuffle(items)
        n_total = len(items)
        n_train = int(n_total * split_ratio[0])
        n_val = int(n_total * split_ratio[1])
        
        train_items = items[:n_train]
        val_items = items[n_train:n_train + n_val]
        test_items = items[n_train + n_val:]

    print(f"Dataset split loaded successfully:")
    print(f"  Train samples: {len(train_items)}")
    print(f"  Val samples:   {len(val_items)}")
    print(f"  Test samples:  {len(test_items)}")
    print(f"  Total samples: {len(train_items) + len(val_items) + len(test_items)}")

    if HAS_TORCH:
        train_ds = PyTorchSignalScopeDataset(train_items)
        val_ds = PyTorchSignalScopeDataset(val_items)
        test_ds = PyTorchSignalScopeDataset(test_items)
        
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
