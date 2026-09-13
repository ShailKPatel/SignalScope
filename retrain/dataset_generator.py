"""
SignalScope Dataset Generator & Manifest Indexing Utility
Builds data folder structures, seeds sample datasets, and generates 100,000 image metadata index manifests.
"""

import os
import json
import argparse
import random
from PIL import Image, ImageDraw, ImageFilter


def generate_procedural_sample_image(is_synthetic=False, generator="sdxl", size=(224, 224)):
    """
    Creates a synthetic or real procedural image sample.
    """
    img = Image.new("RGB", size, color=(random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)))
    draw = ImageDraw.Draw(img)
    
    if is_synthetic:
        # Inject synthetic frequency grid pattern / artifacts
        for i in range(0, size[0], 16):
            draw.line([(i, 0), (i, size[1])], fill=(255, 255, 255, 30), width=1)
            draw.line([(0, i), (size[0], i)], fill=(0, 0, 0, 30), width=1)
        # Add smooth blurry shapes
        draw.ellipse([size[0]*0.2, size[1]*0.2, size[0]*0.8, size[1]*0.8], fill=(random.randint(100, 255), random.randint(50, 180), random.randint(100, 220)))
        img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    else:
        # Real image look: Natural noise & camera grain
        for _ in range(50):
            x, y = random.randint(0, size[0]-1), random.randint(0, size[1]-1)
            r, g, b = img.getpixel((x, y))
            draw.point((x, y), fill=(min(255, r+30), min(255, g+30), min(255, b+30)))
            
    return img


def setup_retraining_environment(data_dir="retrain/data", manifest_path="retrain/data/manifest.json", num_sample_images=100, total_target_images=100000):
    """
    Constructs the retraining dataset directory structure and manifest index.
    """
    print(f"Setting up Retraining Environment at {data_dir}...")
    
    splits = ["train", "val", "test"]
    classes = ["real", "synthetic"]
    
    for split in splits:
        for cls in classes:
            dir_path = os.path.join(data_dir, split, cls)
            os.makedirs(dir_path, exist_ok=True)

    generators = ["stable_diffusion_v1_5", "sdxl", "midjourney_v6", "flux_1", "stylegan3"]
    
    # 1. Generate Sample Physical Files
    samples_created = 0
    split_counts = {"train": 0, "val": 0, "test": 0}
    
    # 80/10/10 split for sample physical images
    num_train = int(num_sample_images * 0.8)
    num_val = int(num_sample_images * 0.1)
    num_test = num_sample_images - num_train - num_val
    
    sample_allocations = [
        ("train", num_train),
        ("val", num_val),
        ("test", num_test)
    ]
    
    manifest_samples = []
    
    for split, count in sample_allocations:
        half = count // 2
        
        # Real images
        for i in range(half):
            fname = f"real_{split}_{i}.png"
            fpath = os.path.join(data_dir, split, "real", fname)
            img = generate_procedural_sample_image(is_synthetic=False)
            img.save(fpath)
            samples_created += 1
            manifest_samples.append({
                "id": f"real_{split}_{i}",
                "path": fpath,
                "label": 0,
                "class_name": "real",
                "split": split,
                "generator": "camera_sensor"
            })
            
        # Synthetic images
        for i in range(half):
            gen = random.choice(generators)
            fname = f"synth_{split}_{i}.png"
            fpath = os.path.join(data_dir, split, "synthetic", fname)
            img = generate_procedural_sample_image(is_synthetic=True, generator=gen)
            img.save(fpath)
            samples_created += 1
            manifest_samples.append({
                "id": f"synth_{split}_{i}",
                "path": fpath,
                "label": 1,
                "class_name": "synthetic",
                "split": split,
                "generator": gen
            })
            
    # 2. Build 100,000 Image Metadata Manifest Index
    print(f"Generating 100,000 Image Dataset Manifest Index ({total_target_images} items)...")
    
    total_manifest_items = []
    # Add physical sample images first
    total_manifest_items.extend(manifest_samples)
    
    # Fill virtual index up to total_target_images for full scale 100,000 data pipeline testing
    remaining = total_target_images - len(total_manifest_items)
    
    n_train_rem = int(remaining * 0.8)
    n_val_rem = int(remaining * 0.1)
    n_test_rem = remaining - n_train_rem - n_val_rem
    
    allocs = [("train", n_train_rem), ("val", n_val_rem), ("test", n_test_rem)]
    
    idx_counter = len(total_manifest_items)
    for split, count in allocs:
        half = count // 2
        for i in range(half):
            idx_counter += 1
            total_manifest_items.append({
                "id": f"idx_{idx_counter}",
                "path": os.path.join(data_dir, split, "real", f"virtual_real_{i}.png"),
                "label": 0,
                "class_name": "real",
                "split": split,
                "generator": "camera_sensor"
            })
            idx_counter += 1
            gen = random.choice(generators)
            total_manifest_items.append({
                "id": f"idx_{idx_counter}",
                "path": os.path.join(data_dir, split, "synthetic", f"virtual_synth_{i}.png"),
                "label": 1,
                "class_name": "synthetic",
                "split": split,
                "generator": gen
            })
            
    manifest_data = {
        "dataset_name": "SignalScope-100k-Master",
        "total_count": len(total_manifest_items),
        "split_ratios": {"train": 0.80, "val": 0.10, "test": 0.10},
        "counts": {
            "train": sum(1 for x in total_manifest_items if x["split"] == "train"),
            "val": sum(1 for x in total_manifest_items if x["split"] == "val"),
            "test": sum(1 for x in total_manifest_items if x["split"] == "test"),
            "real": sum(1 for x in total_manifest_items if x["label"] == 0),
            "synthetic": sum(1 for x in total_manifest_items if x["label"] == 1)
        },
        "samples": total_manifest_items
    }
    
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
        
    print("Environment & Dataset Manifest initialization complete!")
    print(f"  Physical sample images saved: {samples_created}")
    print(f"  Total manifest index records: {manifest_data['total_count']}")
    print(f"  Train / Val / Test split:     {manifest_data['counts']['train']} / {manifest_data['counts']['val']} / {manifest_data['counts']['test']}")
    print(f"  Real / Synthetic split:       {manifest_data['counts']['real']} / {manifest_data['counts']['synthetic']}")
    print(f"  Manifest written to:          {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignalScope Dataset Setup")
    parser.add_argument("--data-dir", default="retrain/data", help="Target data folder")
    parser.add_argument("--manifest-path", default="retrain/data/manifest.json", help="Target manifest JSON path")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of physical sample images to build")
    parser.add_argument("--target-total", type=int, default=100000, help="Total dataset index target size")
    args = parser.parse_args()
    
    setup_retraining_environment(
        data_dir=args.data_dir,
        manifest_path=args.manifest_path,
        num_sample_images=args.num_samples,
        total_target_images=args.target_total
    )
