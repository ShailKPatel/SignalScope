"""
SignalScope CIFAKE-Layout Smoke-Test Generator
Builds the CIFAKE folder structure (train/test x REAL/FAKE) and seeds it with
procedural 32x32 placeholder images, so train.py and evaluate.py can be dry-run
before the real CIFAKE download is in place.

Real training data: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
"""

import os
import argparse
import random
from PIL import Image, ImageDraw, ImageFilter

CIFAKE_SIZE = (32, 32)


def generate_procedural_sample_image(is_synthetic=False, size=CIFAKE_SIZE):
    """
    Creates a placeholder REAL or FAKE image at CIFAKE resolution.
    """
    img = Image.new("RGB", size, color=(random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)))
    draw = ImageDraw.Draw(img)

    if is_synthetic:
        # Inject a periodic grid, a crude stand-in for upsampling artifacts
        for i in range(0, size[0], 4):
            draw.line([(i, 0), (i, size[1])], fill=(255, 255, 255), width=1)
            draw.line([(0, i), (size[0], i)], fill=(0, 0, 0), width=1)
        draw.ellipse([size[0]*0.2, size[1]*0.2, size[0]*0.8, size[1]*0.8], fill=(random.randint(100, 255), random.randint(50, 180), random.randint(100, 220)))
        img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    else:
        # Sensor-like grain
        for _ in range(10):
            x, y = random.randint(0, size[0]-1), random.randint(0, size[1]-1)
            r, g, b = img.getpixel((x, y))
            draw.point((x, y), fill=(min(255, r+30), min(255, g+30), min(255, b+30)))

    return img


def setup_retraining_environment(data_dir="retrain/data", num_sample_images=120):
    """
    Creates data_dir/{train,test}/{REAL,FAKE} and fills it with placeholder images,
    keeping CIFAKE's 5:1 train:test ratio.
    """
    if os.path.isdir(os.path.join(data_dir, "train", "REAL")) and os.listdir(os.path.join(data_dir, "train", "REAL")):
        print(f"{data_dir}/train/REAL already has images - not mixing placeholders into it.")
        return

    print(f"Seeding CIFAKE-layout placeholder images at {data_dir}...")
    num_test = num_sample_images // 6
    allocations = [("train", num_sample_images - num_test), ("test", num_test)]

    counts = {}
    for split, count in allocations:
        for cls, is_synthetic in (("REAL", False), ("FAKE", True)):
            cls_dir = os.path.join(data_dir, split, cls)
            os.makedirs(cls_dir, exist_ok=True)
            for i in range(count // 2):
                generate_procedural_sample_image(is_synthetic=is_synthetic).save(os.path.join(cls_dir, f"placeholder_{i}.png"))
            counts[f"{split}/{cls}"] = count // 2

    print("Placeholder dataset ready:")
    for key, n in counts.items():
        print(f"  {key:12s} {n}")
    print("Replace these with the real CIFAKE download before training for real.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignalScope CIFAKE-layout smoke-test setup")
    parser.add_argument("--data-dir", default="retrain/data", help="Target CIFAKE root folder")
    parser.add_argument("--num-samples", type=int, default=120, help="Total placeholder images to create")
    args = parser.parse_args()

    setup_retraining_environment(data_dir=args.data_dir, num_sample_images=args.num_samples)
