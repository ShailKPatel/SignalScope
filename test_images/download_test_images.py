"""
Downloads a few CIFAKE test-split samples (REAL = CIFAR-10 photos, FAKE =
Stable Diffusion v1.4) for smoke tests. CIFAR-10 has no person class, so no
identifiable real people are used (scope ethics rule).

Source: Hugging Face mirror of CIFAKE, dragonintelligence/CIFAKE-image-dataset
(test split; FAKE rows first, REAL rows from offset 10000).
"""

import io
import json
import os
import urllib.request
from PIL import Image

API = "https://datasets-server.huggingface.co/rows"
DATASET = "dragonintelligence/CIFAKE-image-dataset"
LABEL_NAMES = {0: "fake", 1: "real"}
OFFSETS = {"fake": 0, "real": 10000}


def _rows(offset, length=20):
    url = f"{API}?dataset={DATASET}&config=default&split=test&offset={offset}&length={length}"
    return json.load(urllib.request.urlopen(url, timeout=30))["rows"]


def setup_test_images(output_dir="test_images", per_class=2):
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    for name, offset in OFFSETS.items():
        rows = [r for r in _rows(offset) if LABEL_NAMES[r["row"]["label"]] == name][:per_class]
        for i, r in enumerate(rows):
            path = os.path.join(output_dir, f"cifake_{name}_{i}.png")
            if not os.path.exists(path):
                data = urllib.request.urlopen(r["row"]["image"]["src"], timeout=30).read()
                Image.open(io.BytesIO(data)).convert("RGB").save(path)
                print(f"Saved CIFAKE {name.upper()} test row {r['row_idx']} -> {path}")
            saved.append(path)
    return saved


if __name__ == "__main__":
    setup_test_images()
