"""
Utility to download a real human photograph from Wikimedia Commons
and generate an AI synthetic test image for live model evaluation.
"""

import os
import urllib.request
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def setup_test_images(output_dir="test_images"):
    os.makedirs(output_dir, exist_ok=True)
    
    real_img_path = os.path.join(output_dir, "real_einstein_photo.jpg")
    synth_img_path = os.path.join(output_dir, "synthetic_ai_generated.png")

    # 1. Download famous real human photograph (Albert Einstein portrait from Wikimedia Commons)
    if not os.path.exists(real_img_path):
        urls = [
            "https://upload.wikimedia.org/wikipedia/commons/d/d3/Albert_Einstein_Head.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Einstein-with-tongue.jpg/400px-Einstein-with-tongue.jpg"
        ]
        for url in urls:
            print(f"Downloading real famous human photograph from {url}...")
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) SignalScope/1.0'}
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req) as resp, open(real_img_path, "wb") as out_f:
                    out_f.write(resp.read())
                print(f"Saved real photograph to: {real_img_path}")
                break
            except Exception as e:
                print(f"Notice downloading {url}: {e}")

    # 2. Generate AI Synthetic Test Image with typical generative artifacts
    if not os.path.exists(synth_img_path):
        print(f"Generating synthetic AI test image...")
        img = Image.new("RGB", (512, 512), color=(100, 150, 200))
        draw = ImageDraw.Draw(img)
        
        # Inject upsampling grid patterns & smooth artificial textures
        for i in range(0, 512, 16):
            draw.line([(i, 0), (i, 512)], fill=(255, 255, 255, 40), width=1)
            draw.line([(0, i), (512, i)], fill=(0, 0, 0, 40), width=1)
            
        # Draw smooth synthetic geometric portrait elements
        draw.ellipse([150, 120, 362, 380], fill=(240, 200, 180))
        draw.polygon([(256, 200), (230, 270), (282, 270)], fill=(210, 160, 140))
        img = img.filter(ImageFilter.GaussianBlur(radius=2.0))
        img.save(synth_img_path)
        print(f"Saved synthetic AI image to: {synth_img_path}")

    return real_img_path, synth_img_path


if __name__ == "__main__":
    setup_test_images()
