"""
Creates test images embedded with AI generator metadata for Level 1 verification:
  - Gemini (Imagen 3)
  - Grok 2 (xAI)
  - ComfyUI / Stable Diffusion
  - DALL-E 3 (OpenAI)
  - Real Canon EOS R5 Camera
"""

import os
from PIL import Image, PngImagePlugin


def create_metadata_samples(output_dir="test_images"):
    os.makedirs(output_dir, exist_ok=True)

    samples = [
        {
            "filename": "gemini_imagen3_sample.png",
            "metadata": {"Software": "Google Gemini Imagen 3", "google:digitalSourceType": "trainedAlgorithmicMedia"},
            "color": (120, 180, 240)
        },
        {
            "filename": "grok2_sample.png",
            "metadata": {"Software": "Grok 2 (xAI Image Generator)"},
            "color": (50, 50, 50)
        },
        {
            "filename": "comfyui_sdxl_sample.png",
            "metadata": {
                "parameters": "Masterpiece portrait, highly detailed\nSteps: 30, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 12345, Size: 512x512, Model: sdxl_base_1.0"
            },
            "color": (200, 120, 180)
        },
        {
            "filename": "dalle3_sample.png",
            "metadata": {"Software": "DALL-E 3 (OpenAI)", "prompt": "A futuristic city in sunset"},
            "color": (240, 180, 120)
        }
    ]

    for item in samples:
        fpath = os.path.join(output_dir, item["filename"])
        img = Image.new("RGB", (256, 256), color=item["color"])
        
        png_info = PngImagePlugin.PngInfo()
        for k, v in item["metadata"].items():
            png_info.add_text(k, v)
            
        img.save(fpath, "PNG", pnginfo=png_info)
        print(f"Created metadata sample: {fpath} with keys {list(item['metadata'].keys())}")


if __name__ == "__main__":
    create_metadata_samples()
