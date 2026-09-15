"""
SignalScope Pre-Trained Model & Grad-CAM Heatmap Engine
Loads open-source Hugging Face AI Image Detector models (e.g. dima806/deepfake_vs_real_image_detection)
and extracts spatial saliency Grad-CAM heatmaps.
"""

import os
import io
import time
import base64
import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Global cache for loaded model and processor
from .labels import ai_class_index

_LOADED_MODEL = None
_LOADED_PROCESSOR = None
_LOADED_MODEL_NAME = None


_CANDIDATE_MODELS = [
    "dima806/deepfake_vs_real_image_detection",
]

def get_pretrained_ai_detector(model_name=None):
    """
    Downloads and caches pre-trained HuggingFace AI Image Detector model.
    """
    global _LOADED_MODEL, _LOADED_PROCESSOR, _LOADED_MODEL_NAME
    
    if not (HAS_TORCH and HAS_TRANSFORMERS):
        return None, None
        
    if _LOADED_MODEL is not None:
        return _LOADED_MODEL, _LOADED_PROCESSOR

    models_to_try = [model_name] if model_name else _CANDIDATE_MODELS
    for name in models_to_try:
        if not name:
            continue
        try:
            print(f"Loading pre-trained AI Detector model: '{name}'...")
            processor = AutoImageProcessor.from_pretrained(name, local_files_only=True)
            model = AutoModelForImageClassification.from_pretrained(name, attn_implementation="eager", local_files_only=True)
            model.eval()
            
            _LOADED_MODEL = model
            _LOADED_PROCESSOR = processor
            _LOADED_MODEL_NAME = name
            print(f"Successfully loaded '{name}' from local cache! id2label: {model.config.id2label}")
            return model, processor
        except Exception as e:
            # Try online load once
            try:
                processor = AutoImageProcessor.from_pretrained(name)
                model = AutoModelForImageClassification.from_pretrained(name, attn_implementation="eager")
                model.eval()
                
                _LOADED_MODEL = model
                _LOADED_PROCESSOR = processor
                _LOADED_MODEL_NAME = name
                print(f"Successfully loaded '{name}'! id2label: {model.config.id2label}")
                return model, processor
            except Exception as ex:
                print(f"Notice loading '{name}': {ex}")

    return None, None


def generate_spatial_gradcam(model, inputs, target_class_idx=None):
    """
    Generates spatial Grad-CAM / ViT Self-Attention saliency map from vision model activations.
    Returns normalized 2D numpy array [0.0, 1.0].
    """
    if not HAS_TORCH or model is None:
        return None

    try:
        model_type = getattr(model.config, "model_type", "").lower() if hasattr(model, "config") else ""
        
        # Fast path for Vision Transformers (ViT, Swin, DeiT) using self-attention maps
        if "vit" in model_type or "deit" in model_type or "swin" in model_type:
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)
                if hasattr(outputs, "attentions") and outputs.attentions:
                    # Extract last layer attention map: shape [1, num_heads, seq_len, seq_len]
                    last_attn = outputs.attentions[-1][0]  # [num_heads, seq_len, seq_len]
                    cls_attn = last_attn.mean(dim=0)[0, 1:]  # [num_patches] (skip CLS token)
                    grid_size = int(np.sqrt(cls_attn.shape[0]))
                    if grid_size * grid_size == cls_attn.shape[0]:
                        cam_np = cls_attn.reshape(grid_size, grid_size).cpu().numpy()
                        cam_np = np.maximum(cam_np, 0)
                        if cam_np.max() > 0:
                            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
                        return cam_np

        # Fallback / ConvNet Grad-CAM path
        target_layer = None
        for name, module in model.named_modules():
            if "classifier" not in name and ("layernorm" in name or "encoder" in name or "conv" in name or "stage" in name):
                target_layer = module

        if target_layer is None:
            return None

        activations = []
        gradients = []

        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                activations.append(output[0])
            else:
                activations.append(output)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        h1 = target_layer.register_forward_hook(forward_hook)
        h2 = target_layer.register_full_backward_hook(backward_hook)

        # Forward pass with autograd enabled
        model.zero_grad()
        with torch.enable_grad():
            # Enable gradients on input tensor
            input_tensor = inputs["pixel_values"].detach().requires_grad_(True)
            outputs = model(pixel_values=input_tensor)
            logits = outputs.logits
            
            if target_class_idx is None:
                target_class_idx = logits.argmax(dim=-1).item()

            score = logits[0, target_class_idx]
            score.backward()

        h1.remove()
        h2.remove()

        if not activations or not gradients:
            return None

        act = activations[0].detach()
        grad = gradients[0].detach()

        # Compute weights via global average pooling of gradients
        if grad.ndim == 4:
            weights = torch.mean(grad, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * act, dim=1, keepdim=True)
            cam = F.relu(cam)
        elif grad.ndim == 3: # Vision Transformer tokens [B, N, C]
            cam = torch.mean(act[:, 1:, :], dim=-1) # Skip CLS token
            grid_size = int(np.sqrt(cam.shape[1]))
            if grid_size * grid_size == cam.shape[1]:
                cam = cam.reshape(1, 1, grid_size, grid_size)
            else:
                cam = None

        if cam is not None:
            cam_np = cam.squeeze().cpu().numpy()
            cam_np = np.maximum(cam_np, 0)
            if cam_np.max() > 0:
                cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
            return cam_np
        return None
    except Exception as e:
        print(f"Grad-CAM extraction note: {e}")
        return None


def run_pretrained_inference(pil_img, model_name=None):
    """
    Executes inference using the pre-trained HuggingFace model.
    Returns (confidence_score, is_ai_generated, gradcam_numpy_map).
    """
    model, processor = get_pretrained_ai_detector(model_name)
    if model is None or processor is None:
        return None, None, None

    try:
        inputs = processor(images=pil_img.convert("RGB"), return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
            
        fake_label_idx = ai_class_index(model.config.id2label)
        ai_confidence = float(probs[fake_label_idx].item())
        is_ai = ai_confidence >= 0.50

        # Extract Grad-CAM saliency map
        gradcam_map = generate_spatial_gradcam(model, inputs, target_class_idx=fake_label_idx)

        return ai_confidence, is_ai, gradcam_map
    except Exception as e:
        print(f"Pretrained inference note: {e}")
        return None, None, None
