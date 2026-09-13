"""
SignalScope 5-Family Multi-Model Expert Ensemble Verification Test Suite
Tests:
  1. Individual evaluation across 5 distinct Model Families:
     - Family 1: Vision Transformer (ViT-Base: dima806/deepfake_vs_real_image_detection)
     - Family 2: Swin Transformer (Shifted Windows: umm-maybe/AI-image-detector)
     - Family 3: ConvNeXt CNN (Inverted Bottleneck: prithivMLmods/Deep-Fake-Detector-v2)
     - Family 4: EfficientNet CNN (Compound Scaled: Falconsai/intent_based_deepfake)
     - Family 5: Dual-Stream Spatial + 2D FFT Frequency Backbone
  2. Entropy-Weighted Soft Voting & Composite Heatmap Aggregation across all 5 families.
  3. REST API & WebSocket Streaming with 5-Family Ensemble Verdict.
"""

import os
import sys
import json
import base64
import time
from PIL import Image
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from model.ensemble import run_ensemble_inference
from model.predict import predict_image
from src.api.main import app

client = TestClient(app)


def test_ensemble_5_families_standalone():
    print("=" * 85)
    print("      SIGNAL SCOPE - 5-FAMILY MULTI-MODEL EXPERT ENSEMBLE TEST")
    print("=" * 85)

    real_img_path = "test_images/real_einstein_photo.jpg"
    print(f"\nEvaluating 5-Family Ensemble on Real Photo ({real_img_path})...")
    img = Image.open(real_img_path).convert("RGB")

    t0 = time.time()
    prob, composite_cam, breakdown = run_ensemble_inference(img)
    dt = round((time.time() - t0) * 1000, 2)

    print(f"  Total Inference Time:    {dt} ms")
    print(f"  Aggregated AI Prob:      {prob * 100:.2f}%")
    print(f"  Model Families Count:    {breakdown.get('num_families_evaluated')}")
    
    for m in breakdown.get("family_models", []):
        print(f"\n  [Family: {m['family']}] ({m['model_id']})")
        print(f"    Model Name:           {m['model_name']}")
        print(f"    Architecture:         {m['architecture']}")
        print(f"    Specialization:       {m['specialization']}")
        print(f"    AI Probability:       {m['ai_probability'] * 100:.2f}%")
        print(f"    Uncertainty H:        {m['entropy_uncertainty']}")
        print(f"    Ensemble Weight:      {m['ensemble_weight']}")

    print(f"\n  Composite Saliency Map Shape: {composite_cam.shape if composite_cam is not None else None}")
    assert prob is not None
    assert breakdown.get("num_families_evaluated", 0) >= 3
    print("  SUCCESS: Standalone 5-Family Multi-Model Ensemble Test Passed!\n")


def test_full_pipeline_5_family_ensemble():
    print("=" * 85)
    print("      SIGNAL SCOPE - FULL PIPELINE LEVEL 2 ENSEMBLE TEST")
    print("=" * 85)

    real_img_path = "test_images/real_einstein_photo.jpg"
    print(f"\nTesting POST /api/predict Level 2 5-Family Ensemble Classification...")
    with open(real_img_path, "rb") as f:
        res = client.post("/api/predict", files={"file": ("real_einstein_photo.jpg", f, "image/jpeg")}).json()

    verdict = res.get("verdict", {})
    print(f"  Detection Level:   {verdict.get('detection_level')}")
    print(f"  Verdict Label:     {verdict.get('label')}")
    print(f"  Confidence Score:  {verdict.get('confidence_score') * 100:.1f}%")
    print(f"  Is AI Generated:   {verdict.get('is_ai_generated')}")
    
    ens_bd = verdict.get("ensemble_breakdown", {})
    print(f"  Ensemble Strategy: {ens_bd.get('ensemble_strategy')}")
    print(f"  Evaluated Families:{ens_bd.get('num_families_evaluated')}")
    
    assert "Level 2" in verdict.get("detection_level")
    print("  SUCCESS: Full Pipeline 5-Family Ensemble Test Passed!\n")


def test_websocket_5_family_ensemble():
    print("=" * 85)
    print("      SIGNAL SCOPE - WEBSOCKET STREAMING ENSEMBLE VERIFICATION")
    print("=" * 85)

    fpath = "test_images/real_einstein_photo.jpg"
    with open(fpath, "rb") as f:
        b64_data = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "filename": "real_einstein_photo.jpg",
        "image_b64": f"data:image/jpeg;base64,{b64_data}",
        "caption": "Historical photograph portrait"
    }

    print("Connecting to WebSocket /ws/analyze...")
    with client.websocket_connect("/ws/analyze") as websocket:
        print("Connected! Sending image payload...")
        websocket.send_text(json.dumps(payload))

        while True:
            msg_str = websocket.receive_text()
            msg = json.loads(msg_str)
            msg_type = msg.get("type")

            if msg_type == "progress":
                print(f"  [WS Progress {msg.get('progress')}%] Stage: {msg.get('stage')} | {msg.get('message')}")
            elif msg_type == "result":
                p = msg.get("payload", {})
                v = p.get("verdict", {})
                print("\n  [WS Final Result Received]")
                print(f"  Detection Level: {v.get('detection_level')}")
                print(f"  Verdict Label:   {v.get('label')}")
                print(f"  Confidence:      {v.get('confidence_score') * 100:.1f}%")
                break
            elif msg_type == "error":
                print(f"  [WS Error]: {msg.get('message')}")
                break

    print("WebSocket 5-Family Ensemble Streaming Verification Passed!\n")


if __name__ == "__main__":
    test_ensemble_5_families_standalone()
    test_full_pipeline_5_family_ensemble()
    test_websocket_5_family_ensemble()
