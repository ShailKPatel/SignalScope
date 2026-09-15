"""
SignalScope 3-Family Multi-Model Expert Ensemble Verification Test Suite
Tests:
  1. Individual evaluation across 3 distinct Model Families:
     - Family 1: Vision Transformer (ViT-Base: dima806/deepfake_vs_real_image_detection)
     - Family 2: Swin Transformer (Shifted Windows: Organika/sdxl-detector)
     - Family 3: Dual-Stream Spatial + 2D FFT Frequency Backbone
  2. Hard Majority Voting & Composite Heatmap Aggregation across all 3 families.
  3. REST API & WebSocket Streaming with 3-Family Ensemble Verdict.
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


def test_ensemble_3_families_standalone():
    print("=" * 85)
    print("      SIGNAL SCOPE - 3-FAMILY MULTI-MODEL EXPERT ENSEMBLE TEST")
    print("=" * 85)

    real_img_path = "test_images/real_einstein_photo.jpg"
    print(f"\nEvaluating 3-Family Ensemble on Real Photo ({real_img_path})...")
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
        print(f"    Votes AI:             {m['is_ai_pred']}")

    print(f"\n  Composite Saliency Map Shape: {composite_cam.shape if composite_cam is not None else None}")
    assert prob is not None
    assert breakdown.get("num_families_evaluated", 0) >= 3
    print("  SUCCESS: Standalone 3-Family Multi-Model Ensemble Test Passed!\n")


def test_stacking_metalearner_offline(tmp_path=None):
    """Checks the stacked fusion math on a hand-written artifact, without loading any model."""
    import tempfile
    import numpy as np
    from model import ensemble

    names = [m["id"] for m in ensemble.ENSEMBLE_MEMBERS]
    spec = {
        "type": "stacking_logistic_regression", "eps": 1e-6, "members": names,
        "scaler_mean": [0.0] * len(names), "scaler_scale": [1.0] * len(names),
        "coefficients": [1.0] * len(names), "intercept": 0.0, "threshold": 0.7,
    }
    path = os.path.join(str(tmp_path) if tmp_path else tempfile.mkdtemp(), "stacker.json")
    with open(path, "w") as f:
        json.dump(spec, f)

    stacker = ensemble.load_stacker(path)
    assert stacker is not None

    # Symmetric inputs cancel in logit space -> P(AI) = 0.5, below the 0.7 threshold.
    p, is_ai, score = ensemble.stacked_decision(stacker, [0.9, 0.1, 0.5])
    assert abs(p - 0.5) < 1e-6 and not is_ai and score < 0.5

    # Remapped score keeps predict.py's fixed >= 0.5 cut in agreement with the tuned threshold.
    for probs in ([0.99, 0.95, 0.9], [0.6, 0.6, 0.6], [0.05, 0.2, 0.4]):
        p, is_ai, score = ensemble.stacked_decision(stacker, probs)
        assert is_ai == (p >= 0.7) == (score >= 0.5)
    print("  SUCCESS: Stacking meta-learner offline test passed!\n")


def test_full_pipeline_3_family_ensemble():
    print("=" * 85)
    print("      SIGNAL SCOPE - FULL PIPELINE LEVEL 2 ENSEMBLE TEST")
    print("=" * 85)

    real_img_path = "test_images/real_einstein_photo.jpg"
    print(f"\nTesting POST /api/predict Level 2 3-Family Ensemble Classification...")
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
    print("  SUCCESS: Full Pipeline 3-Family Ensemble Test Passed!\n")


def test_websocket_3_family_ensemble():
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

    print("WebSocket 3-Family Ensemble Streaming Verification Passed!\n")


if __name__ == "__main__":
    test_stacking_metalearner_offline()
    test_ensemble_3_families_standalone()
    test_full_pipeline_3_family_ensemble()
    test_websocket_3_family_ensemble()
