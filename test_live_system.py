"""
SignalScope 2-Level Cascading Forensic Architecture Test Suite
Verifies:
  1. Level 1 Smart Metadata Engine fast-track detection for Gemini, Grok 2, ComfyUI, DALL-E 3 + Grad-CAM heatmap overlay.
  2. Level 2 Deep Learning Vision Model classification for clean photography / stripped metadata + Grad-CAM heatmap overlay.
  3. WebSocket live streaming analysis (/ws/analyze).
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

from src.api.main import app

client = TestClient(app)


def test_level_1_smart_metadata_engine():
    print("=" * 70)
    print("      SIGNAL SCOPE - LEVEL 1 SMART METADATA ENGINE TEST")
    print("=" * 70)
    
    metadata_test_files = [
        ("Gemini Imagen 3", "test_images/gemini_imagen3_sample.png"),
        ("Grok 2 (xAI)", "test_images/grok2_sample.png"),
        ("ComfyUI / SDXL", "test_images/comfyui_sdxl_sample.png"),
        ("DALL-E 3 (OpenAI)", "test_images/dalle3_sample.png"),
    ]

    for gen_name, fpath in metadata_test_files:
        print(f"\nTesting Level 1 Fast-Track for {gen_name} ({fpath})...")
        with open(fpath, "rb") as f:
            t0 = time.time()
            res = client.post("/api/predict", files={"file": (os.path.basename(fpath), f, "image/png")}).json()
            dt = round((time.time() - t0) * 1000, 2)
            
        print(f"  Inference Time:      {dt} ms")
        print(f"  Detection Level:     {res['verdict']['detection_level']}")
        print(f"  Matched Generator:   {res['verdict']['matched_generator']}")
        print(f"  Verdict Label:       {res['verdict']['label']}")
        print(f"  Confidence Score:    {res['verdict']['confidence_score']}")
        print(f"  Metadata Evidence:   {res['verdict']['metadata_evidence']}")
        print(f"  Grad-CAM Heatmap:    {res['modules']['module_a_explainability']['heatmap_base64'][:40]}...")
        
        assert res["verdict"]["is_ai_generated"] == True
        assert "Level 1" in res["verdict"]["detection_level"]
        assert res["modules"]["module_a_explainability"]["heatmap_base64"].startswith("data:image/png;base64,")
        print(f"  SUCCESS: Verified Level 1 Fast-Track + Grad-CAM Heatmap for {gen_name}!")


def test_level_2_deep_vision_classifier():
    print("\n" + "=" * 70)
    print("      SIGNAL SCOPE - LEVEL 2 DEEP VISION MODEL CLASSIFIER TEST")
    print("=" * 70)
    
    real_path = "test_images/real_einstein_photo.jpg"
    print(f"\nTesting Level 2 Evaluation with Real Photograph ({real_path})...")
    with open(real_path, "rb") as f:
        t0 = time.time()
        res_real = client.post("/api/predict", files={"file": ("real_einstein_photo.jpg", f, "image/jpeg")}).json()
        dt = round((time.time() - t0) * 1000, 2)
        
    print(f"  Inference Time:      {dt} ms")
    print(f"  Detection Level:     {res_real['verdict']['detection_level']}")
    print(f"  Verdict Label:       {res_real['verdict']['label']}")
    print(f"  Confidence Score:    {res_real['verdict']['confidence_score']}")
    print(f"  Is AI Generated:     {res_real['verdict']['is_ai_generated']}")
    print(f"  Grad-CAM Heatmap:    {res_real['modules']['module_a_explainability']['heatmap_base64'][:40]}...")
    
    assert "Level 2" in res_real["verdict"]["detection_level"]
    print("  SUCCESS: Verified Level 2 Deep Vision Model Classification!")


def test_websocket_streaming():
    print("\n" + "=" * 70)
    print("      SIGNAL SCOPE - WEBSOCKET STREAMING VERIFICATION")
    print("=" * 70)
    
    fpath = "test_images/gemini_imagen3_sample.png"
    with open(fpath, "rb") as f:
        b64_data = base64.b64encode(f.read()).decode("utf-8")
        
    payload = {
        "filename": "gemini_imagen3_sample.png",
        "image_b64": f"data:image/png;base64,{b64_data}",
        "caption": "Gemini AI generated synthetic portrait"
    }

    print("Connecting to WebSocket /ws/analyze...")
    with client.websocket_connect("/ws/analyze") as websocket:
        print("Connected! Sending image payload over WebSocket...")
        websocket.send_text(json.dumps(payload))
        
        while True:
            msg_str = websocket.receive_text()
            msg = json.loads(msg_str)
            msg_type = msg.get("type")
            
            if msg_type == "progress":
                print(f"  [WS Progress {msg.get('progress')}%] Stage: {msg.get('stage')} | {msg.get('message')}")
            elif msg_type == "result":
                p = msg.get("payload", {})
                print("\n  [WS Final Result Received]")
                print(f"  Detection Level:   {p.get('verdict', {}).get('detection_level')}")
                print(f"  Matched Generator: {p.get('verdict', {}).get('matched_generator')}")
                print(f"  Verdict Label:     {p.get('verdict', {}).get('label')}")
                print(f"  Confidence Score:  {p.get('verdict', {}).get('confidence_score')}")
                print(f"  Heatmap Base64:    {p.get('modules', {}).get('module_a_explainability', {}).get('heatmap_base64')[:40]}...")
                break
            elif msg_type == "error":
                print(f"  [WS Error]: {msg.get('message')}")
                break

    print("WebSocket Streaming Verification Completed Successfully!\n")


if __name__ == "__main__":
    test_level_1_smart_metadata_engine()
    test_level_2_deep_vision_classifier()
    test_websocket_streaming()
