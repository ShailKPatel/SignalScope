"""
SignalScope FastAPI Server
Serves inference REST API endpoints, WebSockets for live streaming analysis,
and hosts the Web Application Dashboard.
"""

import os
import io
import json
import base64
import asyncio
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

from model.predict import predict_image, predict_batch, DetectorUnavailableError

app = FastAPI(
    title="SignalScope Media Authenticity Platform API",
    description="SIH 2026 Core Authenticity Checker with Bonus Modules A, C, D, F",
    version="1.0.0"
)

# Enable CORS for external API consumers / extensions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure app static directory exists
UI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app")
os.makedirs(UI_DIR, exist_ok=True)


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "service": "SignalScope Engine",
        "version": "1.0.0",
        "modules_active": ["Core", "Module A", "Module C", "Module D", "Module F"],
        "modules_not_built": ["Module B", "Module E", "Module G"],
        "websocket_endpoint": "/ws/analyze"
    }


@app.post("/api/predict")
async def api_predict(
    file: UploadFile = File(...),
    caption: Optional[str] = Form(None)
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        result = predict_image(image, caption_text=caption, filename=file.filename)
        return JSONResponse(content=result)
    except DetectorUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")


@app.post("/api/predict-batch")
async def api_predict_batch(
    files: List[UploadFile] = File(...)
):
    try:
        inputs = []
        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            inputs.append((image, None, file.filename))
        
        batch_result = predict_batch(inputs)
        return JSONResponse(content=batch_result)
    except DetectorUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process batch files: {str(e)}")


@app.websocket("/ws/analyze")
async def websocket_analyze(websocket: WebSocket):
    """
    WebSocket endpoint for real-time live image analysis & progress streaming.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            
            b64_str = payload.get("image_b64") or payload.get("image")
            filename = payload.get("filename", "uploaded_image.png")
            caption = payload.get("caption")

            if not b64_str:
                await websocket.send_json({"type": "error", "message": "No image payload provided"})
                continue

            # Strip data URL header if present
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]

            # Stage 1: Decoding
            await websocket.send_json({"type": "progress", "stage": "Ingestion", "message": "Decoding image & verifying header...", "progress": 15})
            img_bytes = base64.b64decode(b64_str)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # Stage 2: Deep Learning Inference
            await websocket.send_json({"type": "progress", "stage": "Inference Engine", "message": "Running Vision Transformer / Dual-Stream Model...", "progress": 45})
            await asyncio.sleep(0.1)

            # Stage 3: Bonus Suite & Grad-CAM Heatmap
            await websocket.send_json({"type": "progress", "stage": "Forensic Suite", "message": "Building saliency overlay, EXIF provenance & degradation re-scores...", "progress": 80})
            
            # Predict
            result = predict_image(pil_img, caption_text=caption, filename=filename)

            # Stage 4: Result Delivery
            await websocket.send_json({"type": "progress", "stage": "Complete", "message": "Authenticity analysis finalized.", "progress": 100})
            await websocket.send_json({"type": "result", "filename": filename, "payload": result})

    except WebSocketDisconnect:
        print("WebSocket client disconnected.")
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})


# Mount static frontend assets
app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(UI_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>SignalScope API Running</h1><p>Web UI index.html loading...</p>")
