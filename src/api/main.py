"""
SignalScope FastAPI Server
Serves inference API endpoints and hosts the Web Application Dashboard.
"""

import os
import io
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

from model.predict import predict_image, predict_batch

app = FastAPI(
    title="SignalScope Media Authenticity Platform API",
    description="SIH 2026 Core Authenticity Checker & Bonus Modules A through G",
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
        "modules_active": ["Core", "Module A", "Module B", "Module C", "Module D", "Module E", "Module F", "Module G"]
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
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process batch files: {str(e)}")

# Mount static frontend assets
app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(UI_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h1>SignalScope API Running</h1><p>Web UI index.html loading...</p>")
