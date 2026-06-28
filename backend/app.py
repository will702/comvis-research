"""
FastAPI app — serves the playground API + the built frontend (StaticFiles).
"""
import os
import sys
import glob
import base64
import json
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from backend import models as model_store
from backend.pipeline import run as run_pipeline
from backend.params import PipelineParams, DEFAULT_PARAMS
from backend.encode import to_png_b64

app = FastAPI(title="Acne CV Playground")

# Allow Vite dev server on localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
model_store.load_models()

# ── Demo image index ──────────────────────────────────────────────────────────
_DEMO_ROOT = os.path.join(_ROOT, "data_split_6535", "test")
_demo_index: list[dict] = []

def _build_demo_index():
    global _demo_index
    for cls_dir in sorted(glob.glob(os.path.join(_DEMO_ROOT, "acne*_1024"))):
        cls_name = os.path.basename(cls_dir).replace("_1024", "")  # acne1 / acne2 / acne3
        imgs = sorted(glob.glob(os.path.join(cls_dir, "*.jpg")))[:8]
        for p in imgs:
            demo_id = str(uuid.uuid5(uuid.NAMESPACE_URL, p))
            thumb_bgr = cv2.imread(p)
            if thumb_bgr is None:
                continue
            thumb = cv2.resize(thumb_bgr, (96, 96))
            _demo_index.append({
                "id":    demo_id,
                "path":  p,
                "label": cls_name,
                "thumb": to_png_b64(thumb),
            })

_build_demo_index()
_demo_by_id = {d["id"]: d for d in _demo_index}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/demos")
async def get_demos():
    """Return demo gallery — id, label, thumb b64."""
    return [{"id": d["id"], "label": d["label"], "thumb": d["thumb"]} for d in _demo_index]


@app.post("/api/run")
async def run_inference(
    demo_id: str | None = Form(None),
    params_json: str     = Form("{}"),
    file: UploadFile     = File(None),
):
    """
    Run the traced pipeline + all classifiers on an image.
    Accepts either a demo_id (picks from gallery) or an uploaded file.
    Image is processed in-memory, never written to disk.
    """
    # Parse params
    try:
        params_dict = json.loads(params_json)
        params = PipelineParams.from_dict({**DEFAULT_PARAMS.to_dict(), **params_dict})
    except Exception as e:
        raise HTTPException(400, f"Invalid params: {e}")

    # Resolve image
    if file is not None:
        raw = await file.read()
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "Could not decode uploaded image")
    elif demo_id:
        demo = _demo_by_id.get(demo_id)
        if not demo:
            raise HTTPException(404, f"Demo id {demo_id!r} not found")
        img = cv2.imread(demo["path"])
        if img is None:
            raise HTTPException(500, "Could not read demo image")
    else:
        raise HTTPException(400, "Provide demo_id or upload a file")

    # Run pipeline
    try:
        trace = run_pipeline(img, params)
    except Exception as e:
        raise HTTPException(500, f"Pipeline error: {e}")

    # Run classifiers
    vec = np.array(trace["feature_vector"])
    predictions = model_store.predict_all(vec)

    return {
        "trace": trace,
        "predictions": predictions,
        "params": params.to_dict(),
    }


@app.get("/api/health")
async def health():
    return {"ok": True, "models_loaded": len(model_store._models)}


# ── Serve built frontend ───────────────────────────────────────────────────────
_DIST = os.path.join(_ROOT, "frontend", "dist")
if os.path.isdir(_DIST):
    from fastapi.responses import FileResponse
    app.mount("/assets", StaticFiles(directory=os.path.join(_DIST, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def spa(full_path: str):
        index = os.path.join(_DIST, "index.html")
        return FileResponse(index)
