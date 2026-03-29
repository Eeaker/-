"""BallShow backend entrypoint."""

from __future__ import annotations

import os
import sys

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from starlette.responses import Response

# Force Ultralytics to use a writable config dir inside workspace.
if not os.getenv("YOLO_CONFIG_DIR"):
    _yolo_cfg_root = os.path.join(os.path.dirname(__file__), ".ultralytics_cfg")
    os.makedirs(_yolo_cfg_root, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = _yolo_cfg_root

PLATFORM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PLATFORM_DIR not in sys.path:
    sys.path.insert(0, PLATFORM_DIR)

TRANSREID_ROOT = os.path.abspath(os.path.join(PLATFORM_DIR, ".."))
if TRANSREID_ROOT not in sys.path:
    sys.path.insert(0, TRANSREID_ROOT)

from backend.database import init_db
from backend.reid_engine import init_engine
from backend.runtime_paths import UPLOAD_DIR, ensure_runtime_dirs
from backend.routes.auth_routes import router as auth_router
from backend.routes.dashboard_routes import router as dashboard_router
from backend.routes.game_analysis_routes import router as game_analysis_router
from backend.routes.reid_routes import router as reid_router
from backend.routes.trajectory_routes import router as trajectory_router
from backend.routes.video_routes import router as video_router


CONFIG_PATH = os.path.join(TRANSREID_ROOT, "configs", "BallShow", "vit_transreid_4090.yml")
DEFAULT_REID_WEIGHT_PATH = os.path.join(TRANSREID_ROOT, "logs", "92.6", "transformer_120.pth")
_weight_hint = os.getenv("REID_SINGLE_WEIGHT_PATH", DEFAULT_REID_WEIGHT_PATH).strip()
REID_WEIGHT_PATH = _weight_hint if os.path.isabs(_weight_hint) else os.path.abspath(
    os.path.join(TRANSREID_ROOT, _weight_hint)
)
GALLERY_DIR = os.path.join(TRANSREID_ROOT, "..", "data", "BallShow", "bounding_box_test")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(
    title="BallShow Basketball Intelligence Platform",
    description="Single-model TransReID + game analysis workflows",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1024)

app.include_router(auth_router)
app.include_router(reid_router)
app.include_router(video_router)
app.include_router(trajectory_router)
app.include_router(game_analysis_router)
app.include_router(dashboard_router)


@app.middleware("http")
async def disable_frontend_cache(request: Request, call_next):
    response: Response = await call_next(request)
    path = request.url.path or ""
    if path == "/" or path.startswith("/js/") or path.startswith("/css/") or path.endswith(".html"):
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.get("/api/health")
async def health_check() -> dict:
    return {
        "status": "ok",
        "platform": "BallShow Basketball Intelligence Platform v2.0",
        "reid_mode": "single_model",
        "reid_weight": REID_WEIGHT_PATH,
    }


FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
ensure_runtime_dirs()

if os.path.isdir(GALLERY_DIR):
    app.mount("/gallery", StaticFiles(directory=GALLERY_DIR), name="gallery")

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


@app.on_event("startup")
async def startup() -> None:
    print("=" * 68)
    print(" BallShow Basketball Intelligence Platform v2.0")
    print(" Powered by TransReID Single-Model + Unified Game Analysis")
    print("=" * 68)

    init_db()
    print("[DB] Initialized")

    if not os.path.exists(REID_WEIGHT_PATH):
        raise FileNotFoundError(f"Single-model weight not found: {REID_WEIGHT_PATH}")
    init_engine(CONFIG_PATH, REID_WEIGHT_PATH, GALLERY_DIR, DEVICE)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
