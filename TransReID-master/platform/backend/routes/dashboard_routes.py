"""Dashboard routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..auth import get_current_user
from ..database import get_dashboard_stats
from ..reid_engine import get_engine

router = APIRouter(prefix="/api", tags=["Dashboard"])


@router.get("/dashboard")
async def dashboard(user: dict = Depends(get_current_user)) -> dict:
    stats = get_dashboard_stats()
    engine = get_engine()

    try:
        import torch

        gpu_online = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if gpu_online else "N/A"
    except Exception:
        gpu_online = False
        gpu_name = "N/A"

    gallery_size = len(engine.gallery_paths) if engine.gallery_paths else 0
    feat_dim = int(engine.gallery_feats.shape[1]) if engine.gallery_feats is not None and engine.gallery_feats.size else 0

    model_info = {
        "mode": "single_model",
        "model": "TransReID ViT-Base/16",
        "weight_path": getattr(engine, "weight_path", ""),
        "feat_dim": feat_dim,
        "device": "CUDA (GPU)" if gpu_online else "CPU",
        "gallery_size": gallery_size,
        "architecture": "TransReID ViT-Base/16",
        "inference_device": "CUDA (GPU)" if gpu_online else "CPU",
        "feature_dim": feat_dim,
        "map_score": "N/A",
        "rank1_accuracy": "N/A",
        # Compatibility placeholders for legacy frontends.
        "model1": "TransReID ViT-Base/16",
        "model2": None,
    }

    stats["gpu_status"] = "Online" if gpu_online else "Offline"
    stats["gpu_name"] = gpu_name
    stats["model_info"] = model_info
    stats["rank1"] = "N/A"
    stats["map"] = "N/A"
    return stats
