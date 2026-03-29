"""ReID engine for BallShow.

Single-model TransReID inference with gallery indexing and top-k search.
"""

from __future__ import annotations

import glob
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Ensure TransReID root is importable.
TRANSREID_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if TRANSREID_ROOT not in sys.path:
    sys.path.insert(0, TRANSREID_ROOT)

from config import cfg  # type: ignore
from model import make_model  # type: ignore

_engine: Optional["ReIDEngine"] = None


def _extract_state_dict(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            sub = payload.get(key)
            if isinstance(sub, dict):
                return sub
        return payload
    raise TypeError(f"Unexpected checkpoint payload type: {type(payload)!r}")


def _load_param_flexible(model: torch.nn.Module, weight_path: str) -> None:
    """Load checkpoint while skipping incompatible layer shapes."""
    raw = torch.load(weight_path, map_location="cpu")
    state_dict = _extract_state_dict(raw)
    cleaned = {str(k).replace("module.", ""): v for k, v in state_dict.items()}

    model_state = model.state_dict()
    loaded = 0
    skipped = 0
    for k, v in cleaned.items():
        if k in model_state and tuple(model_state[k].shape) == tuple(v.shape):
            model_state[k].copy_(v)
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(model_state, strict=False)
    print(f"[ReIDEngine] Flexible checkpoint load: loaded={loaded}, skipped={skipped}")


class ReIDEngine:
    """Single-model ReID inference engine."""

    def __init__(self, config_path: str, weight_path: str, gallery_dir: str, device: str = "cuda"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"ReID config not found: {config_path}")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"ReID weight not found: {weight_path}")

        self.device = self._resolve_device(device)
        self.gallery_dir = gallery_dir
        self.gallery_paths: List[str] = []
        self.gallery_feats: Optional[np.ndarray] = None
        self.weight_path = os.path.abspath(weight_path)

        cfg.merge_from_file(config_path)
        cfg.freeze()
        self.cfg = cfg

        size_test = list(getattr(cfg.INPUT, "SIZE_TEST", [384, 128]))
        if len(size_test) != 2:
            size_test = [384, 128]
        self.input_size = (int(size_test[0]), int(size_test[1]))
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        num_classes = int(os.getenv("REID_NUM_CLASSES", "3353"))
        camera_num = int(os.getenv("REID_CAMERA_NUM", "5"))
        view_num = int(os.getenv("REID_VIEW_NUM", "0"))

        model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
        try:
            model.load_param(weight_path)
        except Exception as exc:
            print(f"[ReIDEngine] Strict checkpoint load failed, fallback to flexible mode: {exc}")
            _load_param_flexible(model, weight_path)
        model.eval()
        self.model = model.to(self.device)

        if os.path.isdir(self.gallery_dir):
            self._build_gallery()

    @staticmethod
    def _resolve_device(device: str) -> str:
        want_cuda = str(device).lower().startswith("cuda")
        if want_cuda and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @torch.no_grad()
    def extract_feature(self, pil_image: Image.Image) -> np.ndarray:
        img_tensor = self.transform(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        cam_label = torch.zeros(1, dtype=torch.long, device=self.device)
        view_label = torch.zeros(1, dtype=torch.long, device=self.device)
        feat = self.model(img_tensor, cam_label=cam_label, view_label=view_label)
        feat = F.normalize(feat, p=2, dim=1)
        return feat.detach().cpu().numpy().reshape(-1)

    @torch.no_grad()
    def _build_gallery(self) -> None:
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        paths: List[str] = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(self.gallery_dir, ext)))
        paths.sort()
        self.gallery_paths = paths

        if not paths:
            self.gallery_feats = np.zeros((0, 1), dtype=np.float32)
            return

        batch_size = int(os.getenv("REID_GALLERY_BATCH", "64"))
        all_feats: List[np.ndarray] = []

        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i : i + batch_size]
            batch_tensors: List[torch.Tensor] = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    batch_tensors.append(self.transform(img))
                except Exception:
                    batch_tensors.append(torch.zeros(3, self.input_size[0], self.input_size[1]))

            batch = torch.stack(batch_tensors).to(self.device)
            cam_labels = torch.zeros(batch.size(0), dtype=torch.long, device=self.device)
            view_labels = torch.zeros(batch.size(0), dtype=torch.long, device=self.device)
            feats = self.model(batch, cam_label=cam_labels, view_label=view_labels)
            feats = F.normalize(feats, p=2, dim=1)
            all_feats.append(feats.detach().cpu().numpy())

        self.gallery_feats = np.vstack(all_feats).astype(np.float32, copy=False)
        print(
            f"[ReIDEngine] Gallery ready: {len(self.gallery_paths)} images, "
            f"feat_dim={self.gallery_feats.shape[1]}, model={self.weight_path}"
        )

    def search(self, query_feat: np.ndarray, topk: int = 10) -> List[Dict[str, Any]]:
        if self.gallery_feats is None or self.gallery_feats.shape[0] == 0:
            return []

        query = query_feat.astype(np.float32, copy=False)
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        gallery_norm = self.gallery_feats / (np.linalg.norm(self.gallery_feats, axis=1, keepdims=True) + 1e-8)
        scores = gallery_norm @ query_norm

        k = max(1, min(int(topk), int(scores.shape[0])))
        top_indices = np.argsort(-scores)[:k]

        out: List[Dict[str, Any]] = []
        for rank, idx in enumerate(top_indices, start=1):
            filename = os.path.basename(self.gallery_paths[int(idx)])
            person_id = filename.split("_")[0] if "_" in filename else "unknown"
            out.append(
                {
                    "rank": rank,
                    "filename": filename,
                    "person_id": person_id,
                    "score": round(float(scores[int(idx)]), 4),
                    "path": f"/gallery/{filename}",
                }
            )
        return out


def get_engine() -> ReIDEngine:
    global _engine
    if _engine is None:
        raise RuntimeError("ReID engine is not initialized. Call init_engine() first.")
    return _engine


def init_engine(
    config_path: str,
    weight_path: str,
    gallery_dir: Optional[str] = None,
    device: str = "cuda",
    *extra: Any,
) -> ReIDEngine:
    """Initialize global ReID engine.

    Notes:
    - Preferred signature: init_engine(config_path, weight_path, gallery_dir, device)
    - Backward compatibility for legacy 5-arg call:
      init_engine(config_path, weight1_path, weight2_path, gallery_dir, device)
      where weight2_path is ignored.
    """
    global _engine

    # Legacy call adapter.
    if gallery_dir and str(gallery_dir).lower().endswith(".pth"):
        legacy_weight2 = str(gallery_dir)
        gallery_dir = str(device)
        device = str(extra[0]) if extra else "cuda"
        print(f"[ReIDEngine] Legacy dual-weight init detected; ignoring weight2: {legacy_weight2}")

    if not gallery_dir:
        raise ValueError("gallery_dir is required for init_engine()")

    print("[ReIDEngine] Initializing single-model ReID...")
    print(f"  Config : {config_path}")
    print(f"  Weight : {weight_path}")
    print(f"  Gallery: {gallery_dir}")
    print(f"  Device : {device}")
    _engine = ReIDEngine(config_path=config_path, weight_path=weight_path, gallery_dir=gallery_dir, device=device)
    print("[ReIDEngine] Initialization done.")
    return _engine
