"""Check required local assets for BallShow runtime.

Usage:
  python tools/check_assets.py
  python tools/check_assets.py --json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


BACKEND_DIR = Path(__file__).resolve().parents[1]
PLATFORM_DIR = BACKEND_DIR.parent
INNER_ROOT = PLATFORM_DIR.parent
OUTER_ROOT = INNER_ROOT.parent


def _default_reid_weight() -> Path:
    return INNER_ROOT / "logs" / "92.6" / "transformer_120.pth"


def _default_gallery_dir() -> Path:
    return OUTER_ROOT / "data" / "BallShow" / "bounding_box_test"


def _default_third_party_repo() -> Path:
    outer = OUTER_ROOT / "_tmp_repo_basketball_analysis"
    inner = INNER_ROOT / "_tmp_repo_basketball_analysis"
    return outer if outer.exists() else inner


MODEL_LINKS = {
    "ball_detector_model.pt": "https://drive.google.com/file/d/1KejdrcEnto2AKjdgdo1U1syr5gODp6EL/view?usp=sharing",
    "court_keypoint_detector.pt": "https://drive.google.com/file/d/1nGoG-pUkSg4bWAUIeQ8aN6n7O1fOkXU0/view?usp=sharing",
    "player_detector.pt": "https://drive.google.com/file/d/1fVBLZtPy9Yu6Tf186oS4siotkioHBLHy/view?usp=sharing",
}


@dataclass
class CheckItem:
    name: str
    path: str
    required: bool
    exists: bool
    kind: str
    size_bytes: int
    hint: str = ""


def _item(name: str, path: Path, required: bool, kind: str, hint: str = "") -> CheckItem:
    exists = path.exists()
    size = path.stat().st_size if exists and path.is_file() else 0
    return CheckItem(
        name=name,
        path=str(path),
        required=required,
        exists=exists,
        kind=kind,
        size_bytes=size,
        hint=hint,
    )


def collect_items() -> List[CheckItem]:
    reid_weight_env = os.getenv("REID_SINGLE_WEIGHT_PATH", "").strip()
    reid_weight = Path(reid_weight_env) if reid_weight_env else _default_reid_weight()
    if not reid_weight.is_absolute():
        reid_weight = (INNER_ROOT / reid_weight).resolve()

    gallery_env = os.getenv("REID_GALLERY_DIR", "").strip()
    gallery_dir = Path(gallery_env) if gallery_env else _default_gallery_dir()
    if not gallery_dir.is_absolute():
        gallery_dir = (INNER_ROOT / gallery_dir).resolve()

    third_party_env = os.getenv("BA_REPO_PATH", "").strip()
    third_party_repo = Path(third_party_env) if third_party_env else _default_third_party_repo()
    if not third_party_repo.is_absolute():
        third_party_repo = (INNER_ROOT / third_party_repo).resolve()

    items: List[CheckItem] = [
        _item("reid_single_weight", reid_weight, True, "file"),
        _item("reid_gallery_dir", gallery_dir, True, "dir"),
        _item("third_party_repo", third_party_repo, False, "dir"),
        _item("traj_yolo11l", BACKEND_DIR / "yolo11l.pt", False, "file"),
        _item("traj_yolo11m", BACKEND_DIR / "yolo11m.pt", False, "file"),
        _item("traj_yolo11s_pose", BACKEND_DIR / "yolo11s-pose.pt", False, "file"),
    ]

    models_dir = third_party_repo / "models"
    for model_name, link in MODEL_LINKS.items():
        items.append(
            _item(
                f"third_party_model::{model_name}",
                models_dir / model_name,
                False,
                "file",
                hint=link,
            )
        )
    return items


def summarize(items: List[CheckItem]) -> Dict[str, object]:
    required_missing = [it for it in items if it.required and not it.exists]
    optional_missing = [it for it in items if not it.required and not it.exists]
    return {
        "ok": len(required_missing) == 0,
        "required_missing": [it.__dict__ for it in required_missing],
        "optional_missing": [it.__dict__ for it in optional_missing],
        "items": [it.__dict__ for it in items],
    }


def print_human(summary: Dict[str, object]) -> None:
    items = summary["items"]  # type: ignore[assignment]
    print("=" * 74)
    print("BallShow Asset Check")
    print("=" * 74)
    for raw in items:  # type: ignore[assignment]
        item = CheckItem(**raw)  # type: ignore[arg-type]
        marker = "OK " if item.exists else "MISS"
        req = "REQ" if item.required else "OPT"
        print(f"[{marker}] [{req}] {item.name}")
        print(f"       {item.path}")
        if item.hint and not item.exists:
            print(f"       hint: {item.hint}")
    print("-" * 74)
    if summary["ok"]:
        print("Status: READY")
    else:
        print("Status: BLOCKED (required files missing)")
    print("=" * 74)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local runtime assets")
    parser.add_argument("--json", action="store_true", help="Print JSON summary")
    args = parser.parse_args()

    items = collect_items()
    summary = summarize(items)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print_human(summary)
    return 0 if summary["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

