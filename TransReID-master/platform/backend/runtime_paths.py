"""Centralized runtime paths for backend uploads and artifacts."""

from __future__ import annotations

import os
from typing import Dict


BACKEND_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BACKEND_DIR, "uploads")

QUERY_DIR = os.path.join(UPLOAD_DIR, "query")
VIDEOS_DIR = os.path.join(UPLOAD_DIR, "videos")

TRAJECTORY_DIR = os.path.join(UPLOAD_DIR, "trajectory")
TRAJECTORY_RAW_DIR = os.path.join(TRAJECTORY_DIR, "raw")

GAME_ANALYSIS_DIR = os.path.join(UPLOAD_DIR, "game_analysis")
GAME_ANALYSIS_RAW_DIR = os.path.join(GAME_ANALYSIS_DIR, "raw")
GAME_ANALYSIS_EXPORTS_DIR = os.path.join(GAME_ANALYSIS_DIR, "exports")
GAME_ANALYSIS_TRAJECTORY_DIR = os.path.join(GAME_ANALYSIS_DIR, "trajectory")
GAME_ANALYSIS_STUBS_DIR = os.path.join(GAME_ANALYSIS_DIR, "stubs")


def ensure_runtime_dirs() -> Dict[str, str]:
    dirs = {
        "upload": UPLOAD_DIR,
        "query": QUERY_DIR,
        "videos": VIDEOS_DIR,
        "trajectory": TRAJECTORY_DIR,
        "trajectory_raw": TRAJECTORY_RAW_DIR,
        "game_analysis": GAME_ANALYSIS_DIR,
        "game_analysis_raw": GAME_ANALYSIS_RAW_DIR,
        "game_analysis_exports": GAME_ANALYSIS_EXPORTS_DIR,
        "game_analysis_trajectory": GAME_ANALYSIS_TRAJECTORY_DIR,
        "game_analysis_stubs": GAME_ANALYSIS_STUBS_DIR,
    }
    for p in dirs.values():
        os.makedirs(p, exist_ok=True)
    return dirs

