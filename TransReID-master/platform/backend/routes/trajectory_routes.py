"""Trajectory analysis routes and history APIs."""

from __future__ import annotations

import json
import math
import os
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from ..auth import get_current_user
from ..database import (
    create_trajectory_task,
    get_trajectory_task,
    get_user_trajectory_history,
    get_user_trajectory_history_paginated,
    get_user_trajectory_task_detail,
    update_trajectory_task,
)
from ..runtime_paths import TRAJECTORY_DIR, TRAJECTORY_RAW_DIR, ensure_runtime_dirs
from ..trajectory_engine import process_trajectory_video

router = APIRouter(prefix="/api/trajectory", tags=["Basketball Trajectory"])

ensure_runtime_dirs()
RAW_DIR = TRAJECTORY_RAW_DIR
OUT_DIR = TRAJECTORY_DIR

active_tasks: Dict[int, Dict[str, Any]] = {}


def _safe_result_json(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _normalize_trajectory_task(task: Dict[str, Any], include_points: bool = True) -> Dict[str, Any]:
    result = _safe_result_json(task.get("result_json", "{}"))

    tracking = (result.get("analysis") or {}).get("tracking") or {}
    shot_prediction = (result.get("analysis") or {}).get("shot_prediction") or {}
    shot_events = (result.get("analysis") or {}).get("shot_events") or []
    metrics = result.get("metrics") or {}
    artifacts = result.get("artifacts") or {}
    live_summary = (result.get("analysis") or {}).get("live_summary") or {}

    if not metrics:
        metrics = {
            "detected_points": int(tracking.get("detected_points", 0) or 0),
            "predicted_points": int(tracking.get("predicted_points", 0) or 0),
            "coverage_percent": float(tracking.get("coverage_percent", 0) or 0),
            "continuity_percent": float(tracking.get("continuity_percent", 0) or 0),
            "suppressed_false_positives": int(tracking.get("suppressed_false_positives", 0) or 0),
            "fit_points_count": int(tracking.get("fit_points_count", 0) or 0),
            "fit_quality_score": float(tracking.get("fit_quality_score", 0) or 0),
            "head_false_positive_suppressed": int(tracking.get("head_false_positive_suppressed", 0) or 0),
            "false_joint_suppressed": int(tracking.get("false_joint_suppressed", 0) or 0),
            "body_zone_suppressed": int(tracking.get("body_zone_suppressed", 0) or 0),
            "hsv_assist_hits": int(tracking.get("hsv_assist_hits", 0) or 0),
            "p95_lag_frames": int(tracking.get("p95_lag_frames", 0) or 0),
            "predicted_trajectory_points": int(len(((result.get("analysis") or {}).get("predicted_trajectory_points") or []))),
            "source_ratio": tracking.get("source_ratio", {}) or {},
            "static_hotspot_topk": tracking.get("static_hotspot_topk", []) or [],
            "reject_reason_histogram": tracking.get("reject_reason_histogram", {}) or {},
            "shot_count": int(tracking.get("shot_count", len(shot_events)) or 0),
            "made_count": int(tracking.get("made_count", 0) or 0),
            "miss_count": int(tracking.get("miss_count", 0) or 0),
            "event_retrigger_count": int(tracking.get("event_retrigger_count", 0) or 0),
            "max_consecutive_miss_in_flight": int(tracking.get("max_consecutive_miss_in_flight", 0) or 0),
            "blocked_release_count": int(tracking.get("blocked_release_count", 0) or 0),
        }
    else:
        metrics = dict(metrics)
        metrics.setdefault("fit_points_count", int(tracking.get("fit_points_count", 0) or 0))
        metrics.setdefault("fit_quality_score", float(tracking.get("fit_quality_score", 0) or 0))
        metrics.setdefault("head_false_positive_suppressed", int(tracking.get("head_false_positive_suppressed", 0) or 0))
        metrics.setdefault("false_joint_suppressed", int(tracking.get("false_joint_suppressed", 0) or 0))
        metrics.setdefault("body_zone_suppressed", int(tracking.get("body_zone_suppressed", 0) or 0))
        metrics.setdefault("hsv_assist_hits", int(tracking.get("hsv_assist_hits", 0) or 0))
        metrics.setdefault("p95_lag_frames", int(tracking.get("p95_lag_frames", 0) or 0))
        metrics.setdefault("predicted_trajectory_points", int(len(((result.get("analysis") or {}).get("predicted_trajectory_points") or []))))
        metrics.setdefault("source_ratio", tracking.get("source_ratio", {}) or {})
        metrics.setdefault("static_hotspot_topk", tracking.get("static_hotspot_topk", []) or [])
        metrics.setdefault("reject_reason_histogram", tracking.get("reject_reason_histogram", {}) or {})
        metrics.setdefault("shot_count", int(tracking.get("shot_count", len(shot_events)) or 0))
        metrics.setdefault("made_count", int(tracking.get("made_count", 0) or 0))
        metrics.setdefault("miss_count", int(tracking.get("miss_count", 0) or 0))
        metrics.setdefault("event_retrigger_count", int(tracking.get("event_retrigger_count", 0) or 0))
        metrics.setdefault("max_consecutive_miss_in_flight", int(tracking.get("max_consecutive_miss_in_flight", 0) or 0))
        metrics.setdefault("blocked_release_count", int(tracking.get("blocked_release_count", 0) or 0))

    if not shot_prediction and shot_events:
        latest = shot_events[-1]
        result_name = str(latest.get("result") or "unknown")
        label = "Unknown"
        if result_name == "made":
            label = "Basket"
        elif result_name in {"miss", "timeout"}:
            label = "No Basket"
        end_info = latest.get("end") or {}
        shot_prediction = {
            "label": label,
            "confidence": 0.0,
            "crossing_frame": end_info.get("frame"),
            "crossing_time": end_info.get("time"),
            "reason": str(latest.get("stop_reason") or "shot_event"),
        }

    if not include_points and result.get("analysis") and result["analysis"].get("trajectory_points"):
        # Keep summary lightweight for list endpoints.
        result = dict(result)
        result["analysis"] = dict(result["analysis"])
        result["analysis"]["trajectory_points"] = []
        result["analysis"]["fit_trajectory_points"] = []
        result["analysis"]["predicted_trajectory_points"] = []

    normalized = {
        "id": task["id"],
        "task_id": task["id"],
        "task_type": "trajectory",
        "status": task.get("status", "pending"),
        "created_at": task.get("created_at"),
        "finished_at": task.get("finished_at"),
        "video_filename": task.get("video_filename", ""),
        "total_frames": int(task.get("total_frames", 0) or 0),
        "detected_points": int(task.get("detected_points", 0) or 0),
        "metrics": metrics,
        "artifacts": artifacts,
        "live_summary": live_summary,
        "shot_prediction": shot_prediction,
        "analysis": result if task.get("status") == "completed" else None,
        "error": result.get("error") if task.get("status") in {"failed", "cancelled"} else None,
    }

    return normalized


def _run_trajectory(task_id: int, raw_video_path: str, rim_calibration: Optional[Dict[str, float]] = None) -> None:
    try:
        update_trajectory_task(task_id, status="processing")

        result = process_trajectory_video(
            raw_video_path,
            output_dir=OUT_DIR,
            rim_calibration=rim_calibration,
            cancel_check=lambda: active_tasks.get(task_id, {}).get("cancelled", False),
            progress_callback=lambda p: active_tasks.setdefault(task_id, {}).update(
                {"progress": max(0, min(100, int(p)))}
            ),
            state_callback=lambda state: active_tasks.setdefault(task_id, {}).update(
                {"live_state": state or {}}
            ),
        )

        if result.get("cancelled"):
            update_trajectory_task(
                task_id,
                status="cancelled",
                result_json=json.dumps({"error": "Task cancelled by user"}, ensure_ascii=False),
                finished_at=datetime.now().isoformat(),
            )
            return

        if result.get("error"):
            raise RuntimeError(str(result["error"]))

        tracking = (result.get("analysis") or {}).get("tracking") or {}
        metrics_detected = result.get("metrics", {}).get("detected_points")
        detected_points = int(metrics_detected if metrics_detected is not None else tracking.get("detected_points", 0))

        update_trajectory_task(
            task_id,
            status="completed",
            total_frames=int((result.get("video_info") or {}).get("total_frames", 0)),
            detected_points=detected_points,
            result_json=json.dumps(result, ensure_ascii=False),
            finished_at=datetime.now().isoformat(),
        )
    except Exception as exc:
        update_trajectory_task(
            task_id,
            status="failed",
            result_json=json.dumps({"error": str(exc)}, ensure_ascii=False),
            finished_at=datetime.now().isoformat(),
        )
    finally:
        active_tasks.pop(task_id, None)


@router.post("/analyze")
async def analyze_trajectory_video(
    video: UploadFile = File(...),
    rim_cx: Optional[float] = Form(default=None),
    rim_cy: Optional[float] = Form(default=None),
    rim_r: Optional[float] = Form(default=None),
    user: Dict = Depends(get_current_user),
):
    ext = os.path.splitext(video.filename or "shot.mp4")[1] or ".mp4"
    raw_name = f"trajraw_{uuid.uuid4().hex}{ext}"
    raw_path = os.path.join(RAW_DIR, raw_name)

    with open(raw_path, "wb") as out:
        out.write(await video.read())

    task_id = create_trajectory_task(user["user_id"], raw_name)
    rim_calibration: Optional[Dict[str, float]] = None
    if rim_cx is not None and rim_cy is not None and rim_r is not None:
        if rim_r <= 0:
            raise HTTPException(400, "篮筐半径必须大于 0")
        rim_calibration = {"cx": float(rim_cx), "cy": float(rim_cy), "r": float(rim_r)}
    active_tasks[task_id] = {
        "cancelled": False,
        "progress": 0,
        "live_state": {
            "processed_frame": 0,
            "last_detected_frame": -1,
            "last_emitted_frame": -1,
            "lag_frames": 0,
            "phase": "pre_shot",
            "event_state": "IDLE",
            "shot_event_active": False,
            "causal_mode": True,
        },
    }

    thread = threading.Thread(target=_run_trajectory, args=(task_id, raw_path, rim_calibration), daemon=True)
    thread.start()

    return {
        "message": "Trajectory analysis task submitted",
        "task_id": task_id,
        "task_type": "trajectory",
        "video_filename": raw_name,
        "status": "pending",
    }


@router.get("/task/{task_id}")
async def get_trajectory_status(task_id: int, user: Dict = Depends(get_current_user)):
    task = get_trajectory_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task["user_id"] != user["user_id"]:
        raise HTTPException(403, "No permission")

    normalized = _normalize_trajectory_task(task, include_points=True)

    payload: Dict[str, Any] = {
        "task_id": task["id"],
        "task_type": "trajectory",
        "status": task["status"],
        "video_filename": task.get("video_filename"),
        "total_frames": task.get("total_frames"),
        "detected_points": task.get("detected_points"),
        "created_at": task.get("created_at"),
        "finished_at": task.get("finished_at"),
        "metrics": normalized.get("metrics"),
        "artifacts": normalized.get("artifacts"),
        "live_state": normalized.get("live_summary") or {
            "processed_frame": 0,
            "last_detected_frame": -1,
            "last_emitted_frame": -1,
            "lag_frames": 0,
            "phase": "pre_shot",
            "event_state": "IDLE",
            "shot_event_active": False,
            "causal_mode": True,
        },
        "task": normalized,
    }

    if task["status"] == "processing" and task_id in active_tasks:
        payload["progress"] = active_tasks[task_id].get("progress", 0)
        payload["live_state"] = active_tasks[task_id].get("live_state", payload["live_state"])

    if task["status"] == "completed":
        payload["analysis"] = normalized["analysis"]
    elif task["status"] in {"failed", "cancelled"}:
        payload["error"] = normalized.get("error") or "Unknown error"

    return payload


@router.post("/cancel/{task_id}")
async def cancel_trajectory_task(task_id: int, user: Dict = Depends(get_current_user)):
    task = get_trajectory_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task["user_id"] != user["user_id"]:
        raise HTTPException(403, "No permission")

    if task["status"] == "processing":
        if task_id not in active_tasks:
            active_tasks[task_id] = {"cancelled": False, "progress": 0}
        active_tasks[task_id]["cancelled"] = True
        return {"message": "Cancel signal sent"}

    return {"message": f"Task status={task['status']}, cannot cancel"}


@router.get("/history")
async def get_trajectory_history(
    user: Dict = Depends(get_current_user),
    page: Optional[int] = Query(default=None, ge=1),
    page_size: Optional[int] = Query(default=None, ge=1, le=100),
    status: Optional[str] = Query(default=None),
    include_points: bool = Query(default=False),
):
    # Legacy mode for compatibility with old clients.
    if page is None and page_size is None and status is None:
        history = get_user_trajectory_history(user["user_id"])
        return [_normalize_trajectory_task(item, include_points=include_points) for item in history]

    safe_page = page or 1
    safe_size = page_size or 10
    rows, total = get_user_trajectory_history_paginated(
        user_id=user["user_id"],
        page=safe_page,
        page_size=safe_size,
        status=status,
    )

    items = [_normalize_trajectory_task(item, include_points=include_points) for item in rows]
    total_pages = int(math.ceil(total / max(1, safe_size)))
    return {
        "items": items,
        "pagination": {
            "page": safe_page,
            "page_size": safe_size,
            "total": total,
            "total_pages": total_pages,
        },
        "filters": {
            "status": status,
            "include_points": include_points,
        },
    }


@router.get("/history/{task_id}")
async def get_trajectory_history_detail(
    task_id: int,
    user: Dict = Depends(get_current_user),
    include_points: bool = Query(default=True),
):
    task = get_user_trajectory_task_detail(user["user_id"], task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    return _normalize_trajectory_task(task, include_points=include_points)
