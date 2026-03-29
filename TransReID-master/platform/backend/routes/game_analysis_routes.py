"""Unified game-analysis routes."""

from __future__ import annotations

import csv
import json
import math
import os
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from ..auth import get_current_user
from ..database import (
    create_game_analysis_export,
    create_game_analysis_task,
    get_game_analysis_task,
    get_user_game_analysis_history_paginated,
    get_user_game_analysis_task_detail,
    get_task_exports,
    update_game_analysis_task,
)
from ..game_analysis_engine import process_game_analysis_video
from ..reid_engine import get_engine
from ..runtime_paths import (
    GAME_ANALYSIS_DIR,
    GAME_ANALYSIS_EXPORTS_DIR,
    GAME_ANALYSIS_RAW_DIR,
    QUERY_DIR,
    UPLOAD_DIR,
    ensure_runtime_dirs,
)

router = APIRouter(prefix="/api/game-analysis", tags=["Game Analysis"])

ensure_runtime_dirs()
RAW_DIR = GAME_ANALYSIS_RAW_DIR
OUT_DIR = GAME_ANALYSIS_DIR
EXPORT_DIR = GAME_ANALYSIS_EXPORTS_DIR

active_tasks: Dict[int, Dict[str, Any]] = {}


def _safe_result_json(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _normalize_task(task: Dict[str, Any], include_analysis: bool = True) -> Dict[str, Any]:
    result = _safe_result_json(task.get("result_json", "{}"))
    artifacts = dict(result.get("artifacts") or {})
    traj_url = str(artifacts.get("trajectory_video_url") or "").strip()
    if traj_url.startswith("/uploads/trajectory/"):
        filename = os.path.basename(traj_url.split("?", 1)[0])
        if filename:
            local = os.path.join(UPLOAD_DIR, "game_analysis", "trajectory", filename)
            if os.path.exists(local):
                artifacts["trajectory_video_url"] = f"/uploads/game_analysis/trajectory/{filename}"
    tracking_url = str(artifacts.get("tracking_video_url") or "").strip()
    if tracking_url.startswith("/uploads/trajectory/"):
        filename = os.path.basename(tracking_url.split("?", 1)[0])
        if filename:
            local = os.path.join(UPLOAD_DIR, "game_analysis", "trajectory", filename)
            if os.path.exists(local):
                artifacts["tracking_video_url"] = f"/uploads/game_analysis/trajectory/{filename}"
    video_name = str(task.get("video_filename") or "").strip()
    if video_name:
        artifacts.setdefault("raw_video_url", f"/uploads/game_analysis/raw/{video_name}")
    artifacts.setdefault(
        "analysis_video_url",
        str(
            artifacts.get("composite_video_url")
            or artifacts.get("highlight_video_url")
            or artifacts.get("trajectory_video_url")
            or artifacts.get("tracking_video_url")
            or artifacts.get("raw_video_url")
            or ""
        ),
    )
    payload: Dict[str, Any] = {
        "id": int(task["id"]),
        "task_id": int(task["id"]),
        "task_type": "game_analysis",
        "status": str(task.get("status") or "pending"),
        "video_filename": task.get("video_filename") or "",
        "query_image": task.get("query_image") or "",
        "total_frames": int(task.get("total_frames") or 0),
        "created_at": task.get("created_at"),
        "finished_at": task.get("finished_at"),
        "options": _safe_result_json(task.get("options_json") or "{}"),
        "metrics": (result.get("metrics") or {}),
        "artifacts": artifacts,
        "error": result.get("error") if str(task.get("status")) in {"failed", "cancelled"} else None,
    }
    if include_analysis and str(task.get("status")) == "completed":
        payload["analysis_bundle"] = result.get("analysis_bundle") or {}
        payload["analysis"] = result.get("analysis") or {}
    return payload


def _run_analysis(task_id: int, raw_video_path: str, query_image_path: Optional[str], options: Dict[str, Any]) -> None:
    try:
        update_game_analysis_task(task_id, status="processing")
        try:
            reid_engine = get_engine()
        except Exception:
            reid_engine = None

        result = process_game_analysis_video(
            video_path=raw_video_path,
            output_dir=OUT_DIR,
            enable_possession=bool(options.get("enable_possession", True)),
            enable_highlight=bool(options.get("enable_highlight", True)),
            highlight_mode=str(options.get("highlight_mode", "action")),
            enable_trajectory=bool(options.get("enable_trajectory", False)),
            query_image_path=query_image_path,
            reid_engine=reid_engine,
            rim_calibration=options.get("rim_calibration"),
            cancel_check=lambda: active_tasks.get(task_id, {}).get("cancelled", False),
            progress_callback=lambda p: active_tasks.setdefault(task_id, {}).update({"progress": max(0, min(100, int(p)))}),
        )

        if result.get("cancelled"):
            update_game_analysis_task(
                task_id,
                status="cancelled",
                result_json=json.dumps({"error": "Task cancelled by user"}, ensure_ascii=False),
                finished_at=datetime.now().isoformat(),
            )
            return
        if result.get("error"):
            raise RuntimeError(str(result["error"]))

        update_game_analysis_task(
            task_id,
            status="completed",
            total_frames=int((result.get("video_info") or {}).get("total_frames", 0)),
            result_json=json.dumps(result, ensure_ascii=False),
            finished_at=datetime.now().isoformat(),
        )
    except Exception as exc:
        update_game_analysis_task(
            task_id,
            status="failed",
            result_json=json.dumps({"error": str(exc)}, ensure_ascii=False),
            finished_at=datetime.now().isoformat(),
        )
    finally:
        active_tasks.pop(task_id, None)


@router.post("/analyze")
async def analyze_game_video(
    video: UploadFile = File(...),
    enable_possession: bool = Form(default=True),
    enable_highlight: bool = Form(default=True),
    highlight_mode: str = Form(default="action"),
    enable_trajectory: bool = Form(default=False),
    query: Optional[UploadFile] = File(default=None),
    rim_cx: Optional[float] = Form(default=None),
    rim_cy: Optional[float] = Form(default=None),
    rim_r: Optional[float] = Form(default=None),
    user: Dict = Depends(get_current_user),
):
    ext = os.path.splitext(video.filename or "game.mp4")[1] or ".mp4"
    raw_name = f"ga_raw_{uuid.uuid4().hex}{ext}"
    raw_path = os.path.join(RAW_DIR, raw_name)
    with open(raw_path, "wb") as out:
        out.write(await video.read())

    query_filename = ""
    query_path: Optional[str] = None
    if query is not None:
        q_ext = os.path.splitext(query.filename or "query.jpg")[1] or ".jpg"
        query_filename = f"ga_query_{uuid.uuid4().hex}{q_ext}"
        query_path = os.path.join(QUERY_DIR, query_filename)
        with open(query_path, "wb") as out:
            out.write(await query.read())

    hm = (highlight_mode or "action").strip().lower()
    if hm not in {"action", "player"}:
        raise HTTPException(400, "highlight_mode must be one of: action, player")
    if hm == "player" and not query_path:
        raise HTTPException(400, "player highlight mode requires query image")

    rim_calibration = None
    if rim_cx is not None and rim_cy is not None and rim_r is not None:
        if float(rim_r) <= 0:
            raise HTTPException(400, "rim_r must be > 0")
        rim_calibration = {"cx": float(rim_cx), "cy": float(rim_cy), "r": float(rim_r)}

    options = {
        "enable_possession": bool(enable_possession),
        "enable_highlight": bool(enable_highlight),
        "highlight_mode": hm,
        "enable_trajectory": bool(enable_trajectory),
        "rim_calibration": rim_calibration,
    }

    task_id = create_game_analysis_task(
        user_id=int(user["user_id"]),
        video_filename=raw_name,
        query_image=query_filename,
        options_json=json.dumps(options, ensure_ascii=False),
    )
    active_tasks[task_id] = {"cancelled": False, "progress": 0}

    thread = threading.Thread(
        target=_run_analysis,
        args=(task_id, raw_path, query_path, options),
        daemon=True,
    )
    thread.start()

    return {
        "message": "Game analysis task submitted",
        "task_id": task_id,
        "task_type": "game_analysis",
        "status": "pending",
        "video_filename": raw_name,
    }


@router.get("/task/{task_id}")
async def get_game_task(task_id: int, user: Dict = Depends(get_current_user)):
    task = get_game_analysis_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if int(task["user_id"]) != int(user["user_id"]):
        raise HTTPException(403, "No permission")

    normalized = _normalize_task(task, include_analysis=True)
    payload: Dict[str, Any] = {
        "task_id": task_id,
        "task_type": "game_analysis",
        "status": task.get("status"),
        "video_filename": task.get("video_filename"),
        "created_at": task.get("created_at"),
        "finished_at": task.get("finished_at"),
        "metrics": normalized.get("metrics", {}),
        "artifacts": normalized.get("artifacts", {}),
        "options": normalized.get("options", {}),
        "task": normalized,
    }

    if str(task.get("status")) == "processing" and task_id in active_tasks:
        payload["progress"] = active_tasks[task_id].get("progress", 0)
    if str(task.get("status")) == "completed":
        payload["analysis_bundle"] = normalized.get("analysis_bundle", {})
        payload["analysis"] = normalized.get("analysis", {})
    elif str(task.get("status")) in {"failed", "cancelled"}:
        payload["error"] = normalized.get("error") or "Unknown error"

    payload["exports"] = get_task_exports(task_id=task_id, user_id=int(user["user_id"]))
    return payload


@router.post("/cancel/{task_id}")
async def cancel_game_task(task_id: int, user: Dict = Depends(get_current_user)):
    task = get_game_analysis_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if int(task["user_id"]) != int(user["user_id"]):
        raise HTTPException(403, "No permission")
    if str(task.get("status")) == "processing":
        active_tasks.setdefault(task_id, {"cancelled": False, "progress": 0})
        active_tasks[task_id]["cancelled"] = True
        return {"message": "Cancel signal sent"}
    return {"message": f"Task status={task.get('status')}, cannot cancel"}


@router.get("/history")
async def get_game_history(
    user: Dict = Depends(get_current_user),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    status: Optional[str] = Query(default=None),
    include_analysis: bool = Query(default=False),
):
    rows, total = get_user_game_analysis_history_paginated(
        user_id=int(user["user_id"]),
        page=page,
        page_size=page_size,
        status=status,
    )
    items = [_normalize_task(r, include_analysis=include_analysis) for r in rows]
    return {
        "items": items,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": int(math.ceil(total / max(1, page_size))),
        },
        "filters": {"status": status, "include_analysis": include_analysis},
    }


@router.get("/export/{task_id}")
async def export_game_analysis(
    task_id: int,
    format: str = Query(default="json"),
    user: Dict = Depends(get_current_user),
):
    task = get_user_game_analysis_task_detail(user_id=int(user["user_id"]), task_id=task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if str(task.get("status")) != "completed":
        raise HTTPException(400, "Task is not completed")

    fmt = (format or "json").strip().lower()
    if fmt not in {"json", "csv"}:
        raise HTTPException(400, "format must be json or csv")

    result = _safe_result_json(task.get("result_json") or "{}")
    export_name = f"ga_export_{task_id}_{uuid.uuid4().hex[:8]}.{fmt}"
    export_path = os.path.join(EXPORT_DIR, export_name)

    if fmt == "json":
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    else:
        bundle = result.get("analysis_bundle") or {}
        possession_tl = (bundle.get("possession") or {}).get("timeline") or []
        events = bundle.get("events") or []
        clips = (bundle.get("highlights") or {}).get("clips") or []
        with open(export_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "record_type",
                    "id",
                    "label",
                    "start_time",
                    "end_time",
                    "duration_s",
                    "frame",
                    "time",
                    "team_id",
                    "player_id",
                    "from_player_id",
                    "to_player_id",
                    "confidence",
                ]
            )
            for i, seg in enumerate(possession_tl, start=1):
                writer.writerow(
                    [
                        "possession",
                        i,
                        "possession_segment",
                        seg.get("start_time"),
                        seg.get("end_time"),
                        seg.get("duration_s"),
                        "",
                        "",
                        seg.get("team_id"),
                        seg.get("player_id"),
                        "",
                        "",
                        "",
                    ]
                )
            for ev in events:
                writer.writerow(
                    [
                        "event",
                        ev.get("event_id"),
                        ev.get("type"),
                        "",
                        "",
                        "",
                        ev.get("frame"),
                        ev.get("time"),
                        ev.get("team_id"),
                        "",
                        ev.get("from_player_id"),
                        ev.get("to_player_id"),
                        ev.get("confidence"),
                    ]
                )
            for cl in clips:
                writer.writerow(
                    [
                        "highlight",
                        cl.get("id"),
                        cl.get("label"),
                        cl.get("start_time"),
                        cl.get("end_time"),
                        cl.get("duration"),
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        cl.get("score", ""),
                    ]
                )

    create_game_analysis_export(
        task_id=task_id,
        user_id=int(user["user_id"]),
        fmt=fmt,
        filename=export_name,
    )

    return {
        "task_id": task_id,
        "format": fmt,
        "filename": export_name,
        "download_url": f"/uploads/game_analysis/exports/{export_name}",
    }
