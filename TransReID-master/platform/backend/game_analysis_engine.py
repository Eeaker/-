"""Unified basketball game analysis engine."""

from __future__ import annotations

import math
import os
import shutil
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .third_party_basketball_adapter import run_third_party_basketball_analysis
from .trajectory_engine import process_trajectory_video
from .video_engine import process_video as process_player_highlight_video

CancelFn = Callable[[], bool]
ProgressFn = Callable[[int], None]


def _safe_progress(cb: ProgressFn, value: float) -> None:
    try:
        cb(max(0, min(100, int(round(float(value))))))
    except Exception:
        pass


def _ensure_ultralytics_writable_config() -> None:
    """Force Ultralytics to use a writable local settings directory."""
    cfg_dir = os.getenv("YOLO_CONFIG_DIR", "").strip()
    if cfg_dir and os.path.isdir(cfg_dir) and os.access(cfg_dir, os.W_OK):
        return
    backend_dir = os.path.dirname(__file__)
    local_cfg = os.path.join(backend_dir, ".ultralytics_cfg")
    os.makedirs(local_cfg, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = local_cfg


def _bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    ab = max(1e-6, (bx2 - bx1) * (by2 - by1))
    return float(inter / (aa + ab - inter + 1e-6))


def _bbox_center(b: Sequence[float]) -> Tuple[float, float]:
    return float((b[0] + b[2]) * 0.5), float((b[1] + b[3]) * 0.5)


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


class _TrackManager:
    def __init__(self, iou_th: float = 0.24, max_missing: int = 8):
        self.iou_th = float(iou_th)
        self.max_missing = int(max_missing)
        self.next_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}

    def update(self, dets: List[Tuple[float, float, float, float]]) -> Dict[int, Dict[str, Any]]:
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["missing"] = int(self.tracks[tid].get("missing", 0)) + 1
            if self.tracks[tid]["missing"] > self.max_missing:
                del self.tracks[tid]

        used_tracks = set()
        used_dets = set()
        pairs: List[Tuple[float, int, int]] = []
        for tid, t in self.tracks.items():
            for di, db in enumerate(dets):
                iou = _bbox_iou(t["bbox"], db)
                if iou >= self.iou_th:
                    pairs.append((iou, tid, di))
        pairs.sort(reverse=True, key=lambda x: x[0])

        for _, tid, di in pairs:
            if tid in used_tracks or di in used_dets:
                continue
            used_tracks.add(tid)
            used_dets.add(di)
            self.tracks[tid] = {"bbox": tuple(dets[di]), "missing": 0}

        for di, db in enumerate(dets):
            if di in used_dets:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {"bbox": tuple(db), "missing": 0}

        return {tid: {"bbox": [float(v) for v in t["bbox"]]} for tid, t in self.tracks.items() if int(t.get("missing", 0)) == 0}


def _interpolate_ball(ball_tracks: List[Dict[int, Dict[str, Any]]], max_gap: int = 24) -> List[Dict[int, Dict[str, Any]]]:
    b: List[Optional[np.ndarray]] = []
    for it in ball_tracks:
        bb = it.get(1, {}).get("bbox")
        b.append(np.asarray(bb, dtype=np.float32) if bb else None)
    idx = [i for i, x in enumerate(b) if x is not None]
    for j in range(len(idx) - 1):
        i0, i1 = idx[j], idx[j + 1]
        gap = i1 - i0 - 1
        if gap <= 0 or gap > max_gap:
            continue
        for t in range(1, gap + 1):
            a = t / float(gap + 1)
            b[i0 + t] = (1.0 - a) * b[i0] + a * b[i1]
    out: List[Dict[int, Dict[str, Any]]] = []
    for bb in b:
        out.append({1: {"bbox": [float(v) for v in bb.tolist()]}} if bb is not None else {})
    return out


def _detect_possession(player_tracks: List[Dict[int, Dict[str, Any]]], ball_tracks: List[Dict[int, Dict[str, Any]]]) -> List[int]:
    possession = [-1] * min(len(player_tracks), len(ball_tracks))
    streak: Dict[int, int] = {}
    for i in range(len(possession)):
        bb = ball_tracks[i].get(1, {}).get("bbox")
        if not bb:
            streak = {}
            continue
        bc = _bbox_center(bb)
        best_pid, best_d = -1, 1e9
        for pid, pinfo in player_tracks[i].items():
            pb = pinfo.get("bbox")
            if not pb:
                continue
            d = _dist(bc, _bbox_center(pb))
            if d < best_d:
                best_d, best_pid = int(pid), d
        if best_pid == -1 or best_d > 65.0:
            streak = {}
            continue
        c = int(streak.get(best_pid, 0)) + 1
        streak = {best_pid: c}
        if c >= 4:
            possession[i] = best_pid
    return possession


def _pass_interception_events(
    possession: List[int],
    frame_indices: List[int],
    fps: float,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    eid = 1
    prev_holder = -1
    for i in range(1, len(possession)):
        if possession[i - 1] != -1:
            prev_holder = possession[i - 1]
        cur = possession[i]
        if prev_holder != -1 and cur != -1 and cur != prev_holder:
            abs_frame = int(frame_indices[i])
            events.append(
                {
                    "event_id": eid,
                    "type": "pass",
                    "frame": abs_frame,
                    "time": round(float(abs_frame / max(1.0, fps)), 3),
                    "from_player_id": int(prev_holder),
                    "to_player_id": int(cur),
                    "team_id": 1,
                    "confidence": 0.7,
                }
            )
            eid += 1
    return events


def _possession_summary(possession: List[int], frame_indices: List[int], fps: float) -> Dict[str, Any]:
    timeline: List[Dict[str, Any]] = []
    holder_secs: Dict[int, float] = {}
    i = 0
    while i < len(possession):
        holder = possession[i]
        j = i
        while j + 1 < len(possession) and possession[j + 1] == holder:
            j += 1
        if holder != -1:
            sf, ef = int(frame_indices[i]), int(frame_indices[j])
            dur = float((ef - sf + 1) / max(1.0, fps))
            timeline.append(
                {
                    "start_frame": sf,
                    "end_frame": ef,
                    "start_time": round(float(sf / max(1.0, fps)), 3),
                    "end_time": round(float(ef / max(1.0, fps)), 3),
                    "duration_s": round(dur, 3),
                    "player_id": int(holder),
                    "team_id": 1,
                }
            )
            holder_secs[int(holder)] = holder_secs.get(int(holder), 0.0) + dur
        i = j + 1
    total = sum(holder_secs.values())
    stats = [{"player_id": int(pid), "duration_s": round(v, 3), "ratio": round(v / max(1e-6, total), 4)} for pid, v in sorted(holder_secs.items(), key=lambda x: x[1], reverse=True)]
    return {"timeline": timeline, "holder_stats": stats, "team_possession_ratio": {"1": 1.0 if total > 0 else 0.0}}


def _action_clips(events: List[Dict[str, Any]], shot_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clips: List[Dict[str, Any]] = []
    for ev in events:
        t = float(ev.get("time", 0.0))
        clips.append({"start_time": max(0.0, t - 1.0), "end_time": t + 1.2, "label": str(ev.get("type") or "event"), "source_event_ids": [int(ev.get("event_id", 0))]})
    for se in shot_events:
        s = float((se.get("start") or {}).get("time") or 0.0)
        e = float((se.get("end") or {}).get("time") or (s + 1.2))
        label = "shot_made" if str(se.get("result")) == "made" else "shot_miss"
        clips.append({"start_time": max(0.0, s - 0.6), "end_time": e + 0.4, "label": label, "source_event_ids": [int(se.get("event_id", 0))]})
    clips.sort(key=lambda x: (float(x["start_time"]), float(x["end_time"])))
    for i, c in enumerate(clips, start=1):
        c["id"] = int(i)
        c["duration"] = round(float(c["end_time"]) - float(c["start_time"]), 3)
        c["labels"] = [c["label"]]
        c["video_clip_url"] = None
    return clips


def _merge_highlight_clips(clips: List[Dict[str, Any]], merge_gap_s: float = 0.12) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for c in clips or []:
        try:
            s = max(0.0, float(c.get("start_time", 0.0)))
            e = float(c.get("end_time", s))
        except Exception:
            continue
        if e <= s + 0.03:
            continue
        labels = list(c.get("labels") or [])
        if not labels:
            labels = [str(c.get("label") or "event")]
        normalized.append(
            {
                "start_time": s,
                "end_time": e,
                "labels": [str(x) for x in labels if str(x).strip()],
                "source_event_ids": [int(x) for x in (c.get("source_event_ids") or []) if str(x).isdigit()],
            }
        )
    if not normalized:
        return []
    normalized.sort(key=lambda x: (float(x["start_time"]), float(x["end_time"])))

    merged: List[Dict[str, Any]] = []
    for c in normalized:
        if not merged:
            merged.append(c)
            continue
        last = merged[-1]
        if float(c["start_time"]) <= float(last["end_time"]) + float(merge_gap_s):
            last["end_time"] = max(float(last["end_time"]), float(c["end_time"]))
            last["labels"] = sorted(set((last.get("labels") or []) + (c.get("labels") or [])))
            last["source_event_ids"] = sorted(set((last.get("source_event_ids") or []) + (c.get("source_event_ids") or [])))
        else:
            merged.append(c)

    out: List[Dict[str, Any]] = []
    for i, c in enumerate(merged, start=1):
        labels = list(c.get("labels") or ["event"])
        out.append(
            {
                "id": int(i),
                "start_time": round(float(c["start_time"]), 3),
                "end_time": round(float(c["end_time"]), 3),
                "duration": round(float(c["end_time"]) - float(c["start_time"]), 3),
                "label": labels[0],
                "labels": labels,
                "source_event_ids": list(c.get("source_event_ids") or []),
                "video_clip_url": None,
            }
        )
    return out


def _open_mp4_writer(output_path: str, fps: float, size: Tuple[int, int]) -> Optional[cv2.VideoWriter]:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    w, h = int(size[0]), int(size[1])
    for codec in ("avc1", "H264", "X264", "mp4v"):
        try:
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*codec), max(1.0, float(fps)), (w, h))
        except Exception:
            writer = None
        if writer is not None and writer.isOpened():
            return writer
        try:
            if writer is not None:
                writer.release()
        except Exception:
            pass
    return None


def _render_tracking_trajectory_video(
    base_video_path: str,
    output_dir: str,
    prefix: str,
    ball_points: List[Dict[str, Any]],
    highlight_clips: Optional[List[Dict[str, Any]]] = None,
    draw_trajectory: bool = True,
    draw_prediction: bool = True,
) -> Optional[str]:
    if not base_video_path or not os.path.exists(base_video_path):
        return None
    cap = cv2.VideoCapture(base_video_path)
    if not cap.isOpened():
        return None
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        return None

    filename = f"{prefix}_{uuid.uuid4().hex[:10]}.mp4"
    output_path = os.path.join(output_dir, filename)
    writer = _open_mp4_writer(output_path, fps=fps, size=(width, height))
    if writer is None:
        cap.release()
        return None

    point_by_index: Dict[int, Tuple[float, float]] = {}
    for p in ball_points or []:
        try:
            idx = int(p.get("sampled_index"))
            x = float(p.get("x"))
            y = float(p.get("y"))
        except Exception:
            continue
        point_by_index[idx] = (x, y)

    highlight_ranges: List[Tuple[float, float]] = []
    for c in highlight_clips or []:
        try:
            s = max(0.0, float(c.get("start_time", 0.0)))
            e = float(c.get("end_time", s))
        except Exception:
            continue
        if e > s:
            highlight_ranges.append((s, e))

    valid_points: List[Tuple[int, float, float]] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx in point_by_index:
            x, y = point_by_index[frame_idx]
            valid_points.append((frame_idx, x, y))

        overlay = frame.copy()
        if draw_trajectory and len(valid_points) >= 2:
            trail = valid_points[-32:]
            for i in range(1, len(trail)):
                _, x1, y1 = trail[i - 1]
                _, x2, y2 = trail[i]
                age = i / max(1.0, float(len(trail)))
                color = (36, 36, min(255, int(120 + 120 * age)))  # red-ish in BGR
                thickness = 1 if i < len(trail) - 6 else 2
                cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, cv2.LINE_AA)
            cx, cy = int(trail[-1][1]), int(trail[-1][2])
            cv2.circle(overlay, (cx, cy), 4, (54, 84, 255), -1, cv2.LINE_AA)

        if draw_prediction and len(valid_points) >= 6:
            recent = valid_points[-12:]
            t = np.asarray([float(p[0]) for p in recent], dtype=np.float32)
            xs = np.asarray([float(p[1]) for p in recent], dtype=np.float32)
            ys = np.asarray([float(p[2]) for p in recent], dtype=np.float32)
            if float(np.ptp(t)) >= 3.0 and float(np.ptp(ys)) >= 8.0:
                try:
                    x_coef = np.polyfit(t, xs, 1)
                    y_coef = np.polyfit(t, ys, 2)
                    fut = np.linspace(float(t[-1]), float(t[-1] + 22.0), num=18)
                    fx = x_coef[0] * fut + x_coef[1]
                    fy = y_coef[0] * fut * fut + y_coef[1] * fut + y_coef[2]
                    pred = []
                    for i in range(len(fut)):
                        px = int(np.clip(fx[i], 0, width - 1))
                        py = int(np.clip(fy[i], 0, height - 1))
                        pred.append((px, py))
                    for i in range(1, len(pred)):
                        cv2.line(overlay, pred[i - 1], pred[i], (84, 224, 140), 1, cv2.LINE_AA)
                except Exception:
                    pass

        if highlight_ranges:
            ts = float(frame_idx / max(1.0, fps))
            in_highlight = any((s <= ts <= e) for s, e in highlight_ranges)
            if in_highlight:
                cv2.rectangle(overlay, (16, 16), (182, 52), (38, 156, 255), -1, cv2.LINE_AA)
                cv2.putText(overlay, "HIGHLIGHT", (28, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)

        frame = cv2.addWeighted(overlay, 0.72, frame, 0.28, 0.0)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    if not os.path.exists(output_path) or os.path.getsize(output_path) <= 0:
        return None
    return filename


def _render_highlight_video(video_path: str, clips: List[Dict[str, Any]], output_dir: str, prefix: str) -> Optional[str]:
    if not clips:
        return None
    try:
        from moviepy.editor import VideoFileClip, concatenate_videoclips  # type: ignore
    except Exception:
        return None
    filename = f"{prefix}_{uuid.uuid4().hex[:10]}.mp4"
    dst = os.path.join(output_dir, filename)
    source = VideoFileClip(video_path)
    parts = []
    try:
        for c in clips:
            s = max(0.0, float(c["start_time"]))
            e = min(float(source.duration), float(c["end_time"]))
            if e > s + 0.05:
                parts.append(source.subclip(s, e))
        if not parts:
            source.close()
            return None
        final = concatenate_videoclips(parts)
        final.write_videofile(dst, codec="libx264", audio_codec="aac", logger=None, preset="fast")
        final.close()
        source.close()
        for p in parts:
            p.close()
        return filename
    except Exception:
        try:
            source.close()
        except Exception:
            pass
        for p in parts:
            try:
                p.close()
            except Exception:
                pass
        return None


def _relocate_highlight_to_output(
    highlight_name_or_path: Optional[str],
    source_video_path: str,
    output_dir: str,
) -> Optional[str]:
    """Move highlight video into unified game_analysis output dir."""
    if not highlight_name_or_path:
        return None
    raw = str(highlight_name_or_path).strip()
    if not raw:
        return None

    if os.path.isabs(raw):
        src = raw
    else:
        base = os.path.basename(raw)
        candidates = [
            os.path.join(os.path.dirname(source_video_path), base),
            os.path.join(output_dir, base),
            os.path.join(os.path.dirname(__file__), "uploads", "game_analysis", "raw", base),
        ]
        src = next((p for p in candidates if os.path.exists(p)), "")
        if not src:
            return base

    if not os.path.exists(src):
        return os.path.basename(raw)

    os.makedirs(output_dir, exist_ok=True)
    dst_name = os.path.basename(src)
    dst = os.path.join(output_dir, dst_name)
    if os.path.abspath(src) != os.path.abspath(dst):
        if os.path.exists(dst):
            stem, ext = os.path.splitext(dst_name)
            dst_name = f"{stem}_{uuid.uuid4().hex[:6]}{ext}"
            dst = os.path.join(output_dir, dst_name)
        try:
            shutil.move(src, dst)
        except Exception:
            try:
                shutil.copy2(src, dst)
            except Exception:
                return os.path.basename(src)
    return dst_name


def process_game_analysis_video(
    video_path: str,
    output_dir: str,
    enable_possession: bool = True,
    enable_highlight: bool = True,
    highlight_mode: str = "action",
    enable_trajectory: bool = False,
    query_image_path: Optional[str] = None,
    reid_engine: Any = None,
    rim_calibration: Optional[Dict[str, float]] = None,
    cancel_check: CancelFn = lambda: False,
    progress_callback: ProgressFn = lambda p: None,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Cannot open video: {video_path}"}
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration_s = float(total_frames / max(1.0, fps))

    _ensure_ultralytics_writable_config()
    possession_data = {"timeline": [], "holder_stats": [], "team_possession_ratio": {}, "events": []}
    tracking_source = "none"
    tracking_errors: List[str] = []
    tp_metrics: Dict[str, Any] = {}
    tp_annotated_video_url = ""
    tp_annotated_video_path = ""
    tp_action_clips: List[Dict[str, Any]] = []
    tp_ball_points: List[Dict[str, Any]] = []
    tp_rim_hint: Optional[Dict[str, float]] = None

    should_run_tracking = bool(enable_possession or (enable_highlight and highlight_mode == "action"))
    use_third_party = str(os.getenv("GA_USE_THIRD_PARTY", "1")).strip().lower() not in {"0", "false", "off", "no"}
    if should_run_tracking and use_third_party:
        if duration_s >= 60.0:
            os.environ.setdefault("GA_BA_FRAME_STRIDE", "3")
        elif duration_s >= 50.0:
            os.environ.setdefault("GA_BA_FRAME_STRIDE", "2")
        os.environ.setdefault("GA_BA_MAX_WIDTH", "960")
        tp = run_third_party_basketball_analysis(
            video_path=video_path,
            output_dir=output_dir,
            cancel_check=cancel_check,
            progress_callback=lambda p: _safe_progress(progress_callback, 2 + 44 * (float(p) / 100.0)),
        )
        if tp.get("cancelled"):
            return {"cancelled": True}
        if tp.get("ok"):
            tracking_source = str(tp.get("source") or "basketball_analysis_repo")
            possession_data = {
                "timeline": ((tp.get("possession") or {}).get("timeline") or []),
                "holder_stats": ((tp.get("possession") or {}).get("holder_stats") or []),
                "team_possession_ratio": ((tp.get("possession") or {}).get("team_possession_ratio") or {}),
                "events": list(tp.get("events") or []),
            }
            tp_action_clips = list(tp.get("clips") or [])
            tp_metrics = dict(tp.get("metrics") or {})
            tp_annotated_video_url = str(((tp.get("artifacts") or {}).get("annotated_video_url") or "")).strip()
            tp_annotated_video_path = str(((tp.get("artifacts") or {}).get("annotated_video_path") or "")).strip()
            tp_ball_points = list(tp.get("ball_points") or [])
            raw_rim_hint = tp.get("rim_hint")
            if isinstance(raw_rim_hint, dict):
                try:
                    tp_rim_hint = {
                        "cx": float(raw_rim_hint.get("cx")),
                        "cy": float(raw_rim_hint.get("cy")),
                        "r": float(raw_rim_hint.get("r")),
                    }
                except Exception:
                    tp_rim_hint = None
            _safe_progress(progress_callback, 46)
        else:
            tracking_errors.append(str(tp.get("error") or "third_party_tracking_failed"))
            missing_models = list(tp.get("missing_models") or [])
            if missing_models:
                tracking_errors.append("missing_models=" + ",".join(str(x) for x in missing_models))

    if should_run_tracking and tracking_source == "none":
        try:
            from ultralytics import YOLO  # type: ignore

            backend_dir = os.path.dirname(__file__)
            model_hint = os.getenv("GA_TRACK_MODEL", "yolo11l.pt").strip() or "yolo11l.pt"
            model_path = model_hint if os.path.isabs(model_hint) else (os.path.join(backend_dir, model_hint) if os.path.exists(os.path.join(backend_dir, model_hint)) else model_hint)
            model = YOLO(model_path)
            sample_interval = max(1, int(os.getenv("GA_SAMPLE_INTERVAL", "3")))
            conf_th = float(os.getenv("GA_TRACK_CONF", "0.22"))
            imgsz = int(os.getenv("GA_TRACK_IMGSZ", "960"))

            cap = cv2.VideoCapture(video_path)
            tracker = _TrackManager()
            frame_idx = 0
            frame_indices: List[int] = []
            player_tracks: List[Dict[int, Dict[str, Any]]] = []
            ball_tracks: List[Dict[int, Dict[str, Any]]] = []
            while True:
                if cancel_check():
                    cap.release()
                    return {"cancelled": True}
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % sample_interval != 0:
                    frame_idx += 1
                    continue
                res = model.predict(frame, conf=conf_th, imgsz=imgsz, classes=[0, 32], verbose=False)
                p_dets: List[Tuple[float, float, float, float]] = []
                b_best = None
                b_conf = -1.0
                if res and res[0].boxes is not None:
                    boxes = res[0].boxes
                    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
                    conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
                    cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
                    for i in range(len(xyxy)):
                        bb = tuple(float(v) for v in xyxy[i].tolist())
                        if int(cls[i]) == 0:
                            p_dets.append(bb)
                        elif int(cls[i]) == 32 and float(conf[i]) > b_conf:
                            b_best = bb
                            b_conf = float(conf[i])
                player_tracks.append(tracker.update(p_dets))
                ball_tracks.append({1: {"bbox": [float(v) for v in b_best], "confidence": round(b_conf, 4)}} if b_best else {})
                frame_indices.append(int(frame_idx))
                _safe_progress(progress_callback, 2 + 44 * (frame_idx / max(1, total_frames)))
                frame_idx += 1
            cap.release()

            ball_tracks = _interpolate_ball(ball_tracks)
            possession = _detect_possession(player_tracks, ball_tracks)
            events = _pass_interception_events(possession, frame_indices, fps)
            summary = _possession_summary(possession, frame_indices, fps)
            possession_data = {
                "timeline": summary["timeline"],
                "holder_stats": summary["holder_stats"],
                "team_possession_ratio": summary["team_possession_ratio"],
                "events": events,
                "sample_interval": sample_interval,
                "model": os.path.basename(model_path),
            }
            tracking_source = "internal_yolo_fallback"
        except Exception as exc:
            tracking_errors.append(f"internal_tracking_failed: {exc}")

    trajectory_summary: Dict[str, Any] = {"enabled": False}
    shot_events: List[Dict[str, Any]] = []
    effective_rim_calibration = rim_calibration
    if effective_rim_calibration is None and tp_rim_hint is not None:
        effective_rim_calibration = tp_rim_hint
    if enable_trajectory:
        # Fast defaults for unified game-analysis tasks (avoid hour-level latency).
        ga_traj_fast = str(os.getenv("GA_TRAJECTORY_FAST", "1")).strip().lower() not in {"0", "false", "off", "no"}
        traj_fast_model = "yolo11n.pt" if os.path.exists(os.path.join(os.path.dirname(__file__), "yolo11n.pt")) else "yolo11m.pt"
        if ga_traj_fast and duration_s >= 60.0:
            os.environ.setdefault("TRAJ_BALL_MODEL", traj_fast_model)
            os.environ.setdefault("TRAJ_YOLO_IMGSZ", "448")
            os.environ.setdefault("TRAJ_POSE_IMGSZ", "320")
            os.environ.setdefault("TRAJ_POSE_STRIDE", "6")
            os.environ.setdefault("TRAJ_DET_STRIDE", "4")
        elif ga_traj_fast and duration_s >= 45.0:
            os.environ.setdefault("TRAJ_BALL_MODEL", traj_fast_model)
            os.environ.setdefault("TRAJ_YOLO_IMGSZ", "512")
            os.environ.setdefault("TRAJ_POSE_IMGSZ", "384")
            os.environ.setdefault("TRAJ_POSE_STRIDE", "4")
            os.environ.setdefault("TRAJ_DET_STRIDE", "3")
        else:
            os.environ.setdefault("TRAJ_BALL_MODEL", "yolo11m.pt")
            os.environ.setdefault("TRAJ_YOLO_IMGSZ", "640")
            os.environ.setdefault("TRAJ_POSE_IMGSZ", "512")
            os.environ.setdefault("TRAJ_POSE_STRIDE", "2")
            os.environ.setdefault("TRAJ_DET_STRIDE", "2")

        os.environ.setdefault("TRAJ_YOLO_HALF", "1")
        os.environ.setdefault("TRAJ_POSE_HALF", "1")
        os.environ.setdefault("TRAJ_LOWLIGHT_DUAL_BRANCH", "0")
        os.environ.setdefault("TRAJ_YOLO_DEVICE", "0")
        os.environ.setdefault("TRAJ_POSE_DEVICE", os.environ.get("TRAJ_YOLO_DEVICE", "0"))
        # Broadcast game-analysis defaults: slightly relaxed trigger to reduce missed shots.
        os.environ.setdefault("TRAJ_SHOT_GATE_STRICT", "balanced")
        os.environ.setdefault("TRAJ_SHOT_REQUIRE_RELEASE_TRANSITION", "0")

        traj_started = time.time()
        traj_timeout_s = float(os.getenv("GA_TRAJECTORY_TIMEOUT_S", "300") or 300)

        def _traj_cancel_check() -> bool:
            if cancel_check():
                return True
            if traj_timeout_s > 0 and (time.time() - traj_started) >= traj_timeout_s:
                return True
            return False

        tr = process_trajectory_video(
            video_path,
            output_dir=os.path.join(output_dir, "trajectory"),
            rim_calibration=effective_rim_calibration,
            cancel_check=_traj_cancel_check,
            progress_callback=lambda p: _safe_progress(progress_callback, 46 + 34 * (float(p) / 100.0)),
        )
        if tr.get("cancelled"):
            if cancel_check():
                return {"cancelled": True}
            trajectory_summary = {"enabled": True, "error": "trajectory_timeout"}
        elif tr.get("error"):
            trajectory_summary = {"enabled": True, "error": str(tr.get("error"))}
        else:
            ta = tr.get("analysis") or {}
            shot_events = list(ta.get("shot_events") or [])
            annotated_video_url = ((tr.get("artifacts") or {}).get("annotated_video_url") or "")
            if annotated_video_url.startswith("/uploads/trajectory/"):
                filename = os.path.basename(annotated_video_url.split("?", 1)[0])
                if filename:
                    local_candidate = os.path.join(output_dir, "trajectory", filename)
                    if os.path.exists(local_candidate):
                        annotated_video_url = f"/uploads/game_analysis/trajectory/{filename}"
            trajectory_summary = {
                "enabled": True,
                "shot_prediction": ta.get("shot_prediction") or {},
                "shot_events": shot_events,
                "predicted_trajectory_points": ta.get("predicted_trajectory_points") or [],
                "annotated_video_url": annotated_video_url,
                "metrics": tr.get("metrics") or {},
                "rim_source": "manual" if rim_calibration else ("auto_third_party" if tp_rim_hint else "missing"),
                "rim_calibration": effective_rim_calibration or {},
            }

    clips: List[Dict[str, Any]] = []
    highlight_video = None
    highlight_video_url = ""
    highlight_errors: List[str] = []
    if enable_highlight:
        if highlight_mode == "player":
            if not query_image_path or not os.path.exists(query_image_path):
                highlight_errors.append("player_highlight_requires_query_image")
            elif reid_engine is None:
                highlight_errors.append("reid_engine_unavailable")
            else:
                vr = process_player_highlight_video(video_path, query_image_path, reid_engine, cancel_check=cancel_check, progress_callback=lambda p: _safe_progress(progress_callback, 80 + 20 * (float(p) / 100.0)))
                if vr.get("cancelled"):
                    return {"cancelled": True}
                if vr.get("error"):
                    highlight_errors.append(str(vr.get("error")))
                else:
                    segs = (vr.get("analysis") or {}).get("segments") or []
                    for i, s in enumerate(segs, start=1):
                        st = float(s.get("start_time", 0.0))
                        et = float(s.get("end_time", st))
                        clips.append({"id": i, "start_time": round(st, 3), "end_time": round(et, 3), "duration": round(max(0.0, et - st), 3), "label": "player_focus", "labels": ["player_focus"], "source_event_ids": [], "video_clip_url": None, "score": float(s.get("best_similarity", 0.0) or 0.0)})
                    highlight_video = _relocate_highlight_to_output(
                        (vr.get("analysis") or {}).get("highlight_video"),
                        source_video_path=video_path,
                        output_dir=output_dir,
                    )
                    if highlight_video:
                        highlight_video_url = f"/uploads/game_analysis/{highlight_video}"
        else:
            raw_action_clips = tp_action_clips or _action_clips(list(possession_data.get("events") or []), shot_events)
            clips = _merge_highlight_clips(raw_action_clips, merge_gap_s=0.15)
            highlight_video = _render_highlight_video(video_path, clips, output_dir, "ga_action_highlight")
            if highlight_video:
                highlight_video_url = f"/uploads/game_analysis/{highlight_video}"
            elif tp_annotated_video_url:
                # Fallback only when clip render failed.
                highlight_video_url = tp_annotated_video_url
                highlight_errors.append("action_highlight_render_fallback_tracking_video")

    # Build trajectory overlay from third-party tracked basketball points when needed.
    if enable_trajectory and tp_ball_points and tp_annotated_video_url:
        traj_video_missing = not str(trajectory_summary.get("annotated_video_url") or "").strip()
        no_shot_events = int(len(shot_events)) == 0
        no_pred_points = int(len(trajectory_summary.get("predicted_trajectory_points") or [])) == 0
        if traj_video_missing or no_shot_events or no_pred_points:
            base_path = tp_annotated_video_path
            if not base_path:
                base_name = os.path.basename(tp_annotated_video_url.split("?", 1)[0])
                base_path = os.path.join(output_dir, base_name)
            fallback_name = _render_tracking_trajectory_video(
                base_video_path=base_path,
                output_dir=output_dir,
                prefix="ga_traj_from_tracking",
                ball_points=tp_ball_points,
                highlight_clips=None,
                draw_trajectory=True,
                draw_prediction=True,
            )
            if fallback_name:
                fallback_url = f"/uploads/game_analysis/{fallback_name}"
                trajectory_summary["annotated_video_url"] = fallback_url
                trajectory_summary["source"] = "tracking_ball_fallback"
                if no_shot_events:
                    shot_prediction = dict(trajectory_summary.get("shot_prediction") or {})
                    shot_prediction["reason"] = shot_prediction.get("reason") or "shot_event_not_triggered_tracking_fallback"
                    trajectory_summary["shot_prediction"] = shot_prediction

    # Build a composite video that includes enabled modules on top of possession/tracking view.
    composite_video_url = ""
    if tp_ball_points and tp_annotated_video_url and (enable_possession or enable_highlight or enable_trajectory):
        base_path = tp_annotated_video_path
        if not base_path:
            base_name = os.path.basename(tp_annotated_video_url.split("?", 1)[0])
            base_path = os.path.join(output_dir, base_name)
        composite_name = _render_tracking_trajectory_video(
            base_video_path=base_path,
            output_dir=output_dir,
            prefix="ga_composite",
            ball_points=tp_ball_points,
            highlight_clips=clips if enable_highlight else None,
            draw_trajectory=bool(enable_trajectory),
            draw_prediction=bool(enable_trajectory),
        )
        if composite_name:
            composite_video_url = f"/uploads/game_analysis/{composite_name}"

    _safe_progress(progress_callback, 100)

    analysis_bundle = {
        "possession": {
            "timeline": possession_data.get("timeline") or [],
            "holder_stats": possession_data.get("holder_stats") or [],
            "team_possession_ratio": possession_data.get("team_possession_ratio") or {},
        },
        "events": possession_data.get("events") or [],
        "highlights": {"enabled": bool(enable_highlight), "mode": str(highlight_mode), "clips": clips, "highlight_video": highlight_video, "errors": highlight_errors},
        "trajectory": trajectory_summary,
        "tracking": {
            "source": tracking_source,
            "errors": tracking_errors,
            "metrics": tp_metrics,
            "annotated_video_url": tp_annotated_video_url,
            "rim_hint": tp_rim_hint or {},
            "ball_points_count": int(len(tp_ball_points)),
        },
    }

    raw_video_url = ""
    uploads_root = os.path.join(os.path.dirname(__file__), "uploads")
    try:
        rel = os.path.relpath(os.path.abspath(video_path), os.path.abspath(uploads_root))
        if not rel.startswith(".."):
            raw_video_url = "/uploads/" + rel.replace("\\", "/")
    except Exception:
        raw_video_url = ""
    trajectory_video_url = str(trajectory_summary.get("annotated_video_url", "") or "")
    # Composite video is the default when available.
    if composite_video_url:
        analysis_video_url = composite_video_url
    elif enable_trajectory and trajectory_video_url:
        analysis_video_url = trajectory_video_url
    elif enable_highlight and highlight_video_url:
        analysis_video_url = highlight_video_url
    elif tp_annotated_video_url:
        analysis_video_url = tp_annotated_video_url
    else:
        analysis_video_url = raw_video_url

    return {
        "task_type": "game_analysis",
        "video_info": {
            "filename": os.path.basename(video_path),
            "fps": round(float(fps), 3),
            "total_frames": int(total_frames),
            "duration": round(float(total_frames / max(1.0, fps)), 3),
            "resolution": f"{width}x{height}",
        },
        "analysis_bundle": analysis_bundle,
        "analysis": {
            "segments": clips,
            "highlight_video": highlight_video,
            "shot_events": shot_events,
            "shot_prediction": trajectory_summary.get("shot_prediction", {}),
            "predicted_trajectory_points": trajectory_summary.get("predicted_trajectory_points", []),
        },
        "metrics": {
            "event_count": int(len(analysis_bundle["events"])),
            "highlight_clip_count": int(len(clips)),
            "shot_count": int(len(shot_events)),
            "tracking_error_count": int(len(tracking_errors)),
            "tracking_source": tracking_source,
        },
        "artifacts": {
            "composite_video_url": composite_video_url,
            "highlight_video_url": highlight_video_url,
            "trajectory_video_url": trajectory_video_url,
            "tracking_video_url": tp_annotated_video_url,
            "analysis_video_url": analysis_video_url,
            "raw_video_url": raw_video_url,
        },
        "options": {
            "enable_possession": bool(enable_possession),
            "enable_highlight": bool(enable_highlight),
            "highlight_mode": str(highlight_mode),
            "enable_trajectory": bool(enable_trajectory),
        },
    }
