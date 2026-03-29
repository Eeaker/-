from __future__ import annotations

import os
import sys
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2

CancelFn = Callable[[], bool]
ProgressFn = Callable[[int], None]


MODEL_LINKS = {
    "ball_detector_model.pt": "https://drive.google.com/file/d/1KejdrcEnto2AKjdgdo1U1syr5gODp6EL/view?usp=sharing",
    "court_keypoint_detector.pt": "https://drive.google.com/file/d/1nGoG-pUkSg4bWAUIeQ8aN6n7O1fOkXU0/view?usp=sharing",
    "player_detector.pt": "https://drive.google.com/file/d/1fVBLZtPy9Yu6Tf186oS4siotkioHBLHy/view?usp=sharing",
}


def _default_repo_path() -> str:
    return os.getenv(
        "BA_REPO_PATH",
        r"C:\Users\23159\Downloads\TransReID-master\_tmp_repo_basketball_analysis",
    )


def _repo_models(repo_path: str) -> Dict[str, str]:
    models_dir = os.path.join(repo_path, "models")
    return {
        "ball_detector_model.pt": os.path.join(models_dir, "ball_detector_model.pt"),
        "court_keypoint_detector.pt": os.path.join(models_dir, "court_keypoint_detector.pt"),
        "player_detector.pt": os.path.join(models_dir, "player_detector.pt"),
    }


def _check_models_ready(repo_path: str) -> Tuple[bool, List[str]]:
    missing: List[str] = []
    for name, path in _repo_models(repo_path).items():
        if not os.path.exists(path) or os.path.getsize(path) <= 0:
            missing.append(name)
    return (len(missing) == 0), missing


def _read_sampled_frames(
    video_path: str,
    max_width: int,
    frame_stride: int,
    cancel_check: CancelFn,
    progress_callback: ProgressFn,
) -> Tuple[List[Any], List[int], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], [], 0.0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: List[Any] = []
    frame_map: List[int] = []
    idx = 0
    while True:
        if cancel_check():
            cap.release()
            return [], [], fps
        ok, frame = cap.read()
        if not ok:
            break
        if frame_stride > 1 and idx % frame_stride != 0:
            idx += 1
            continue
        h, w = frame.shape[:2]
        if max_width > 0 and w > max_width:
            ratio = max_width / float(max(1, w))
            nh = max(2, int(h * ratio))
            frame = cv2.resize(frame, (max_width, nh), interpolation=cv2.INTER_AREA)
        frames.append(frame)
        frame_map.append(idx)
        idx += 1
        if total > 0 and idx % 60 == 0:
            progress_callback(int(min(25, 25 * (idx / max(1, total)))))
    cap.release()
    return frames, frame_map, fps


def _write_video_mp4(frames: List[Any], fps: float, output_path: str) -> bool:
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = None
    for codec in ("avc1", "H264", "X264", "mp4v"):
        try:
            candidate = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*codec), max(1.0, fps), (w, h))
        except Exception:
            candidate = None
        if candidate is not None and candidate.isOpened():
            writer = candidate
            break
        try:
            if candidate is not None:
                candidate.release()
        except Exception:
            pass
    if writer is None:
        return False
    for frame in frames:
        writer.write(frame)
    writer.release()
    return True


def _build_possession_and_events(
    frame_map: List[int],
    fps: float,
    player_assignment: List[Dict[int, int]],
    ball_acquisition: List[int],
    passes: List[int],
    interceptions: List[int],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    timeline: List[Dict[str, Any]] = []
    holder_secs: Dict[int, float] = {}
    team_secs: Dict[int, float] = {1: 0.0, 2: 0.0}

    i = 0
    while i < len(ball_acquisition):
        holder = int(ball_acquisition[i]) if i < len(ball_acquisition) else -1
        j = i
        while j + 1 < len(ball_acquisition) and ball_acquisition[j + 1] == holder:
            j += 1
        if holder != -1 and i < len(frame_map) and j < len(frame_map):
            sf = int(frame_map[i])
            ef = int(frame_map[j])
            dur = float((ef - sf + 1) / max(1.0, fps))
            team_id = int((player_assignment[i] or {}).get(holder, 1))
            timeline.append(
                {
                    "start_frame": sf,
                    "end_frame": ef,
                    "start_time": round(sf / max(1.0, fps), 3),
                    "end_time": round(ef / max(1.0, fps), 3),
                    "duration_s": round(dur, 3),
                    "player_id": holder,
                    "team_id": team_id,
                }
            )
            holder_secs[holder] = holder_secs.get(holder, 0.0) + dur
            team_secs[team_id] = team_secs.get(team_id, 0.0) + dur
        i = j + 1

    total_secs = max(1e-6, sum(holder_secs.values()))
    holder_stats = [
        {"player_id": int(pid), "duration_s": round(secs, 3), "ratio": round(secs / total_secs, 4)}
        for pid, secs in sorted(holder_secs.items(), key=lambda x: x[1], reverse=True)
    ]
    team_total = max(1e-6, team_secs.get(1, 0.0) + team_secs.get(2, 0.0))
    team_ratio = {
        "1": round(team_secs.get(1, 0.0) / team_total, 4),
        "2": round(team_secs.get(2, 0.0) / team_total, 4),
    }

    events: List[Dict[str, Any]] = []
    clips: List[Dict[str, Any]] = []
    eid = 1
    for idx, team in enumerate(passes):
        if idx >= len(frame_map) or int(team) not in {1, 2}:
            continue
        frame = int(frame_map[idx])
        t = round(frame / max(1.0, fps), 3)
        events.append(
            {
                "event_id": eid,
                "type": "pass",
                "frame": frame,
                "time": t,
                "team_id": int(team),
                "confidence": 0.7,
            }
        )
        clips.append(
            {
                "id": eid,
                "label": "pass",
                "labels": ["pass"],
                "start_time": max(0.0, t - 1.0),
                "end_time": t + 1.0,
                "duration": round(2.0, 3),
                "source_event_ids": [eid],
                "video_clip_url": None,
            }
        )
        eid += 1

    for idx, team in enumerate(interceptions):
        if idx >= len(frame_map) or int(team) not in {1, 2}:
            continue
        frame = int(frame_map[idx])
        t = round(frame / max(1.0, fps), 3)
        events.append(
            {
                "event_id": eid,
                "type": "interception",
                "frame": frame,
                "time": t,
                "team_id": int(team),
                "confidence": 0.7,
            }
        )
        clips.append(
            {
                "id": eid,
                "label": "interception",
                "labels": ["interception"],
                "start_time": max(0.0, t - 1.0),
                "end_time": t + 1.0,
                "duration": round(2.0, 3),
                "source_event_ids": [eid],
                "video_clip_url": None,
            }
        )
        eid += 1

    possession = {
        "timeline": timeline,
        "holder_stats": holder_stats,
        "team_possession_ratio": team_ratio,
    }
    return possession, events, clips


def _estimate_rim_hint(player_tracker: Any, frames: List[Any]) -> Optional[Dict[str, float]]:
    if not frames:
        return None
    max_samples = max(1, min(16, len(frames)))
    step = max(1, len(frames) // max_samples)
    sampled = frames[::step][:max_samples]
    try:
        detections = player_tracker.detect_frames(sampled)
    except Exception:
        return None

    rim_boxes: List[Tuple[float, float, float, float, float]] = []
    for det in detections or []:
        try:
            names = det.names or {}
            cls_inv = {str(v).strip().lower(): int(k) for k, v in names.items()}
            hoop_cls = cls_inv.get("hoop")
            if hoop_cls is None or det.boxes is None:
                continue
            boxes = det.boxes
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
            conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
            cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls
            best_i = -1
            best_conf = -1.0
            for i in range(len(xyxy)):
                if int(cls[i]) != int(hoop_cls):
                    continue
                c = float(conf[i])
                if c > best_conf:
                    best_conf = c
                    best_i = i
            if best_i >= 0:
                x1, y1, x2, y2 = [float(v) for v in xyxy[best_i].tolist()]
                rim_boxes.append((x1, y1, x2, y2, best_conf))
        except Exception:
            continue

    if not rim_boxes:
        return None
    rim_boxes.sort(key=lambda x: x[4], reverse=True)
    top = rim_boxes[: max(1, min(8, len(rim_boxes)))]
    cx = sum((b[0] + b[2]) * 0.5 for b in top) / len(top)
    cy = sum((b[1] + b[3]) * 0.5 for b in top) / len(top)
    r = sum(max(6.0, max(b[2] - b[0], b[3] - b[1]) * 0.35) for b in top) / len(top)
    conf = sum(b[4] for b in top) / len(top)
    return {
        "cx": round(float(cx), 2),
        "cy": round(float(cy), 2),
        "r": round(float(r), 2),
        "confidence": round(float(conf), 4),
        "source": "third_party_hoop_detection",
    }


def _extract_ball_points(ball_tracks: List[Dict[int, Dict[str, Any]]], frame_map: List[int], fps: float, out_fps: float) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for i, tr in enumerate(ball_tracks):
        bb = (tr or {}).get(1, {}).get("bbox")
        if not bb or len(bb) < 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bb[:4]]
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        orig_frame = int(frame_map[i]) if i < len(frame_map) else int(i)
        points.append(
            {
                "sampled_index": int(i),
                "sampled_time": round(float(i / max(1.0, out_fps)), 4),
                "frame": orig_frame,
                "time": round(float(orig_frame / max(1.0, fps)), 4),
                "x": round(float(cx), 3),
                "y": round(float(cy), 3),
            }
        )
    return points


def run_third_party_basketball_analysis(
    video_path: str,
    output_dir: str,
    cancel_check: CancelFn = lambda: False,
    progress_callback: ProgressFn = lambda p: None,
) -> Dict[str, Any]:
    repo_path = _default_repo_path()
    if not os.path.isdir(repo_path):
        return {"ok": False, "error": f"third_party_repo_not_found: {repo_path}"}

    models_ready, missing = _check_models_ready(repo_path)
    if not models_ready:
        return {
            "ok": False,
            "error": "third_party_models_missing",
            "missing_models": missing,
            "download_links": {name: MODEL_LINKS.get(name, "") for name in missing},
        }

    # Reduce OpenMP collisions in mixed scientific stacks.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    try:
        from trackers import BallTracker, PlayerTracker  # type: ignore
        from ball_aquisition import BallAquisitionDetector  # type: ignore
        from pass_and_interception_detector import PassAndInterceptionDetector  # type: ignore
        from drawers import BallTracksDrawer, FrameNumberDrawer, PlayerTracksDrawer  # type: ignore
    except Exception as exc:
        return {"ok": False, "error": f"third_party_import_failed: {exc}"}

    models = _repo_models(repo_path)
    max_width = int(os.getenv("GA_BA_MAX_WIDTH", "960") or 960)
    frame_stride = int(os.getenv("GA_BA_FRAME_STRIDE", "1") or 1)
    frame_stride = max(1, frame_stride)

    frames, frame_map, fps = _read_sampled_frames(
        video_path=video_path,
        max_width=max_width,
        frame_stride=frame_stride,
        cancel_check=cancel_check,
        progress_callback=progress_callback,
    )
    if not frames:
        return {"ok": False, "error": "third_party_no_frames"}
    if cancel_check():
        return {"ok": False, "cancelled": True}

    progress_callback(28)
    stub_dir = os.path.join(output_dir, "stubs", "basketball_analysis")
    os.makedirs(stub_dir, exist_ok=True)
    key = f"{os.path.basename(video_path)}_{max_width}_{frame_stride}"
    key = key.replace(" ", "_")
    player_stub = os.path.join(stub_dir, f"{key}_player.pkl")
    ball_stub = os.path.join(stub_dir, f"{key}_ball.pkl")

    try:
        player_tracker = PlayerTracker(models["player_detector.pt"])
        ball_tracker = BallTracker(models["ball_detector_model.pt"])
        player_tracks = player_tracker.get_object_tracks(frames, read_from_stub=True, stub_path=player_stub)
        progress_callback(55)
        ball_tracks = ball_tracker.get_object_tracks(frames, read_from_stub=True, stub_path=ball_stub)
    except Exception as exc:
        return {"ok": False, "error": f"third_party_tracking_failed: {exc}"}
    rim_hint = _estimate_rim_hint(player_tracker, frames)

    if cancel_check():
        return {"ok": False, "cancelled": True}

    try:
        ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
        ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)
    except Exception:
        pass

    player_assignment: List[Dict[int, int]] = []
    for frame_tracks in player_tracks:
        frame_assignment: Dict[int, int] = {}
        for pid in frame_tracks.keys():
            pid_int = int(pid)
            frame_assignment[pid_int] = 1 if (pid_int % 2 == 1) else 2
        player_assignment.append(frame_assignment)

    ball_aquisition_detector = BallAquisitionDetector()
    ball_acquisition = ball_aquisition_detector.detect_ball_possession(player_tracks, ball_tracks)
    pass_detector = PassAndInterceptionDetector()
    passes = pass_detector.detect_passes(ball_acquisition, player_assignment)
    interceptions = pass_detector.detect_interceptions(ball_acquisition, player_assignment)

    progress_callback(72)
    try:
        player_drawer = PlayerTracksDrawer()
        ball_drawer = BallTracksDrawer()
        frame_drawer = FrameNumberDrawer()
        output_frames = player_drawer.draw(frames, player_tracks, player_assignment, ball_acquisition)
        output_frames = ball_drawer.draw(output_frames, ball_tracks)
        output_frames = frame_drawer.draw(output_frames)
    except Exception as exc:
        return {"ok": False, "error": f"third_party_draw_failed: {exc}"}

    if cancel_check():
        return {"ok": False, "cancelled": True}

    output_name = f"ga_tp_{uuid.uuid4().hex[:10]}.mp4"
    output_path = os.path.join(output_dir, output_name)
    out_fps = max(1.0, float(fps / max(1, frame_stride)))
    if not _write_video_mp4(output_frames, fps=out_fps, output_path=output_path):
        return {"ok": False, "error": "third_party_video_write_failed"}

    possession, events, clips = _build_possession_and_events(
        frame_map=frame_map,
        fps=fps,
        player_assignment=player_assignment,
        ball_acquisition=ball_acquisition,
        passes=passes,
        interceptions=interceptions,
    )
    ball_points = _extract_ball_points(ball_tracks=ball_tracks, frame_map=frame_map, fps=fps, out_fps=out_fps)

    progress_callback(80)
    return {
        "ok": True,
        "source": "basketball_analysis_repo",
        "possession": possession,
        "events": events,
        "clips": clips,
        "artifacts": {
            "annotated_video_filename": output_name,
            "annotated_video_url": f"/uploads/game_analysis/{output_name}",
            "annotated_video_path": output_path,
        },
        "metrics": {
            "sampled_frames": int(len(frames)),
            "frame_stride": int(frame_stride),
            "max_width": int(max_width),
            "event_count": int(len(events)),
            "ball_points": int(len(ball_points)),
        },
        "rim_hint": rim_hint,
        "ball_points": ball_points,
        "sampled_fps": round(float(out_fps), 4),
    }
