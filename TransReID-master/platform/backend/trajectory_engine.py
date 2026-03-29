from __future__ import annotations

import math
import os
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


CancelCheck = Callable[[], bool]
ProgressCallback = Callable[[int], None]
StateCallback = Callable[[Dict], None]


@dataclass
class Detection:
    x: float
    y: float
    radius: float
    confidence: float
    source: str
    motion_score: float
    bbox: Tuple[int, int, int, int]


@dataclass
class RimCalibration:
    cx: float
    cy: float
    r: float


def _read_diag_from_env(env_key: str, default: List[float], expected: int) -> np.ndarray:
    raw = os.getenv(env_key, "").strip()
    if not raw:
        return np.diag(np.asarray(default, dtype=np.float32))
    try:
        values = [float(x.strip()) for x in raw.split(",") if x.strip()]
        if len(values) != expected:
            return np.diag(np.asarray(default, dtype=np.float32))
        return np.diag(np.asarray(values, dtype=np.float32))
    except Exception:
        return np.diag(np.asarray(default, dtype=np.float32))


def _read_float_from_env(env_key: str, default: float, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
    raw = os.getenv(env_key, "").strip()
    if not raw:
        return float(default)
    try:
        val = float(raw)
    except Exception:
        return float(default)
    if min_value is not None:
        val = max(float(min_value), val)
    if max_value is not None:
        val = min(float(max_value), val)
    return float(val)


def _read_int_from_env(env_key: str, default: int, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    raw = os.getenv(env_key, "").strip()
    if not raw:
        return int(default)
    try:
        val = int(raw)
    except Exception:
        return int(default)
    if min_value is not None:
        val = max(int(min_value), val)
    if max_value is not None:
        val = min(int(max_value), val)
    return int(val)


def _read_bool_from_env(env_key: str, default: bool) -> bool:
    raw = os.getenv(env_key, "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "y", "on"}


def _resolve_ball_model_path(model_hint: Optional[str], backend_dir: str) -> str:
    hint = (model_hint or "").strip()
    if not hint:
        hint = "yolo11l.pt"
    if os.path.isabs(hint):
        return hint
    candidate_local = os.path.join(backend_dir, hint)
    if os.path.exists(candidate_local):
        return candidate_local
    return hint


class BallDetector:
    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        raise NotImplementedError

    @property
    def mode_name(self) -> str:
        raise NotImplementedError


class YOLOBallDetector(BallDetector):
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.22,
        imgsz: int = 960,
        min_aspect: float = 0.70,
        max_aspect: float = 1.40,
    ):
        self.conf_threshold = float(conf_threshold)
        self.imgsz = int(max(320, min(1600, imgsz)))
        self.min_aspect = float(max(0.2, min_aspect))
        self.max_aspect = float(max(self.min_aspect + 0.05, max_aspect))
        self.model_path = model_path
        self.device = os.getenv("TRAJ_YOLO_DEVICE", "0" if torch.cuda.is_available() else "cpu").strip() or "cpu"
        self.use_half = _read_bool_from_env("TRAJ_YOLO_HALF", self.device != "cpu")
        self._model = None
        self._error = ""
        try:
            from ultralytics import YOLO  # type: ignore

            fallback_model = os.path.join(os.path.dirname(__file__), "yolo11l.pt")
            model_to_load = model_path if os.path.exists(model_path) else fallback_model
            self._model = YOLO(model_to_load)
        except Exception as exc:  # pragma: no cover
            self._error = str(exc)

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def mode_name(self) -> str:
        return "yolo" if self.available else "fallback"

    @property
    def error(self) -> str:
        return self._error

    def detect(self, frame_bgr: np.ndarray, conf_override: Optional[float] = None) -> List[Detection]:
        if self._model is None:
            return []
        h, w = frame_bgr.shape[:2]
        min_r = max(2.0, min(h, w) * 0.0045)
        # Tighten maximum ball radius to reduce "head as ball" false positives.
        max_r = max(7.0, min(h, w) * 0.07)
        conf_th = float(np.clip(conf_override if conf_override is not None else self.conf_threshold, 0.01, 0.95))
        results = self._model.predict(
            source=frame_bgr,
            classes=[32],
            conf=conf_th,
            verbose=False,
            imgsz=self.imgsz,
            max_det=10,
            device=self.device,
            half=self.use_half,
        )
        out: List[Detection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                try:
                    xyxy = np.asarray(box.xyxy[0].cpu().numpy(), dtype=np.float32).reshape(-1)
                    if xyxy.size < 4:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in xyxy[:4]]
                    bw = max(1.0, x2 - x1)
                    bh = max(1.0, y2 - y1)
                    aspect = bw / max(1.0, bh)
                    if aspect < self.min_aspect or aspect > self.max_aspect:
                        continue
                    radius = float(min(bw, bh) * 0.5)
                    if radius < min_r or radius > max_r:
                        continue
                    conf_arr = np.asarray(box.conf[0].cpu().numpy(), dtype=np.float32).reshape(-1)
                    conf = float(conf_arr[0]) if conf_arr.size else 0.0
                    out.append(
                        Detection(
                            x=float((x1 + x2) * 0.5),
                            y=float((y1 + y2) * 0.5),
                            radius=radius,
                            confidence=conf,
                            source="yolo",
                            motion_score=0.0,
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                        )
                    )
                except Exception:
                    continue
        return out


class PoseShotTrigger:
    """Pose-based shot trigger helper (high precision by default)."""

    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12

    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.backend_dir = os.path.dirname(__file__)
        self.enabled = _read_bool_from_env("TRAJ_POSE_ENABLE", True)
        self.model_hint = os.getenv("TRAJ_POSE_MODEL", "yolo11s-pose.pt").strip() or "yolo11s-pose.pt"
        self.fallback_model_hint = "yolo11n-pose.pt"
        self.imgsz = _read_int_from_env("TRAJ_POSE_IMGSZ", 640, min_value=256, max_value=1280)
        self.conf = _read_float_from_env("TRAJ_POSE_CONF", 0.16, min_value=0.05, max_value=0.8)
        self.device = os.getenv("TRAJ_POSE_DEVICE", os.getenv("TRAJ_YOLO_DEVICE", "0" if torch.cuda.is_available() else "cpu")).strip() or "cpu"
        self.use_half = _read_bool_from_env("TRAJ_POSE_HALF", self.device != "cpu")
        self.infer_stride = _read_int_from_env("TRAJ_POSE_STRIDE", 2, min_value=1, max_value=6)
        self.hand_contact_px = _read_float_from_env(
            "TRAJ_POSE_HAND_CONTACT_PX",
            max(24.0, self.frame_height * 0.065),
            min_value=8.0,
            max_value=max(30.0, self.frame_height * 0.22),
        )
        self.release_separation_px = _read_float_from_env(
            "TRAJ_POSE_RELEASE_PX",
            max(30.0, self.frame_height * 0.105),
            min_value=14.0,
            max_value=max(40.0, self.frame_height * 0.35),
        )
        self.min_kpt_conf = _read_float_from_env("TRAJ_POSE_MIN_KPT_CONF", 0.18, min_value=0.01, max_value=0.9)
        self._model = None
        self._model_error = ""
        self.prev_near_hand = False
        self.prev_elbow_angle: Optional[float] = None
        self._infer_tick = -1
        self._last_signal: Optional[Dict[str, Any]] = None
        self._load_model(self.model_hint)

    @property
    def model_error(self) -> str:
        return self._model_error

    @property
    def available(self) -> bool:
        return bool(self._model is not None and self.enabled)

    def _load_model(self, hint: str) -> None:
        if not self.enabled:
            self._model = None
            self._model_error = "pose disabled"
            return
        try:
            from ultralytics import YOLO  # type: ignore

            model_to_load = hint
            if not os.path.isabs(model_to_load):
                local_candidate = os.path.join(self.backend_dir, model_to_load)
                if os.path.exists(local_candidate):
                    model_to_load = local_candidate
            self._model = YOLO(model_to_load)
            self._model_error = ""
            self.model_hint = hint
        except Exception as exc:  # pragma: no cover
            self._model = None
            self._model_error = str(exc)

    @staticmethod
    def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[float]:
        try:
            ba = a - b
            bc = c - b
            nba = float(np.linalg.norm(ba))
            nbc = float(np.linalg.norm(bc))
            if nba < 1e-4 or nbc < 1e-4:
                return None
            cosv = float(np.dot(ba, bc) / (nba * nbc))
            cosv = float(np.clip(cosv, -1.0, 1.0))
            return float(math.degrees(math.acos(cosv)))
        except Exception:
            return None

    def _parse_pose(self, result: Any) -> List[Tuple[np.ndarray, np.ndarray]]:
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None:
            return []
        xy = getattr(keypoints, "xy", None)
        if xy is None:
            return []
        try:
            xy_np = np.asarray(xy.cpu().numpy(), dtype=np.float32)
        except Exception:
            return []
        if xy_np.ndim == 2:
            xy_np = xy_np[None, ...]
        conf_attr = getattr(keypoints, "conf", None)
        conf_np: Optional[np.ndarray] = None
        if conf_attr is not None:
            try:
                conf_np = np.asarray(conf_attr.cpu().numpy(), dtype=np.float32)
            except Exception:
                conf_np = None
        people: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(xy_np.shape[0]):
            pts = np.asarray(xy_np[i], dtype=np.float32)
            if pts.ndim != 2 or pts.shape[0] < 13 or pts.shape[1] < 2:
                continue
            if conf_np is not None and conf_np.ndim >= 2 and i < conf_np.shape[0]:
                c = np.asarray(conf_np[i], dtype=np.float32).reshape(-1)
            else:
                c = np.ones((pts.shape[0],), dtype=np.float32)
            people.append((pts, c))
        return people

    def infer(self, frame_bgr: np.ndarray, ball_xy: Optional[Tuple[float, float]], frame_idx: Optional[int] = None) -> Dict[str, Any]:
        signal: Dict[str, Any] = {
            "model_available": self.available,
            "person_found": False,
            "wrist_above_shoulder": False,
            "elbow_extended": False,
            "release_transition": False,
            "ball_to_hand_px": None,
            "hand_side": None,
            "elbow_angle": None,
            "elbow_extend_trend": False,
            "joint_points": [],
            "upper_body_bbox": None,
            "person_bbox": None,
            "primary_wrist": None,
            "sampled": True,
        }
        if not self.available:
            return signal
        tick = int(frame_idx) if frame_idx is not None else (self._infer_tick + 1)
        self._infer_tick = tick
        if self.infer_stride > 1 and self._last_signal is not None and (tick % self.infer_stride) != 0:
            cached = dict(self._last_signal)
            cached["release_transition"] = False
            cached["sampled"] = False
            return cached
        try:
            results = self._model.predict(  # type: ignore[union-attr]
                source=frame_bgr,
                conf=self.conf,
                imgsz=self.imgsz,
                classes=[0],
                max_det=2,
                verbose=False,
                device=self.device,
                half=self.use_half,
            )
        except Exception as exc:
            msg = str(exc).lower()
            if ("out of memory" in msg or "cuda" in msg) and self.model_hint != self.fallback_model_hint:
                self._load_model(self.fallback_model_hint)
                if self.available:
                    return self.infer(frame_bgr, ball_xy, frame_idx=frame_idx)
            self._model_error = str(exc)
            return signal

        people: List[Tuple[np.ndarray, np.ndarray]] = []
        for result in results:
            people.extend(self._parse_pose(result))
        if not people:
            self.prev_near_hand = False
            self._last_signal = dict(signal)
            return signal

        signal["person_found"] = True
        ball_arr = np.asarray(ball_xy, dtype=np.float32) if ball_xy is not None else None
        best: Optional[Tuple[np.ndarray, np.ndarray, float, str]] = None
        for pts, confs in people:
            ls = pts[self.LEFT_SHOULDER]
            rs = pts[self.RIGHT_SHOULDER]
            le = pts[self.LEFT_ELBOW]
            re = pts[self.RIGHT_ELBOW]
            lw = pts[self.LEFT_WRIST]
            rw = pts[self.RIGHT_WRIST]
            left_ok = min(float(confs[self.LEFT_SHOULDER]), float(confs[self.LEFT_ELBOW]), float(confs[self.LEFT_WRIST])) >= self.min_kpt_conf
            right_ok = min(float(confs[self.RIGHT_SHOULDER]), float(confs[self.RIGHT_ELBOW]), float(confs[self.RIGHT_WRIST])) >= self.min_kpt_conf
            if not left_ok and not right_ok:
                continue
            wrist_above = bool((left_ok and lw[1] < ls[1]) or (right_ok and rw[1] < rs[1]))
            left_angle = self._angle_deg(ls, le, lw) if left_ok else None
            right_angle = self._angle_deg(rs, re, rw) if right_ok else None
            use_left = True
            elbow_angle = left_angle
            if right_ok and (not left_ok or (right_angle is not None and (left_angle is None or right_angle > left_angle))):
                use_left = False
                elbow_angle = right_angle
            elbow_extended = bool(elbow_angle is not None and elbow_angle >= 140.0)
            side = "left" if use_left else "right"
            if ball_arr is None:
                score = 0.35 * float(wrist_above) + 0.35 * float(elbow_extended) + 0.30 * (float(confs.mean()) if confs.size else 0.0)
                hand_dist = float("inf")
            else:
                hand = lw if use_left else rw
                hand_dist = float(np.linalg.norm(hand - ball_arr))
                proximity = 1.0 / (1.0 + hand_dist / max(1.0, self.hand_contact_px))
                score = 0.40 * float(wrist_above) + 0.35 * float(elbow_extended) + 0.25 * proximity
            candidate = (pts, confs, score, side)
            if best is None or score > best[2]:
                best = candidate

        if best is None:
            self.prev_near_hand = False
            return signal

        pts, confs, _, side = best
        if side == "left":
            shoulder = pts[self.LEFT_SHOULDER]
            elbow = pts[self.LEFT_ELBOW]
            wrist = pts[self.LEFT_WRIST]
        else:
            shoulder = pts[self.RIGHT_SHOULDER]
            elbow = pts[self.RIGHT_ELBOW]
            wrist = pts[self.RIGHT_WRIST]
        wrist_above_shoulder = bool(wrist[1] < shoulder[1] and float(confs.mean()) >= self.min_kpt_conf)
        elbow_angle = self._angle_deg(shoulder, elbow, wrist)
        elbow_extended = bool(elbow_angle is not None and elbow_angle >= 140.0)
        elbow_trend = False
        if elbow_angle is not None and self.prev_elbow_angle is not None:
            elbow_trend = bool(elbow_angle - self.prev_elbow_angle >= 6.5)
        if elbow_angle is not None:
            self.prev_elbow_angle = elbow_angle
        hand_dist = None
        if ball_arr is not None:
            hand_dist = float(np.linalg.norm(wrist - ball_arr))
            near_now = hand_dist <= self.hand_contact_px
            release_transition = bool(self.prev_near_hand and hand_dist >= self.release_separation_px)
            self.prev_near_hand = near_now
        else:
            release_transition = False
            self.prev_near_hand = False

        signal.update(
            {
                "wrist_above_shoulder": wrist_above_shoulder,
                "elbow_extended": elbow_extended,
                "release_transition": release_transition,
                "ball_to_hand_px": hand_dist,
                "hand_side": side,
                "elbow_angle": round(float(elbow_angle), 2) if elbow_angle is not None else None,
                "elbow_extend_trend": bool(elbow_trend),
                "primary_wrist": [round(float(wrist[0]), 2), round(float(wrist[1]), 2)],
            }
        )

        joint_ids = [
            self.LEFT_SHOULDER,
            self.RIGHT_SHOULDER,
            self.LEFT_ELBOW,
            self.RIGHT_ELBOW,
            self.LEFT_WRIST,
            self.RIGHT_WRIST,
            self.LEFT_HIP,
            self.RIGHT_HIP,
        ]
        joint_points: List[List[float]] = []
        for jid in joint_ids:
            if jid >= len(confs) or jid >= len(pts):
                continue
            if float(confs[jid]) < self.min_kpt_conf:
                continue
            joint_points.append([round(float(pts[jid][0]), 2), round(float(pts[jid][1]), 2)])
        signal["joint_points"] = joint_points
        body_points: List[List[float]] = []
        for idx in range(min(len(confs), len(pts))):
            if float(confs[idx]) < self.min_kpt_conf:
                continue
            body_points.append([round(float(pts[idx][0]), 2), round(float(pts[idx][1]), 2)])
        if body_points:
            body_np = np.asarray(body_points, dtype=np.float32)
            bx1, by1 = float(np.min(body_np[:, 0])), float(np.min(body_np[:, 1]))
            bx2, by2 = float(np.max(body_np[:, 0])), float(np.max(body_np[:, 1]))
            signal["person_bbox"] = [round(bx1, 2), round(by1, 2), round(bx2, 2), round(by2, 2)]
        if joint_points:
            pts_np = np.asarray(joint_points, dtype=np.float32)
            x1, y1 = float(np.min(pts_np[:, 0])), float(np.min(pts_np[:, 1]))
            x2, y2 = float(np.max(pts_np[:, 0])), float(np.max(pts_np[:, 1]))
            signal["upper_body_bbox"] = [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
        self._last_signal = dict(signal)
        return signal


class HSVBallCandidateDetector(BallDetector):
    """HSV + motion + shape candidate generator used as YOLO assist."""

    def __init__(self):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=360, varThreshold=20, detectShadows=False)

    @property
    def mode_name(self) -> str:
        return "hsv_assist"

    @staticmethod
    def _contour_circularity(contour: np.ndarray, area: float) -> float:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 1e-6:
            return 0.0
        return float(4.0 * math.pi * area / (perimeter * perimeter + 1e-6))

    @staticmethod
    def _dedupe(detections: List[Detection], max_dist_scale: float = 1.55) -> List[Detection]:
        if not detections:
            return []
        kept: List[Detection] = []
        for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
            duplicate = False
            for existed in kept:
                dist = math.hypot(det.x - existed.x, det.y - existed.y)
                if dist <= max(det.radius, existed.radius) * max_dist_scale:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(det)
        return kept

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        h, w = frame_bgr.shape[:2]
        min_r = max(2.0, min(h, w) * 0.0032)
        max_r = max(8.0, min(h, w) * 0.065)

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        orange_mask = cv2.bitwise_or(
            cv2.inRange(hsv, (5, 65, 48), (24, 255, 255)),
            cv2.inRange(hsv, (0, 45, 42), (8, 255, 255)),
        )
        white_mask = cv2.inRange(hsv, (0, 0, 175), (180, 80, 255))

        motion = self.bg.apply(frame_bgr)
        motion = cv2.GaussianBlur(motion, (5, 5), 0)
        _, motion = cv2.threshold(motion, 185, 255, cv2.THRESH_BINARY)
        motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 64, 148)

        orange_motion = cv2.bitwise_and(orange_mask, motion)
        white_motion = cv2.bitwise_and(white_mask, motion)
        contours_orange, _ = cv2.findContours(orange_motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_white, _ = cv2.findContours(white_motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_motion, _ = cv2.findContours(motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: List[Detection] = []

        def add_from_contours(contours: List[np.ndarray], source: str, color_weight: float) -> None:
            for contour in contours:
                area = float(cv2.contourArea(contour))
                if area < 6.0 or area > (h * w) * 0.005:
                    continue
                (cx, cy), radius = cv2.minEnclosingCircle(contour)
                if radius < min_r or radius > max_r:
                    continue
                circularity = self._contour_circularity(contour, area)
                if circularity < 0.06:
                    continue
                x1 = max(0, int(cx - radius))
                y1 = max(0, int(cy - radius))
                x2 = min(w - 1, int(cx + radius))
                y2 = min(h - 1, int(cy + radius))
                if x2 <= x1 or y2 <= y1:
                    continue
                motion_roi = motion[y1 : y2 + 1, x1 : x2 + 1]
                motion_ratio = float(np.mean(motion_roi > 0)) if motion_roi.size else 0.0
                if motion_ratio < 0.006:
                    continue
                edge_roi = edges[y1 : y2 + 1, x1 : x2 + 1]
                edge_ratio = float(np.mean(edge_roi > 0)) if edge_roi.size else 0.0
                conf = min(
                    0.86,
                    0.12
                    + 0.32 * circularity
                    + 0.30 * motion_ratio
                    + 0.18 * edge_ratio
                    + color_weight,
                )
                candidates.append(
                    Detection(
                        x=float(cx),
                        y=float(cy),
                        radius=float(radius),
                        confidence=float(conf),
                        source=source,
                        motion_score=float(motion_ratio),
                        bbox=(x1, y1, x2, y2),
                    )
                )

        add_from_contours(contours_orange, source="hsv_orange", color_weight=0.12)
        add_from_contours(contours_white, source="hsv_white", color_weight=0.05)

        # Motion-only rescue candidate for non-orange / low-saturation ball.
        if not candidates:
            for contour in contours_motion:
                area = float(cv2.contourArea(contour))
                if area < 8.0 or area > (h * w) * 0.01:
                    continue
                x, y, bw, bh = cv2.boundingRect(contour)
                if bw <= 0 or bh <= 0:
                    continue
                radius = float(min(bw, bh) * 0.5)
                if radius < min_r or radius > max_r:
                    continue
                x1, y1, x2, y2 = int(x), int(y), int(x + bw), int(y + bh)
                motion_roi = motion[y1:y2, x1:x2]
                motion_ratio = float(np.mean(motion_roi > 0)) if motion_roi.size else 0.0
                if motion_ratio < 0.02:
                    continue
                cx = float(x + bw * 0.5)
                cy = float(y + bh * 0.5)
                conf = float(min(0.42, 0.08 + 0.42 * motion_ratio))
                candidates.append(
                    Detection(
                        x=cx,
                        y=cy,
                        radius=radius,
                        confidence=conf,
                        source="hsv_motion",
                        motion_score=motion_ratio,
                        bbox=(x1, y1, x2, y2),
                    )
                )

        return self._dedupe(candidates)


class OpenCVFallbackDetector(BallDetector):
    def __init__(self):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=420, varThreshold=20, detectShadows=False)
        # Motion-only rescue can capture limbs/head blobs on hard videos, keep it off by default.
        self.enable_motion_rescue = _read_bool_from_env("TRAJ_ENABLE_MOTION_RESCUE", False)

    @property
    def mode_name(self) -> str:
        return "fallback"

    def _orange_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(cv2.inRange(hsv, (5, 60, 48), (22, 255, 255)), cv2.inRange(hsv, (0, 50, 45), (8, 255, 255)))
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    @staticmethod
    def _contour_circularity(contour: np.ndarray, area: float) -> float:
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 1e-6:
            return 0.0
        return float(4.0 * math.pi * area / (perimeter * perimeter + 1e-6))

    @staticmethod
    def _dedupe(detections: List[Detection], max_dist_scale: float = 1.6) -> List[Detection]:
        if not detections:
            return []
        kept: List[Detection] = []
        for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
            merged = False
            for kd in kept:
                dist = math.hypot(det.x - kd.x, det.y - kd.y)
                if dist <= max(det.radius, kd.radius) * max_dist_scale:
                    merged = True
                    break
            if not merged:
                kept.append(det)
        return kept

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        h, w = frame_bgr.shape[:2]
        min_r = max(2.0, min(h, w) * 0.0035)
        max_r = max(6.0, min(h, w) * 0.055)
        motion = self.bg.apply(frame_bgr)
        motion = cv2.GaussianBlur(motion, (5, 5), 0)
        _, motion = cv2.threshold(motion, 180, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, kernel)
        motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, kernel)

        orange = self._orange_mask(frame_bgr)
        fused = cv2.bitwise_and(orange, motion)

        contours_motion, _ = cv2.findContours(motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_fused, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: List[Detection] = []

        for contour in contours_motion + contours_fused:
            area = cv2.contourArea(contour)
            if area < 5.0:
                continue
            if area > (h * w) * 0.0035:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            if radius < min_r or radius > max_r:
                continue
            circularity = self._contour_circularity(contour, area)
            if circularity < 0.10:
                continue
            x1 = max(0, int(cx - radius))
            y1 = max(0, int(cy - radius))
            x2 = min(w - 1, int(cx + radius))
            y2 = min(h - 1, int(cy + radius))
            roi = motion[y1 : y2 + 1, x1 : x2 + 1]
            motion_score = float(np.mean(roi > 0)) if roi.size else 0.0
            orange_roi = orange[y1 : y2 + 1, x1 : x2 + 1]
            orange_ratio = float(np.mean(orange_roi > 0)) if orange_roi.size else 0.0
            if motion_score < 0.008:
                continue

            conf = float(min(0.95, 0.17 + 0.40 * circularity + 0.32 * motion_score + 0.20 * orange_ratio))
            out.append(
                Detection(float(cx), float(cy), float(radius), conf, "fallback", motion_score, (x1, y1, x2, y2))
            )

        # Hough circle branch (motion-gated, color agnostic)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 1.3)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(8, int(min_r * 2.2)),
            param1=100,
            param2=15,
            minRadius=int(max(2, min_r)),
            maxRadius=int(max_r),
        )
        if circles is not None:
            for c in np.round(circles[0, :]).astype(int):
                cx, cy, radius = int(c[0]), int(c[1]), int(c[2])
                if radius < min_r or radius > max_r:
                    continue
                x1 = max(0, cx - radius)
                y1 = max(0, cy - radius)
                x2 = min(w - 1, cx + radius)
                y2 = min(h - 1, cy + radius)
                roi = motion[y1 : y2 + 1, x1 : x2 + 1]
                motion_score = float(np.mean(roi > 0)) if roi.size else 0.0
                if motion_score < 0.006:
                    continue
                orange_roi = orange[y1 : y2 + 1, x1 : x2 + 1]
                orange_ratio = float(np.mean(orange_roi > 0)) if orange_roi.size else 0.0
                conf = float(min(0.86, 0.14 + 0.45 * motion_score + 0.24 * orange_ratio))
                out.append(Detection(float(cx), float(cy), float(radius), conf, "fallback", motion_score, (x1, y1, x2, y2)))

        out = self._dedupe(out, max_dist_scale=1.7)

        # Visibility-first rescue is optional because it can introduce limb/body false positives.
        if self.enable_motion_rescue and not out and contours_motion:
            frame_area = float(max(1, w * h))
            best = None
            best_score = -1.0
            for contour in contours_motion:
                area = float(cv2.contourArea(contour))
                if area < 6.0 or area > frame_area * 0.02:
                    continue
                x, y, bw, bh = cv2.boundingRect(contour)
                if bw <= 0 or bh <= 0:
                    continue
                roi = motion[y : y + bh, x : x + bw]
                motion_score = float(np.mean(roi > 0)) if roi.size else 0.0
                if motion_score < 0.01:
                    continue
                aspect = float(min(bw, bh) / max(1.0, max(bw, bh)))
                score = 0.6 * motion_score + 0.3 * aspect + 0.1 * (1.0 / (1.0 + area / 200.0))
                if score > best_score:
                    best_score = score
                    best = (x, y, bw, bh, motion_score)
            if best is not None:
                x, y, bw, bh, motion_score = best
                cx = float(x + bw * 0.5)
                cy = float(y + bh * 0.5)
                radius = float(max(2.0, min(max_r, min(bw, bh) * 0.5)))
                conf = float(min(0.28, 0.08 + 0.45 * motion_score))
                out.append(Detection(cx, cy, radius, conf, "fallback", motion_score, (x, y, x + bw, y + bh)))

        return out


class CompositeBallDetector(BallDetector):
    def __init__(self, model_path: str):
        yolo_conf = _read_float_from_env("TRAJ_YOLO_CONF_MIN", 0.08, min_value=0.01, max_value=0.95)
        yolo_imgsz = _read_int_from_env("TRAJ_YOLO_IMGSZ", 960, min_value=320, max_value=1600)
        yolo_min_aspect = _read_float_from_env("TRAJ_YOLO_MIN_ASPECT", 0.70, min_value=0.25, max_value=2.5)
        yolo_max_aspect = _read_float_from_env("TRAJ_YOLO_MAX_ASPECT", 1.40, min_value=0.4, max_value=3.0)
        self.yolo = YOLOBallDetector(
            model_path=model_path,
            conf_threshold=yolo_conf,
            imgsz=yolo_imgsz,
            min_aspect=yolo_min_aspect,
            max_aspect=yolo_max_aspect,
        )
        self.hsv_assist = HSVBallCandidateDetector()
        self.fallback = OpenCVFallbackDetector()
        self.motion_bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=24, detectShadows=False)
        self.enable_hsv_white = _read_bool_from_env("TRAJ_ENABLE_HSV_WHITE", False)
        self.takeover_missing_frames = _read_int_from_env("TRAJ_YOLO_TAKEOVER_MISSING", 8, min_value=1, max_value=20)
        self.takeover_aux_limit = _read_int_from_env("TRAJ_TAKEOVER_AUX_LIMIT", 1, min_value=1, max_value=4)
        self.yolo_motion_gate = _read_float_from_env("TRAJ_YOLO_MOTION_GATE", 0.006, min_value=0.0, max_value=0.4)
        self.yolo_keep_conf = _read_float_from_env("TRAJ_YOLO_KEEP_CONF", 0.66, min_value=0.05, max_value=0.99)
        self.hsv_orange_min_conf = _read_float_from_env("TRAJ_HSV_ORANGE_MIN_CONF", 0.16, min_value=0.0, max_value=1.0)
        self.hsv_orange_min_motion = _read_float_from_env("TRAJ_HSV_ORANGE_MIN_MOTION", 0.020, min_value=0.0, max_value=1.0)
        self.hsv_white_min_conf = _read_float_from_env("TRAJ_HSV_WHITE_MIN_CONF", 0.30, min_value=0.0, max_value=1.0)
        self.hsv_white_min_motion = _read_float_from_env("TRAJ_HSV_WHITE_MIN_MOTION", 0.040, min_value=0.0, max_value=1.0)
        self.fallback_min_conf = _read_float_from_env("TRAJ_FALLBACK_MIN_CONF", 0.24, min_value=0.0, max_value=1.0)
        self.fallback_min_motion = _read_float_from_env("TRAJ_FALLBACK_MIN_MOTION", 0.030, min_value=0.0, max_value=1.0)
        self.fallback_require_both = _read_bool_from_env("TRAJ_FALLBACK_REQUIRE_BOTH", True)
        self.lowlight_enable = _read_bool_from_env("TRAJ_LOWLIGHT_ENABLE", True)
        self.lowlight_luma_th = _read_float_from_env("TRAJ_LOWLIGHT_LUMA_TH", 75.0, min_value=12.0, max_value=180.0)
        self.lowlight_relax_ratio = _read_float_from_env("TRAJ_LOWLIGHT_RELAX_RATIO", 0.82, min_value=0.5, max_value=1.0)
        self.lowlight_conf_ratio = _read_float_from_env("TRAJ_LOWLIGHT_CONF_RATIO", 0.78, min_value=0.55, max_value=1.0)
        self.lowlight_aux_strict_ratio = _read_float_from_env("TRAJ_LOWLIGHT_AUX_STRICT_RATIO", 1.25, min_value=1.0, max_value=2.2)
        self.lowlight_dual_branch = _read_bool_from_env("TRAJ_LOWLIGHT_DUAL_BRANCH", True)
        self.lowlight_clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        self.last_frame_meta: Dict[str, Any] = {"lowlight_active": False, "avg_luma": 0.0}
        self.yolo_miss_streak = 0

    @property
    def mode_name(self) -> str:
        return "yolo" if self.yolo.available else "fallback"

    @property
    def fallback_reason(self) -> str:
        return self.yolo.error if not self.yolo.available else ""

    def _motion_ratio(self, motion_mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = bbox
        h, w = motion_mask.shape[:2]
        x1 = max(0, min(w - 1, int(x1)))
        x2 = max(0, min(w - 1, int(x2)))
        y1 = max(0, min(h - 1, int(y1)))
        y2 = max(0, min(h - 1, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return 0.0
        roi = motion_mask[y1:y2, x1:x2]
        return float(np.mean(roi > 0)) if roi.size else 0.0

    def _enhance_lowlight(self, frame_bgr: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_enhanced = self.lowlight_clahe.apply(y)
        merged = cv2.merge((y_enhanced, cr, cb))
        return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

    def _preprocess_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        avg_luma = float(np.mean(gray)) if gray.size else 0.0
        lowlight_active = bool(self.lowlight_enable and avg_luma <= self.lowlight_luma_th)
        self.last_frame_meta = {"lowlight_active": lowlight_active, "avg_luma": round(avg_luma, 2)}
        if not lowlight_active:
            return frame_bgr
        return self._enhance_lowlight(frame_bgr)

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        frame_for_det = self._preprocess_frame(frame_bgr)
        lowlight_active = bool(self.last_frame_meta.get("lowlight_active", False))
        aux_strict = self.lowlight_aux_strict_ratio if lowlight_active else 1.0

        yolo_conf_override = self.yolo.conf_threshold * (self.lowlight_conf_ratio if lowlight_active else 1.0)
        yolo_cands = self.yolo.detect(frame_bgr, conf_override=yolo_conf_override)
        if lowlight_active and self.lowlight_dual_branch:
            yolo_cands.extend(self.yolo.detect(frame_for_det, conf_override=yolo_conf_override))
            yolo_cands = OpenCVFallbackDetector._dedupe(yolo_cands, max_dist_scale=1.30)

        hsv_cands = self.hsv_assist.detect(frame_for_det)
        fb_cands = self.fallback.detect(frame_for_det)
        motion = self.motion_bg.apply(frame_for_det)
        motion = cv2.GaussianBlur(motion, (5, 5), 0)
        _, motion = cv2.threshold(motion, 180, 255, cv2.THRESH_BINARY)
        for cand in yolo_cands:
            cand.motion_score = self._motion_ratio(motion, cand.bbox)
        for cand in hsv_cands:
            cand.motion_score = max(cand.motion_score, self._motion_ratio(motion, cand.bbox))
        for cand in fb_cands:
            cand.motion_score = max(cand.motion_score, self._motion_ratio(motion, cand.bbox))

        hsv_orange = [d for d in hsv_cands if d.source == "hsv_orange"]
        hsv_white = [d for d in hsv_cands if d.source == "hsv_white"]

        if not self.yolo.available:
            fallback_pool = list(hsv_orange)
            if self.enable_hsv_white:
                fallback_pool.extend(hsv_white)
            fallback_pool.extend(fb_cands)
            merged = OpenCVFallbackDetector._dedupe(fallback_pool, max_dist_scale=1.65)
            merged.sort(key=lambda d: d.confidence + 0.32 * min(1.0, d.motion_score * 1.8), reverse=True)
            return merged[:26]

        # Keep strong YOLO boxes and motion-consistent lower-confidence boxes.
        yolo_motion_gate = self.yolo_motion_gate * (self.lowlight_relax_ratio if lowlight_active else 1.0)
        yolo_keep_conf = self.yolo_keep_conf * (self.lowlight_relax_ratio if lowlight_active else 1.0)
        filtered_yolo = [d for d in yolo_cands if d.motion_score >= yolo_motion_gate or d.confidence >= yolo_keep_conf]
        if filtered_yolo:
            self.yolo_miss_streak = 0
        else:
            self.yolo_miss_streak += 1

        if filtered_yolo:
            out = list(filtered_yolo)
            # Short-time assist: keep only orange HSV candidates close to YOLO.
            for hsv_det in hsv_orange:
                orange_min_conf = min(0.98, self.hsv_orange_min_conf * aux_strict)
                orange_min_motion = min(1.0, self.hsv_orange_min_motion * aux_strict)
                if hsv_det.confidence < orange_min_conf or hsv_det.motion_score < orange_min_motion:
                    continue
                near_yolo = False
                for yd in filtered_yolo:
                    dist = math.hypot(hsv_det.x - yd.x, hsv_det.y - yd.y)
                    if dist <= max(yd.radius, hsv_det.radius) * 1.18:
                        near_yolo = True
                        break
                if not near_yolo:
                    continue
                out.append(hsv_det)

            out = OpenCVFallbackDetector._dedupe(out, max_dist_scale=1.42)
            out.sort(
                key=lambda d: (
                    d.confidence
                    + 0.28 * min(1.0, d.motion_score * 1.8)
                    + (0.08 if d.source == "yolo" else 0.0)
                    + (0.02 if d.source == "hsv_orange" else 0.0)
                ),
                reverse=True,
            )
            return out[:18]

        # Do not immediately trust auxiliary detections after brief YOLO misses.
        if self.yolo_miss_streak < self.takeover_missing_frames:
            return []

        takeover_pool: List[Detection] = []
        for d in hsv_orange:
            orange_min_conf = min(0.98, self.hsv_orange_min_conf * aux_strict)
            orange_min_motion = min(1.0, self.hsv_orange_min_motion * aux_strict)
            if d.confidence >= orange_min_conf and d.motion_score >= orange_min_motion:
                takeover_pool.append(d)
        if self.enable_hsv_white:
            for d in hsv_white:
                white_min_conf = min(0.98, self.hsv_white_min_conf * aux_strict)
                white_min_motion = min(1.0, self.hsv_white_min_motion * aux_strict)
                if d.confidence >= white_min_conf and d.motion_score >= white_min_motion:
                    takeover_pool.append(d)
        for d in fb_cands:
            fb_min_conf = min(0.98, self.fallback_min_conf * aux_strict)
            fb_min_motion = min(1.0, self.fallback_min_motion * aux_strict)
            if self.fallback_require_both:
                if d.confidence >= fb_min_conf and d.motion_score >= fb_min_motion:
                    takeover_pool.append(d)
            else:
                if d.confidence >= fb_min_conf or d.motion_score >= fb_min_motion:
                    takeover_pool.append(d)

        takeover_pool = OpenCVFallbackDetector._dedupe(takeover_pool, max_dist_scale=1.45)
        takeover_pool.sort(
            key=lambda d: (
                d.confidence
                + 0.40 * min(1.0, d.motion_score * 1.8)
                + (0.05 if d.source == "hsv_orange" else 0.0)
            ),
            reverse=True,
        )
        return takeover_pool[: self.takeover_aux_limit]


class TrajectoryTracker:
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        fps: float,
        rim: Optional[RimCalibration] = None,
        pose_trigger: Optional[PoseShotTrigger] = None,
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = max(1.0, float(fps))
        self.rim = rim
        self.pose_trigger = pose_trigger
        self.kalman = cv2.KalmanFilter(6, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=np.float32)
        self.kalman.processNoiseCov = _read_diag_from_env(
            "TRAJECTORY_KF_Q",
            default=[0.18, 0.18, 0.35, 0.35, 0.12, 0.12],
            expected=6,
        )
        self.kalman.measurementNoiseCov = _read_diag_from_env(
            "TRAJECTORY_KF_R",
            default=[0.045, 0.045],
            expected=2,
        )
        self.kalman.errorCovPost = _read_diag_from_env(
            "TRAJECTORY_KF_P",
            default=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            expected=6,
        )
        self._set_dt(1.0 / self.fps)
        self.initialized = False
        # Longer hold improves endpoint continuity when detector briefly loses the ball.
        self.max_missing = max(2, int(os.getenv("TRAJECTORY_MAX_MISSING", "12")))
        self.missing_count = 0
        self.reinit_count = 0
        self.prev_raw_point: Optional[Tuple[float, float]] = None
        self.prev_speed_px_s = 0.0
        self.points: List[Dict] = []
        self.detected_points = 0
        self.predicted_points = 0
        self.total_conf = 0.0
        self.static_hotspots: Dict[Tuple[int, int], int] = {}
        self.suppressed_count = 0
        self.last_detected_frame = -1
        self.last_emitted_frame = -1
        self.lag_samples: List[int] = []
        self.phase = "pre_shot"
        self.phase_enter_frame = 0
        self.detected_recent: Deque[Dict] = deque(maxlen=8)
        self.flight_samples: Deque[Dict] = deque(maxlen=80)
        self.fit_points_count = 0
        self.fit_quality_sum = 0.0
        self.fit_quality_samples = 0
        self.parabola_segments: List[Dict] = []
        self.current_parabola_start: Optional[int] = None
        self.parabola_last_frame = -1
        self.segments: List[Tuple[int, int]] = []
        self.last_accepted_det: Optional[Detection] = None
        self.head_false_positive_suppressed = 0
        self.false_joint_suppressed = 0
        self.body_zone_suppressed = 0
        self.hsv_assist_hits = 0
        self.prev_detected_source = ""
        self.source_detect_counts: Dict[str, int] = {}
        self.reject_reason_histogram: Dict[str, int] = {}
        self.strong_yolo_conf = _read_float_from_env("TRAJ_YOLO_STRONG_CONF", 0.78, min_value=0.4, max_value=0.99)
        self.static_hotspot_threshold = _read_int_from_env("TRAJ_STATIC_HOTSPOT_TH", 42, min_value=8, max_value=160)
        self.enable_choose_motion_rescue = _read_bool_from_env("TRAJ_ENABLE_CHOOSE_MOTION_RESCUE", False)
        self.aux_min_score = _read_float_from_env("TRAJ_AUX_MIN_SCORE", 0.20, min_value=0.02, max_value=0.95)
        self.aux_flight_min_score = _read_float_from_env("TRAJ_AUX_FLIGHT_MIN_SCORE", 0.14, min_value=0.02, max_value=0.95)
        self.parabola_gate_px = _read_float_from_env("TRAJ_PARABOLA_GATE_PX", 22.0, min_value=8.0, max_value=180.0)
        self.predict_horizon_s = _read_float_from_env("TRAJ_PREDICT_HORIZON_S", 0.72, min_value=0.2, max_value=2.2)
        self.last_flight_model: Optional[Dict[str, Any]] = None
        self.shot_trigger_mode = os.getenv("TRAJ_SHOT_TRIGGER_MODE", "pose_track").strip().lower() or "pose_track"
        self.shot_gate_strict = os.getenv("TRAJ_SHOT_GATE_STRICT", "high_precision").strip().lower() or "high_precision"
        self.shot_timeout_s = _read_float_from_env("TRAJ_SHOT_TIMEOUT_S", 1.5, min_value=0.8, max_value=2.6)
        self.event_reset_frames = _read_int_from_env("TRAJ_EVENT_RESET_FRAMES", 8, min_value=2, max_value=40)
        self.shot_min_event_gap_s = _read_float_from_env("TRAJ_SHOT_MIN_EVENT_GAP_S", 0.45, min_value=0.15, max_value=2.5)
        self.shot_post_cooldown_s = _read_float_from_env("TRAJ_SHOT_POST_COOLDOWN_S", 0.35, min_value=0.1, max_value=3.0)
        self.require_release_transition = _read_bool_from_env("TRAJ_SHOT_REQUIRE_RELEASE_TRANSITION", True)
        self.release_block_fallback_th = _read_int_from_env("TRAJ_RELEASE_BLOCK_FALLBACK_TH", 6, min_value=2, max_value=80)
        self.event_state = "IDLE"
        self.event_reset_countdown = 0
        self.release_frame = -1
        self.release_votes: Deque[int] = deque(maxlen=6)
        self.arm_votes: Deque[int] = deque(maxlen=6)
        self.shot_events: List[Dict[str, Any]] = []
        self.current_event: Optional[Dict[str, Any]] = None
        self.next_event_id = 1
        self.last_event_start_frame = -10**9
        self.event_retrigger_count = 0
        self.blocked_release_count = 0
        self.release_block_streak = 0
        self.release_block_max_streak = 0
        self.release_candidate_count = 0
        self.lowlight_frames = 0
        self.lowlight_active_now = False
        self.pose_frames = 0
        self.pose_person_found_frames = 0
        self.pose_trigger_hits = 0
        self.pose_info_last: Dict[str, Any] = {}
        self.max_consecutive_miss_in_flight = 0
        self.current_consecutive_miss_in_flight = 0

    def _is_near_primary_wrist(self, det: Detection, scale: float = 3.8, base_px: float = 18.0) -> bool:
        wrist = self.pose_info_last.get("primary_wrist")
        if not (isinstance(wrist, list) and len(wrist) >= 2):
            return False
        try:
            wx, wy = float(wrist[0]), float(wrist[1])
            gate = max(float(base_px), float(det.radius) * float(scale))
            return bool(math.hypot(det.x - wx, det.y - wy) <= gate)
        except Exception:
            return False

    def _set_dt(self, dt: float) -> None:
        dt = max(1.0 / 240.0, min(1.0 / 8.0, float(dt)))
        dt2 = 0.5 * dt * dt
        self.kalman.transitionMatrix = np.array(
            [[1, 0, dt, 0, dt2, 0], [0, 1, 0, dt, 0, dt2], [0, 0, 1, 0, dt, 0], [0, 0, 0, 1, 0, dt], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]],
            dtype=np.float32,
        )

    def _safe_state(self, arr: np.ndarray) -> np.ndarray:
        flat = np.asarray(arr, dtype=np.float32).reshape(-1)
        if flat.size >= 6:
            return flat[:6]
        padded = np.zeros(6, dtype=np.float32)
        padded[: flat.size] = flat
        return padded

    def _bucket(self, x: float, y: float) -> Tuple[int, int]:
        step = 24
        return int(x // step), int(y // step)

    def _bump_reject(self, reason: str) -> None:
        if not reason:
            return
        self.reject_reason_histogram[reason] = int(self.reject_reason_histogram.get(reason, 0) + 1)

    def _update_hotspot(self, det: Detection, speed_px_frame: float) -> None:
        key = self._bucket(det.x, det.y)
        count = self.static_hotspots.get(key, 0)
        is_static = det.motion_score < 0.07 and speed_px_frame < 4.0 and det.y < self.frame_height * 0.72
        self.static_hotspots[key] = min(240, count + 1) if is_static else max(0, count - 2)

    def _is_static_false_positive(self, det: Detection, pred_x: float, pred_y: float) -> bool:
        # Keep strong YOLO detections, but allow suppression for weak YOLO/background-like blobs.
        if det.source == "yolo" and det.confidence >= self.strong_yolo_conf:
            return False
        # During flight phase, keep candidates to avoid losing trajectory around hoop.
        if self.phase == "flight" and det.confidence >= 0.35:
            return False
        if not self.initialized:
            return False
        hotspot_count = self.static_hotspots.get(self._bucket(det.x, det.y), 0)
        pred_dist = math.hypot(det.x - pred_x, det.y - pred_y)
        static_like = (
            det.y < self.frame_height * 0.76
            and det.motion_score < 0.014
            and det.confidence < 0.36
        )
        far_from_pred = pred_dist > max(220.0, det.radius * 28.0)
        if (
            static_like
            and (hotspot_count >= self.static_hotspot_threshold or far_from_pred)
        ):
            self.suppressed_count += 1
            return True
        return False

    def _score(self, det: Detection, pred_x: float, pred_y: float) -> float:
        dist = math.hypot(det.x - pred_x, det.y - pred_y) if self.initialized else 0.0
        dist_score = max(0.0, 1.0 - dist / max(130.0, det.radius * 20.0))
        velocity_score = 0.5
        if self.prev_raw_point is not None:
            frame_speed = math.hypot(det.x - self.prev_raw_point[0], det.y - self.prev_raw_point[1])
            velocity_score = max(0.0, 1.0 - abs(frame_speed - (self.prev_speed_px_s / max(1.0, self.fps))) / 120.0)
        source_bonus = 0.08 if det.source == "yolo" else (0.03 if det.source.startswith("hsv") else 0.0)
        switch_penalty = 0.0
        if self.prev_detected_source and det.source != self.prev_detected_source:
            if self.prev_detected_source == "yolo" and det.source == "hsv_orange":
                switch_penalty = 0.03
            elif self.phase == "flight":
                switch_penalty = 0.05
            else:
                switch_penalty = 0.08
        return (
            0.38 * det.confidence
            + 0.34 * dist_score
            + 0.16 * velocity_score
            + 0.12 * min(1.0, det.motion_score * 1.8)
            + source_bonus
            - switch_penalty
        )

    def _is_pose_joint_false_positive(self, det: Detection) -> bool:
        if det.source == "yolo" and det.confidence >= self.strong_yolo_conf:
            return False
        joint_points = self.pose_info_last.get("joint_points") or []
        if not joint_points:
            return False
        joint_gate = max(14.0, det.radius * (2.8 if self.phase == "flight" else 2.4))
        near_joint = False
        for jp in joint_points:
            try:
                jx, jy = float(jp[0]), float(jp[1])
            except Exception:
                continue
            if math.hypot(det.x - jx, det.y - jy) <= joint_gate:
                near_joint = True
                break
        if not near_joint:
            return False
        upper_body_bbox = self.pose_info_last.get("upper_body_bbox") or []
        in_upper_body = False
        if isinstance(upper_body_bbox, list) and len(upper_body_bbox) >= 4:
            try:
                x1, y1, x2, y2 = [float(v) for v in upper_body_bbox[:4]]
                pad = max(8.0, det.radius * 2.0)
                in_upper_body = (x1 - pad) <= det.x <= (x2 + pad) and (y1 - pad) <= det.y <= (y2 + pad)
            except Exception:
                in_upper_body = False
        if self._is_near_primary_wrist(det, scale=2.7, base_px=14.0):
            return False
        weak_like = det.confidence < 0.70 and det.motion_score < 0.20
        if in_upper_body and weak_like:
            return True
        return bool(det.confidence < 0.58 and det.motion_score < 0.14)

    def _is_body_zone_false_positive(self, det: Detection) -> bool:
        if self.phase == "flight":
            return False
        body_bbox = self.pose_info_last.get("person_bbox") or self.pose_info_last.get("upper_body_bbox") or []
        if not (isinstance(body_bbox, list) and len(body_bbox) >= 4):
            return False
        try:
            x1, y1, x2, y2 = [float(v) for v in body_bbox[:4]]
        except Exception:
            return False
        pad = max(12.0, det.radius * 2.3)
        in_box = (x1 - pad) <= det.x <= (x2 + pad) and (y1 - pad) <= det.y <= (y2 + pad)
        if not in_box:
            return False
        wrist = self.pose_info_last.get("primary_wrist")
        wrist_far = True
        if isinstance(wrist, list) and len(wrist) >= 2:
            try:
                wx, wy = float(wrist[0]), float(wrist[1])
                wrist_far = math.hypot(det.x - wx, det.y - wy) > max(18.0, det.radius * 3.6)
            except Exception:
                wrist_far = True
        if det.source == "yolo" and det.confidence >= 0.76 and (not wrist_far):
            return False
        weak_like = det.confidence < 0.52 and det.motion_score < 0.16
        torso_mid_y = y1 + (y2 - y1) * 0.45
        torso_like = det.y >= torso_mid_y
        if self.phase == "pre_shot" and det.source == "fallback" and wrist_far and torso_like and det.confidence < 0.70:
            return True
        return bool(wrist_far and weak_like and torso_like)

    def _is_head_like_false_positive(self, det: Detection) -> bool:
        # Before release, head/shoulder blobs are frequently mistaken for the ball.
        if self.phase == "flight":
            return False
        if self.prev_raw_point is None:
            return False
        large_blob = det.radius >= max(7.5, self.frame_height * 0.018)
        in_player_zone = det.y > self.frame_height * 0.40
        near_prev = abs(det.x - self.prev_raw_point[0]) < 26.0 and abs(det.y - self.prev_raw_point[1]) < 26.0
        weak_motion = det.motion_score < 0.30
        speed_mismatch = self.prev_speed_px_s > 20 and (
            math.hypot(det.x - self.prev_raw_point[0], det.y - self.prev_raw_point[1]) * self.fps
            < self.prev_speed_px_s * 0.34
        )
        hotspot_count = self.static_hotspots.get(self._bucket(det.x, det.y), 0)
        sticky_hotspot = hotspot_count >= max(10, self.static_hotspot_threshold // 2) and det.motion_score < 0.08
        not_strong_yolo = not (det.source == "yolo" and det.confidence >= self.strong_yolo_conf)
        return bool(large_blob and in_player_zone and near_prev and weak_motion and (speed_mismatch or sticky_hotspot) and not_strong_yolo)

    def _choose(self, detections: List[Detection], pred_x: float, pred_y: float) -> Optional[Detection]:
        if not detections:
            return None

        # Bootstrapping strategy: initialize from best available moving candidate.
        if not self.initialized:
            bootstrap_pool: List[Detection] = []
            for det in detections:
                if self._is_body_zone_false_positive(det):
                    self.body_zone_suppressed += 1
                    self._bump_reject("bootstrap_body_zone")
                    continue
                if self._is_pose_joint_false_positive(det):
                    self.false_joint_suppressed += 1
                    self._bump_reject("bootstrap_pose_joint")
                    continue
                bootstrap_pool.append(det)
            if not bootstrap_pool:
                self._bump_reject("bootstrap_empty")
                return None

            def bootstrap_score(det: Detection) -> float:
                source_bias = 0.0
                if self.phase == "pre_shot" and det.source == "fallback":
                    source_bias = -0.28
                elif self.phase == "pre_shot" and det.source.startswith("hsv"):
                    source_bias = -0.08
                return 0.52 * det.confidence + 0.33 * min(1.0, det.motion_score * 2.0) + 0.15 * (1.0 / (1.0 + det.radius)) + source_bias

            best_boot = max(bootstrap_pool, key=bootstrap_score)
            if self.phase == "pre_shot" and best_boot.source == "fallback":
                near_wrist = self._is_near_primary_wrist(best_boot, scale=4.2, base_px=20.0)
                if (not near_wrist) and best_boot.confidence < 0.72:
                    self._bump_reject("bootstrap_fallback_not_wrist")
                    return None
                if best_boot.confidence < 0.36 or best_boot.motion_score < 0.08:
                    self._bump_reject("bootstrap_fallback_weak")
                    return None
            if best_boot.confidence >= 0.08 or best_boot.motion_score >= 0.02:
                return best_boot

        best = None
        best_score = -1e9
        for det in detections:
            if self.phase == "pre_shot" and det.source == "fallback":
                near_wrist = self._is_near_primary_wrist(det, scale=4.2, base_px=20.0)
                if (not near_wrist) and det.confidence < 0.72:
                    self._bump_reject("pre_fallback_not_wrist")
                    continue
                if det.confidence < 0.34 or det.motion_score < 0.08:
                    self._bump_reject("pre_fallback_weak")
                    continue
                if det.radius > max(11.5, self.frame_height * 0.018):
                    self._bump_reject("pre_fallback_large")
                    continue
            if self.lowlight_active_now:
                if det.source != "yolo" and det.confidence < 0.28 and det.motion_score < 0.05:
                    self._bump_reject("lowlight_aux_weak")
                    continue
                if det.source == "yolo" and det.confidence < 0.24 and det.motion_score < 0.006:
                    self._bump_reject("lowlight_yolo_weak")
                    continue
            if self._is_body_zone_false_positive(det):
                self.body_zone_suppressed += 1
                self._bump_reject("body_zone")
                continue
            if self._is_pose_joint_false_positive(det):
                self.false_joint_suppressed += 1
                self._bump_reject("pose_joint")
                continue
            if self._is_head_like_false_positive(det):
                self.head_false_positive_suppressed += 1
                self._bump_reject("head_like")
                continue
            if self._is_static_false_positive(det, pred_x, pred_y):
                self._bump_reject("static_hotspot")
                continue
            if self.initialized:
                dist = math.hypot(det.x - pred_x, det.y - pred_y)
                speed_term = self.prev_speed_px_s / max(1.0, self.fps)
                base_gate = det.radius * 10.5 + speed_term * 1.8 + self.missing_count * 22.0
                if self.phase == "flight":
                    dynamic_gate = max(95.0, base_gate * 1.22)
                else:
                    dynamic_gate = max(72.0, base_gate * 0.85)
                if dist > dynamic_gate and det.confidence < 0.90:
                    self._bump_reject("gate_dist")
                    continue
                if det.source != "yolo":
                    aux_gate = max(64.0, det.radius * 8.5 + speed_term * 1.35 + self.missing_count * 16.0)
                    if self.phase == "flight":
                        aux_gate *= 1.10
                    if dist > aux_gate and det.confidence < 0.78:
                        self._bump_reject("aux_far")
                        continue
                    if det.source == "fallback" and det.confidence < 0.30 and det.motion_score < 0.05:
                        self._bump_reject("aux_weak")
                        continue
            score = self._score(det, pred_x, pred_y)
            if score > best_score:
                best_score = score
                best = det
        if best is None:
            # Optional rescue branch; disabled by default to avoid body-motion false positives.
            if self.enable_choose_motion_rescue and self.phase == "flight":
                moving = [d for d in detections if d.motion_score >= 0.05 and d.confidence >= 0.10]
                if moving:
                    return max(moving, key=lambda d: 0.6 * d.motion_score + 0.4 * d.confidence)
            self._bump_reject("no_candidate")
            return None
        if self.initialized and best.source != "yolo":
            required = self.aux_flight_min_score if self.phase == "flight" else self.aux_min_score
            if best_score < required:
                dist = math.hypot(best.x - pred_x, best.y - pred_y)
                if not (
                    self.phase == "flight"
                    and dist <= max(140.0, best.radius * 20.0)
                    and best.motion_score >= 0.05
                    and best.confidence >= 0.12
                ):
                    self._bump_reject("aux_low_score")
                    return None
        if self.initialized and best_score < 0.08:
            # Keep continuity when a candidate is still close to predicted position.
            dist = math.hypot(best.x - pred_x, best.y - pred_y)
            if not (dist <= max(240.0, best.radius * 38.0) and (best.confidence >= 0.05 or best.motion_score >= 0.010)):
                self._bump_reject("low_score_far")
                return None
        if (not self.initialized) and best_score < 0.06:
            self._bump_reject("bootstrap_low_score")
            return None
        return best

    def _should_enter_flight(self) -> bool:
        if len(self.detected_recent) < 3:
            return False
        s = list(self.detected_recent)[-3:]
        y_drop = float(s[0]["y"] - s[-1]["y"])
        avg_speed = float(np.mean([x["speed"] for x in s]))
        upward_ratio = float(np.mean([1.0 if x["vy"] < -24.0 else 0.0 for x in s]))
        x_span = abs(float(s[-1]["x"] - s[0]["x"]))
        return (
            y_drop > 4.0
            and avg_speed > 45.0
            and upward_ratio >= 0.34
            and x_span < self.frame_width * 0.68
            and s[-1]["y"] < self.frame_height * 0.94
        )

    def _fit_flight_model(self) -> Optional[Dict[str, Any]]:
        if len(self.flight_samples) < 6:
            return None
        window = list(self.flight_samples)[-22:]
        t_abs = np.array([float(p["time"]) for p in window], dtype=np.float32)
        x = np.array([float(p["x"]) for p in window], dtype=np.float32)
        y = np.array([float(p["y"]) for p in window], dtype=np.float32)
        t0 = float(t_abs[0])
        t = t_abs - t0
        if np.ptp(t) < 1e-3:
            return None
        mask = np.ones_like(t, dtype=bool)
        y_coef = None
        best_mask: Optional[np.ndarray] = None
        if len(t) >= 8:
            rng = np.random.default_rng(int(window[-1]["frame"]) + 17)
            best_score = -1e9
            sample_size = min(6, len(t))
            for _ in range(28):
                try:
                    sample_idx = np.sort(rng.choice(len(t), size=sample_size, replace=False))
                    coef_try = np.polyfit(t[sample_idx], y[sample_idx], 2)
                except Exception:
                    continue
                residual = np.abs(y - np.polyval(coef_try, t))
                th = max(3.8, float(np.percentile(residual, 58)) * 1.55)
                candidate_mask = residual <= th
                inliers = int(np.sum(candidate_mask))
                if inliers < 5:
                    continue
                score = float(inliers) - 0.045 * float(np.mean(residual[candidate_mask]))
                if score > best_score:
                    best_score = score
                    best_mask = candidate_mask
            if best_mask is not None:
                mask = best_mask
        for _ in range(3):
            if int(mask.sum()) < 5:
                return None
            try:
                inlier_idx = np.where(mask)[0]
                if inlier_idx.size < 5:
                    return None
                recency = np.linspace(0.72, 1.18, num=len(t), dtype=np.float32)
                robust_weight = np.ones_like(t, dtype=np.float32)
                if y_coef is not None:
                    residual_prev = np.abs(y - np.polyval(y_coef, t))
                    robust_weight = 1.0 / (1.0 + residual_prev / 8.0)
                w = np.clip(recency * robust_weight, 0.12, 2.5)
                y_coef = np.polyfit(t[inlier_idx], y[inlier_idx], 2, w=w[inlier_idx])
            except Exception:
                return None
            residual = np.abs(y - np.polyval(y_coef, t))
            active = residual[mask]
            if active.size == 0:
                return None
            mad = float(np.median(np.abs(active - np.median(active))) + 1e-6)
            mask = residual <= max(3.2, 2.7 * mad + 1.05)
        if int(mask.sum()) < 5 or y_coef is None:
            return None
        try:
            inlier_idx = np.where(mask)[0]
            recency = np.linspace(0.72, 1.18, num=len(t), dtype=np.float32)
            x_coef = np.polyfit(t[inlier_idx], x[inlier_idx], 1, w=recency[inlier_idx])
        except Exception:
            return None
        fitted_y = np.polyval(y_coef, t[mask])
        residual = np.abs(y[mask] - fitted_y)
        residual_mean = float(np.mean(residual)) if residual.size else 999.0
        inlier_ratio = float(mask.sum() / max(1, len(mask)))
        quality = max(0.0, min(1.0, 0.72 * inlier_ratio + 0.28 * (1.0 - min(1.0, residual_mean / 18.0))))
        model = {
            "t0": t0,
            "x_coef": [float(v) for v in np.asarray(x_coef).reshape(-1)],
            "y_coef": [float(v) for v in np.asarray(y_coef).reshape(-1)],
            "quality": float(quality),
        }
        self.last_flight_model = model
        return model

    @staticmethod
    def _eval_flight_model(model: Dict[str, Any], time_s: float) -> Optional[Tuple[float, float]]:
        if not model:
            return None
        try:
            t0 = float(model.get("t0", 0.0))
            x_coef = np.asarray(model.get("x_coef"), dtype=np.float32).reshape(-1)
            y_coef = np.asarray(model.get("y_coef"), dtype=np.float32).reshape(-1)
            if x_coef.size < 2 or y_coef.size < 3:
                return None
            tt = float(time_s - t0)
            return float(np.polyval(x_coef, tt)), float(np.polyval(y_coef, tt))
        except Exception:
            return None

    def _fit_flight_point(self, query_time_s: float) -> Optional[Tuple[float, float, float]]:
        model = self._fit_flight_model()
        if model is None:
            model = self.last_flight_model
        xy = self._eval_flight_model(model, query_time_s) if model is not None else None
        if xy is None:
            return None
        return float(xy[0]), float(xy[1]), float(model.get("quality", 0.0))

    def _build_future_curve(self, start_time_s: float) -> List[Dict[str, float]]:
        model = self.last_flight_model
        if model is None:
            return []
        step = max(1.0 / max(1.0, self.fps), 0.04)
        n_steps = max(4, int(self.predict_horizon_s / step))
        out: List[Dict[str, float]] = []
        for i in range(1, n_steps + 1):
            ts = float(start_time_s + i * step)
            xy = self._eval_flight_model(model, ts)
            if xy is None:
                continue
            x = float(np.clip(xy[0], 0, self.frame_width - 1))
            y = float(np.clip(xy[1], 0, self.frame_height - 1))
            out.append({"time": round(ts, 3), "x": round(x, 2), "y": round(y, 2)})
            if y >= self.frame_height - 1:
                break
        return out

    def _phase_from_event_state(self) -> str:
        if self.event_state in {"RELEASED", "FLIGHT_ACTIVE"}:
            return "flight"
        if self.event_state == "TERMINATED":
            return "post_shot"
        return "pre_shot"

    def _shot_to_rim_alignment(self, x: float, y: float, vx: float, vy: float) -> float:
        if self.rim is None:
            return 0.0
        to_rim = np.array([float(self.rim.cx - x), float(self.rim.cy - y)], dtype=np.float32)
        vel = np.array([float(vx), float(vy)], dtype=np.float32)
        denom = float(np.linalg.norm(to_rim) * np.linalg.norm(vel))
        if denom < 1e-4:
            return 0.0
        return float(np.dot(to_rim, vel) / denom)

    def _start_shot_event(self, frame_idx: int, time_s: float, trigger: Dict[str, Any]) -> None:
        if self.shot_events:
            self.event_retrigger_count += 1
        self.last_event_start_frame = int(frame_idx)
        self.release_block_streak = 0
        self.event_state = "RELEASED"
        self.release_frame = int(frame_idx)
        self.phase = "flight"
        self.phase_enter_frame = int(frame_idx)
        self.current_parabola_start = int(frame_idx)
        self.flight_samples.clear()
        self.last_flight_model = None
        event = {
            "event_id": int(self.next_event_id),
            "trigger": {"frame": int(frame_idx), "time": round(float(time_s), 3), "mode": self.shot_trigger_mode, "evidence": trigger},
            "release": {"frame": int(frame_idx), "time": round(float(time_s), 3)},
            "start": {"frame": int(frame_idx), "time": round(float(time_s), 3)},
            "end": None,
            "result": "ongoing",
            "stop_reason": None,
            "fit_curve": [],
            "predicted_curve": [],
            "fit_quality": 0.0,
            "pred_quality": 0.0,
            "quality": {},
            "_samples": [],
        }
        self.current_event = event
        self.next_event_id += 1

    def _finalize_event_curves(self, event: Dict[str, Any]) -> None:
        samples = list(event.get("_samples") or [])
        fit_curve: List[Dict[str, Any]] = []
        predicted_curve: List[Dict[str, float]] = []
        fit_quality_values: List[float] = []
        for p in samples:
            if p.get("fit_valid") and p.get("fit_x") is not None and p.get("fit_y") is not None:
                fit_curve.append({"frame": int(p["frame"]), "time": float(p["time"]), "x": float(p["fit_x"]), "y": float(p["fit_y"])})
                if p.get("fit_quality") is not None:
                    try:
                        fit_quality_values.append(float(p["fit_quality"]))
                    except Exception:
                        pass
            if not predicted_curve and p.get("future_curve"):
                predicted_curve = list(p.get("future_curve") or [])
        if not predicted_curve:
            for p in reversed(samples):
                curve = p.get("future_curve") or []
                if curve:
                    predicted_curve = list(curve)
                    break
        event["fit_curve"] = fit_curve
        event["predicted_curve"] = predicted_curve
        fit_quality = round(float(np.mean(fit_quality_values)) if fit_quality_values else 0.0, 4)
        pred_quality = 0.0
        if predicted_curve:
            pred_quality = min(1.0, max(0.0, 0.28 + min(0.72, len(predicted_curve) / 18.0)))
            if fit_quality_values:
                pred_quality = min(1.0, max(0.0, 0.45 * pred_quality + 0.55 * fit_quality))
        event["fit_quality"] = fit_quality
        event["pred_quality"] = round(float(pred_quality), 4)
        duration_s = 0.0
        if event.get("start") and event.get("end"):
            duration_s = max(0.0, float(event["end"]["time"]) - float(event["start"]["time"]))
        event["quality"] = {
            "fit_quality_score": fit_quality,
            "pred_quality_score": round(float(pred_quality), 4),
            "point_count": int(len(samples)),
            "fit_point_count": int(len(fit_curve)),
            "duration_s": round(float(duration_s), 3),
        }
        event.pop("_samples", None)

    def _terminate_shot_event(self, frame_idx: int, time_s: float, stop_reason: str) -> None:
        post_cooldown_frames = max(2, int(round(self.shot_post_cooldown_s * self.fps)))
        if self.current_event is None:
            self.event_state = "TERMINATED"
            self.event_reset_countdown = int(max(self.event_reset_frames, post_cooldown_frames))
            self.phase = "post_shot"
            return
        result = "miss"
        if stop_reason == "MADE_STOP":
            result = "made"
        elif stop_reason == "TIMEOUT_STOP":
            result = "timeout"
        self.current_event["result"] = result
        self.current_event["stop_reason"] = stop_reason
        self.current_event["end"] = {"frame": int(frame_idx), "time": round(float(time_s), 3)}
        self._finalize_event_curves(self.current_event)
        self.shot_events.append(self.current_event)
        self.current_event = None
        if self.current_parabola_start is not None:
            self.parabola_segments.append(
                {
                    "start_frame": int(self.current_parabola_start),
                    "end_frame": int(frame_idx),
                    "length": int(max(1, frame_idx - self.current_parabola_start + 1)),
                }
            )
        self.current_parabola_start = None
        self.flight_samples.clear()
        self.last_flight_model = None
        self.current_consecutive_miss_in_flight = 0
        self.release_block_streak = 0
        self.release_votes.clear()
        self.arm_votes.clear()
        self.event_state = "TERMINATED"
        self.event_reset_countdown = int(max(self.event_reset_frames, post_cooldown_frames))
        self.phase = "post_shot"

    def _check_made_stop(self, prev_xy: Tuple[float, float], curr_xy: Tuple[float, float]) -> bool:
        if self.rim is None:
            return False
        px, py = prev_xy
        cx, cy = curr_xy
        upper = self.rim.cy - self.rim.r * 0.78
        lower = self.rim.cy + self.rim.r * 0.50
        x_gate = self.rim.r * 1.30
        desc = (cy - py) > max(0.8, self.rim.r * 0.015)
        cross_vertical = py <= upper and cy >= lower
        near_center_x = abs(px - self.rim.cx) <= x_gate and abs(cx - self.rim.cx) <= x_gate
        return bool(desc and cross_vertical and near_center_x)

    def _update_shot_state(
        self,
        frame_idx: int,
        time_s: float,
        kind: str,
        x_raw: float,
        y_raw: float,
        vx: float,
        vy: float,
        speed_px_s: float,
        pose_signal: Dict[str, Any],
    ) -> None:
        if self.event_state == "TERMINATED":
            self.event_reset_countdown -= 1
            if self.event_reset_countdown <= 0:
                self.event_state = "IDLE"
                self.phase = "pre_shot"
            return

        if self.event_state in {"RELEASED", "FLIGHT_ACTIVE"}:
            if self.event_state == "RELEASED" and frame_idx > self.release_frame:
                self.event_state = "FLIGHT_ACTIVE"
            return

        if kind != "detected":
            self.arm_votes.append(0)
            self.release_votes.append(0)
            return

        min_event_gap_frames = max(2, int(round(self.shot_min_event_gap_s * self.fps)))
        if (frame_idx - self.last_event_start_frame) < min_event_gap_frames:
            self.arm_votes.append(0)
            self.release_votes.append(0)
            self.event_state = "IDLE"
            self.phase = "pre_shot"
            return

        elbow_trend = bool(pose_signal.get("elbow_extend_trend"))
        elbow_angle = pose_signal.get("elbow_angle")
        high_precision = self.shot_gate_strict == "high_precision"
        pose_reliable = bool(pose_signal.get("model_available") and pose_signal.get("person_found"))
        toward_rim = self._shot_to_rim_alignment(x_raw, y_raw, vx, vy)
        pose_ready = bool(
            pose_signal.get("wrist_above_shoulder")
            and pose_signal.get("elbow_extended")
            and (not high_precision or elbow_trend or (elbow_angle is not None and float(elbow_angle) >= 150.0))
        )
        near_hand = False
        ball_to_hand = pose_signal.get("ball_to_hand_px")
        if ball_to_hand is not None:
            near_hand_th = max(10.0, self.frame_height * (0.09 if high_precision else 0.12))
            near_hand = float(ball_to_hand) <= near_hand_th
        arm_candidate = bool(
            pose_ready
            and near_hand
            and speed_px_s >= (26.0 if high_precision else 24.0)
            and vy < 26.0
            and y_raw < self.frame_height * 0.95
            and (self.rim is None or toward_rim > (-0.08 if high_precision else -0.05))
        )
        if (not arm_candidate) and self.shot_trigger_mode == "pose_track" and (not pose_reliable):
            arm_candidate = bool(
                speed_px_s >= (32.0 if high_precision else 28.0)
                and vy < 8.0
                and y_raw < self.frame_height * 0.92
                and (self.rim is None or toward_rim > (-0.03 if high_precision else -0.06))
            )
        self.arm_votes.append(1 if arm_candidate else 0)

        arm_required = 2 if high_precision else 1
        if self.event_state == "IDLE" and sum(list(self.arm_votes)[-4:]) >= arm_required:
            self.event_state = "ARMED"
            self.phase = "pre_shot"

        ballistic_release = bool(vy < (-26.0 if high_precision else -20.0) and speed_px_s > (46.0 if high_precision else 42.0) and (self.rim is None or toward_rim > (0.08 if high_precision else 0.04)))
        elbow_release_ready = bool((elbow_angle is not None and float(elbow_angle) >= 148.0) or elbow_trend or not high_precision)
        pose_release = bool(pose_signal.get("release_transition") and pose_ready and ballistic_release and elbow_release_ready)
        pose_sustain_release = bool(self.event_state == "ARMED" and pose_ready and (not near_hand) and ballistic_release)
        blocked_transition = bool(
            self.event_state == "ARMED"
            and high_precision
            and pose_reliable
            and pose_ready
            and ballistic_release
            and not bool(pose_signal.get("release_transition"))
        )
        fallback_block_release = bool(
            high_precision
            and self.require_release_transition
            and self.release_block_streak >= self.release_block_fallback_th
            and self.event_state == "ARMED"
            and pose_ready
            and ballistic_release
            and toward_rim > 0.02
            and speed_px_s > 45.0
        )
        if self.shot_trigger_mode == "pose_track":
            if pose_reliable:
                if high_precision and self.require_release_transition:
                    release_candidate = bool(self.event_state == "ARMED" and (pose_release or fallback_block_release) and toward_rim > -0.04)
                else:
                    release_candidate = bool(pose_release or pose_sustain_release)
            else:
                release_candidate = bool(
                    self.event_state == "ARMED"
                    and ballistic_release
                    and toward_rim > (0.02 if high_precision else -0.05)
                    and speed_px_s > (56.0 if high_precision else 46.0)
                )
        else:
            release_candidate = bool(self.event_state == "ARMED" and ballistic_release)
        if blocked_transition:
            self.blocked_release_count += 1
            self.release_block_streak = min(240, self.release_block_streak + 1)
            self.release_block_max_streak = max(self.release_block_max_streak, self.release_block_streak)
        elif release_candidate:
            self.release_block_streak = 0
        elif self.event_state != "ARMED":
            self.release_block_streak = 0
        if release_candidate:
            self.release_candidate_count += 1
        self.release_votes.append(1 if release_candidate else 0)

        votes_window = 4 if high_precision else 4
        if high_precision and self.require_release_transition:
            votes_required = 1
        else:
            votes_required = 2 if high_precision else 1
        if self.event_state in {"ARMED", "IDLE"} and sum(list(self.release_votes)[-votes_window:]) >= votes_required:
            trigger = {
                "pose_ready": pose_ready,
                "release_transition": bool(pose_signal.get("release_transition")),
                "toward_rim": round(float(toward_rim), 4),
                "vy": round(float(vy), 3),
                "speed_px_s": round(float(speed_px_s), 3),
                "pose_model_available": bool(pose_signal.get("model_available")),
            }
            self._start_shot_event(frame_idx=frame_idx, time_s=time_s, trigger=trigger)
            self.pose_trigger_hits += 1
        elif self.event_state != "ARMED":
            self.release_block_streak = 0

    def update(
        self,
        frame_idx: int,
        time_s: float,
        detections: List[Detection],
        frame_bgr: Optional[np.ndarray] = None,
        lowlight_active: bool = False,
    ) -> Optional[Dict]:
        self._set_dt(1.0 / self.fps)
        self.lowlight_active_now = bool(lowlight_active)
        if self.lowlight_active_now:
            self.lowlight_frames += 1

        pred_x, pred_y = (0.0, 0.0)
        if self.initialized:
            pred_state = self._safe_state(self.kalman.predict())
            pred_x, pred_y = float(pred_state[0]), float(pred_state[1])

        in_flight = self.event_state in {"RELEASED", "FLIGHT_ACTIVE"}
        if in_flight and len(self.flight_samples) >= 8 and detections:
            flight_model = self._fit_flight_model()
            expected_xy = self._eval_flight_model(flight_model, time_s) if flight_model is not None else None
            model_quality = float((flight_model or {}).get("quality", 0.0))
            if expected_xy is not None and model_quality >= 0.58:
                gate_px = max(24.0, self.parabola_gate_px + self.missing_count * 4.0)
                gated: List[Detection] = []
                ex, ey = float(expected_xy[0]), float(expected_xy[1])
                for det in detections:
                    if det.source == "yolo" and det.confidence >= self.strong_yolo_conf:
                        gated.append(det)
                        continue
                    if math.hypot(det.x - ex, det.y - ey) <= max(12.0, gate_px):
                        gated.append(det)
                    else:
                        self._bump_reject("parabola_gate")
                if gated:
                    detections = gated

        chosen = self._choose(detections, pred_x, pred_y)
        if chosen is not None:
            meas = np.array([[np.float32(chosen.x)], [np.float32(chosen.y)]], dtype=np.float32)
            if not self.initialized:
                self.kalman.statePost = np.array([[np.float32(chosen.x)], [np.float32(chosen.y)], [0.0], [0.0], [0.0], [0.0]], dtype=np.float32)
                self.initialized = True
            else:
                self.kalman.correct(meas)
            state = self._safe_state(self.kalman.statePost)
            x_raw, y_raw, vx, vy = float(state[0]), float(state[1]), float(state[2]), float(state[3])
            kind, source, conf = "detected", chosen.source, float(chosen.confidence)
            self.detected_points += 1
            self.source_detect_counts[source] = int(self.source_detect_counts.get(source, 0) + 1)
            if chosen.source.startswith("hsv"):
                self.hsv_assist_hits += 1
            self.total_conf += conf
            self.last_detected_frame = frame_idx
            self.missing_count = 0
            self.last_accepted_det = chosen
            self.prev_detected_source = source
            if in_flight:
                self.current_consecutive_miss_in_flight = 0
        else:
            if not self.initialized:
                return None
            self.missing_count += 1
            if in_flight:
                self.current_consecutive_miss_in_flight += 1
                self.max_consecutive_miss_in_flight = max(
                    self.max_consecutive_miss_in_flight,
                    self.current_consecutive_miss_in_flight,
                )
            if self.missing_count > self.max_missing:
                self.reinit_count += 1
                self.initialized = False
                self.prev_raw_point = None
                self.last_accepted_det = None
                return None
            state = self._safe_state(self.kalman.statePre)
            x_raw, y_raw, vx, vy = float(state[0]), float(state[1]), float(state[2]), float(state[3])
            kind, source, conf = "predicted", "kalman", 0.0
            self.predicted_points += 1

        x_raw = float(np.clip(x_raw, 0, self.frame_width - 1))
        y_raw = float(np.clip(y_raw, 0, self.frame_height - 1))
        if self.prev_raw_point is None:
            speed_px_s = 0.0
        else:
            speed_px_s = math.hypot(x_raw - self.prev_raw_point[0], y_raw - self.prev_raw_point[1]) * self.fps
        if chosen is not None:
            self._update_hotspot(chosen, speed_px_s / max(1.0, self.fps))
        if kind == "detected":
            self.detected_recent.append({"x": x_raw, "y": y_raw, "speed": speed_px_s, "vy": vy})

        pose_signal: Dict[str, Any] = {
            "model_available": False,
            "person_found": False,
            "wrist_above_shoulder": False,
            "elbow_extended": False,
            "release_transition": False,
            "ball_to_hand_px": None,
        }
        if self.pose_trigger is not None and frame_bgr is not None:
            pose_signal = self.pose_trigger.infer(
                frame_bgr,
                (x_raw, y_raw) if kind == "detected" else None,
                frame_idx=frame_idx,
            )
            self.pose_frames += 1
            if bool(pose_signal.get("person_found")):
                self.pose_person_found_frames += 1
            self.pose_info_last = pose_signal

        self._update_shot_state(
            frame_idx=frame_idx,
            time_s=time_s,
            kind=kind,
            x_raw=x_raw,
            y_raw=y_raw,
            vx=vx,
            vy=vy,
            speed_px_s=speed_px_s,
            pose_signal=pose_signal,
        )
        self.phase = self._phase_from_event_state()
        in_flight = self.event_state in {"RELEASED", "FLIGHT_ACTIVE"}

        fit = None
        if in_flight:
            self.parabola_last_frame = frame_idx
            self.flight_samples.append({"frame": frame_idx, "time": time_s, "x": x_raw, "y": y_raw, "kind": kind})
            fit = self._fit_flight_point(query_time_s=time_s)

        fit_x = float(np.clip(fit[0], 0, self.frame_width - 1)) if fit is not None else None
        fit_y = float(np.clip(fit[1], 0, self.frame_height - 1)) if fit is not None else None
        fit_quality = float(fit[2]) if fit is not None else 0.0
        fit_valid = fit is not None
        if fit_valid:
            self.fit_points_count += 1
            self.fit_quality_sum += fit_quality
            self.fit_quality_samples += 1
        future_curve = self._build_future_curve(start_time_s=time_s) if (fit_valid and in_flight) else []

        emit_x = fit_x if fit_valid else x_raw
        emit_y = fit_y if fit_valid else y_raw
        lag_frames = max(0, frame_idx - self.last_detected_frame if self.last_detected_frame >= 0 else frame_idx + 1)
        self.prev_raw_point = (x_raw, y_raw)
        self.prev_speed_px_s = speed_px_s
        self.lag_samples.append(int(lag_frames))
        self.last_emitted_frame = frame_idx
        if not self.segments:
            self.segments.append((frame_idx, frame_idx))
        else:
            s0, e0 = self.segments[-1]
            if frame_idx - e0 <= 1:
                self.segments[-1] = (s0, frame_idx)
            else:
                self.segments.append((frame_idx, frame_idx))
        point = {
            "frame": int(frame_idx),
            "time": round(float(time_s), 3),
            "x": round(float(emit_x), 2),
            "y": round(float(emit_y), 2),
            "raw_x": round(float(x_raw), 2),
            "raw_y": round(float(y_raw), 2),
            "fit_x": round(float(fit_x), 2) if fit_x is not None else None,
            "fit_y": round(float(fit_y), 2) if fit_y is not None else None,
            "fit_valid": bool(fit_valid),
            "fit_quality": round(float(fit_quality), 4) if fit_valid else None,
            "kind": kind,
            "source": source,
            "phase": self.phase,
            "event_state": self.event_state,
            "lag_frames": int(lag_frames),
            "confidence": round(float(conf), 4),
            "speed_px_s": round(float(speed_px_s), 2),
            "radius": round(float(chosen.radius), 3) if chosen is not None else (round(float(self.last_accepted_det.radius), 3) if self.last_accepted_det is not None else None),
            "future_curve": future_curve,
        }
        if self.current_event is not None and in_flight:
            self.current_event["_samples"].append(point)
            point["shot_event_id"] = int(self.current_event["event_id"])

            stop_reason: Optional[str] = None
            samples = self.current_event["_samples"]
            if len(samples) >= 2:
                p0, p1 = samples[-2], samples[-1]
                if self._check_made_stop((float(p0["x"]), float(p0["y"])), (float(p1["x"]), float(p1["y"]))):
                    stop_reason = "MADE_STOP"
            start_time = float(self.current_event.get("start", {}).get("time", time_s))
            if stop_reason is None and (time_s - start_time) >= self.shot_timeout_s:
                stop_reason = "TIMEOUT_STOP"
            if stop_reason is None:
                if self.rim is not None:
                    dist_to_rim = math.hypot(float(emit_x) - self.rim.cx, float(emit_y) - self.rim.cy)
                    life_frames = frame_idx - int(self.current_event["start"]["frame"])
                    if life_frames >= max(6, int(round(0.18 * self.fps))) and vy > 35.0 and emit_y > (self.rim.cy + self.rim.r * 1.8) and dist_to_rim > self.rim.r * 1.5:
                        stop_reason = "MISS_STOP"
                else:
                    if vy > 58.0 and emit_y > self.frame_height * 0.78:
                        stop_reason = "MISS_STOP"
            if stop_reason is None and self.missing_count >= max(5, self.event_reset_frames):
                stop_reason = "MISS_STOP"
            if stop_reason is not None:
                self._terminate_shot_event(frame_idx=frame_idx, time_s=time_s, stop_reason=stop_reason)
                point["future_curve"] = []
                point["event_state"] = self.event_state
                self.phase = self._phase_from_event_state()
                point["phase"] = self.phase

        self.points.append(point)
        return point

    def get_live_state(self, processed_frame: int) -> Dict:
        lag = max(0, processed_frame - self.last_detected_frame) if self.last_detected_frame >= 0 else max(0, processed_frame + 1)
        return {
            "processed_frame": int(processed_frame),
            "last_detected_frame": int(self.last_detected_frame),
            "last_emitted_frame": int(self.last_emitted_frame),
            "lag_frames": int(lag),
            "phase": self.phase,
            "event_state": self.event_state,
            "shot_event_active": bool(self.current_event is not None),
            "causal_mode": True,
        }

    def get_shot_events(self) -> List[Dict[str, Any]]:
        return [dict(event) for event in self.shot_events]

    def get_predicted_trajectory_points(self) -> List[Dict[str, float]]:
        if self.current_event is not None:
            samples = self.current_event.get("_samples") or []
            for p in reversed(samples):
                curve = p.get("future_curve") or []
                if curve:
                    return list(curve)
        if self.shot_events:
            curve = self.shot_events[-1].get("predicted_curve") or []
            if curve:
                return list(curve)
        return []

    def get_latest_shot_prediction(self, fps: float) -> Dict[str, Any]:
        if self.shot_events:
            last = self.shot_events[-1]
            result = str(last.get("result") or "unknown")
            label = "Unknown"
            if result == "made":
                label = "Basket"
            elif result in {"miss", "timeout"}:
                label = "No Basket"
            end_info = last.get("end") or {}
            confidence = 0.0
            if result == "made":
                confidence = 0.92
            elif result == "miss":
                confidence = 0.82
            elif result == "timeout":
                confidence = 0.62
            return {
                "label": label,
                "confidence": round(float(confidence), 4),
                "crossing_frame": end_info.get("frame"),
                "crossing_time": end_info.get("time"),
                "reason": str(last.get("stop_reason") or "shot_event"),
            }
        return _predict_shot_by_rim(self.points, rim=self.rim, fps=fps)

    def finalize_metrics(self, total_frames: int) -> Dict:
        coverage = (len(self.points) / max(1, total_frames)) * 100.0
        avg_conf = (self.total_conf / max(1, self.detected_points)) if self.detected_points else 0.0
        fit_quality_score = (self.fit_quality_sum / max(1, self.fit_quality_samples)) if self.fit_quality_samples else 0.0
        continuity = 0.0
        if self.segments:
            longest = max((e - s + 1) for s, e in self.segments)
            continuity = (float(longest) / max(1, total_frames)) * 100.0
        p95 = 0
        if self.lag_samples:
            sv = sorted(self.lag_samples)
            p95 = int(sv[max(0, min(len(sv) - 1, int(round((len(sv) - 1) * 0.95))))])
        source_counts = dict(sorted(self.source_detect_counts.items(), key=lambda kv: kv[1], reverse=True))
        total_source_count = max(1, sum(source_counts.values()))
        source_ratio = {k: round(float(v / total_source_count), 4) for k, v in source_counts.items()}
        hotspot_threshold = max(6, self.static_hotspot_threshold // 4)
        static_hotspot_topk = [
            {"bucket_x": int(xb), "bucket_y": int(yb), "count": int(cnt)}
            for (xb, yb), cnt in sorted(self.static_hotspots.items(), key=lambda kv: kv[1], reverse=True)
            if int(cnt) >= hotspot_threshold
        ][:8]
        reject_hist = dict(sorted(self.reject_reason_histogram.items(), key=lambda kv: kv[1], reverse=True))
        made_count = int(sum(1 for e in self.shot_events if str(e.get("result")) == "made"))
        miss_count = int(sum(1 for e in self.shot_events if str(e.get("result")) in {"miss", "timeout"}))
        return {
            "total_points": int(len(self.points)),
            "detected_points": int(self.detected_points),
            "predicted_points": int(self.predicted_points),
            "coverage_percent": round(float(coverage), 2),
            "avg_confidence": round(float(avg_conf), 4),
            "continuity_percent": round(float(min(100.0, continuity)), 2),
            "valid_segments": [{"start_frame": int(s), "end_frame": int(e), "length": int(e - s + 1)} for s, e in self.segments if (e - s + 1) >= 3],
            "valid_segment_count": int(sum(1 for s, e in self.segments if (e - s + 1) >= 3)),
            "fit_points_count": int(self.fit_points_count),
            "fit_quality_score": round(float(fit_quality_score), 4),
            "parabola_segments": self.parabola_segments,
            "parabola_segment_count": int(len(self.parabola_segments)),
            "suppressed_false_positives": int(self.suppressed_count),
            "head_false_positive_suppressed": int(self.head_false_positive_suppressed),
            "false_joint_suppressed": int(self.false_joint_suppressed),
            "body_zone_suppressed": int(self.body_zone_suppressed),
            "hsv_assist_hits": int(self.hsv_assist_hits),
            "p95_lag_frames": int(p95),
            "reinit_count": int(self.reinit_count),
            "source_counts": source_counts,
            "source_ratio": source_ratio,
            "static_hotspot_topk": static_hotspot_topk,
            "reject_reason_histogram": reject_hist,
            "shot_count": int(len(self.shot_events)),
            "made_count": int(made_count),
            "miss_count": int(miss_count),
            "event_retrigger_count": int(self.event_retrigger_count),
            "max_consecutive_miss_in_flight": int(self.max_consecutive_miss_in_flight),
            "blocked_release_count": int(self.blocked_release_count),
            "release_block_max_streak": int(self.release_block_max_streak),
            "release_candidate_count": int(self.release_candidate_count),
            "lowlight_frame_ratio": round(float(self.lowlight_frames / max(1, total_frames)), 4),
            "pose_frames": int(self.pose_frames),
            "pose_person_found_frames": int(self.pose_person_found_frames),
            "pose_person_found_ratio": round(float(self.pose_person_found_frames / max(1, self.pose_frames)), 4),
            "pose_trigger_hits": int(self.pose_trigger_hits),
            "causal_mode": True,
        }


def _draw_overlay(frame: np.ndarray, point: Optional[Dict], recent_points: List[Dict], detector_mode: str, frame_idx: int, phase: str, lag_frames: int) -> np.ndarray:
    canvas = frame.copy()
    trail_points = max(16, min(96, _read_int_from_env("TRAJ_OVERLAY_TRAIL_POINTS", 44, min_value=16, max_value=140)))
    pred_points = max(4, min(30, _read_int_from_env("TRAJ_OVERLAY_PRED_POINTS", 12, min_value=4, max_value=48)))
    trail_stride = max(1, min(4, _read_int_from_env("TRAJ_OVERLAY_TRAIL_STRIDE", 1, min_value=1, max_value=4)))

    fast_mode = lag_frames >= 3
    if lag_frames >= 2:
        trail_points = max(14, int(trail_points * 0.72))
        pred_points = max(4, int(pred_points * 0.75))
        trail_stride = min(3, trail_stride + 1)
    if fast_mode:
        trail_points = max(12, int(trail_points * 0.78))
        pred_points = max(4, int(pred_points * 0.7))
        trail_stride = min(4, trail_stride + 1)

    points = recent_points[-trail_points:]
    active_event_id = point.get("shot_event_id") if point is not None else None
    if active_event_id is not None:
        same_event: List[Dict[str, Any]] = []
        for p in reversed(recent_points):
            if p.get("shot_event_id") == active_event_id:
                same_event.append(p)
                if len(same_event) >= trail_points * 2:
                    break
        if same_event:
            points = list(reversed(same_event))[-trail_points:]
    elif phase == "pre_shot":
        # In pre-shot, render only stronger YOLO detections to avoid body-joint sticky trails.
        pre_points: List[Dict[str, Any]] = []
        for p in points:
            try:
                conf = float(p.get("confidence") or 0.0)
            except Exception:
                conf = 0.0
            if p.get("source") == "yolo" and conf >= 0.45:
                pre_points.append(p)
        points = pre_points[-6:]
        trail_stride = max(2, trail_stride)

    if trail_stride > 1 and len(points) > 6:
        points = points[::trail_stride] + [points[-1]]

    unified_poly: List[Tuple[int, int]] = []
    for p in points:
        if p.get("fit_valid") and p.get("fit_x") is not None and p.get("fit_y") is not None:
            x, y = int(p["fit_x"]), int(p["fit_y"])
        elif p.get("raw_x") is not None and p.get("raw_y") is not None:
            x, y = int(p["raw_x"]), int(p["raw_y"])
        elif p.get("x") is not None and p.get("y") is not None:
            x, y = int(p["x"]), int(p["y"])
        else:
            continue
        if unified_poly and math.hypot(x - unified_poly[-1][0], y - unified_poly[-1][1]) > 120.0:
            continue
        unified_poly.append((x, y))

    pred_curve_poly: List[Tuple[int, int]] = []
    if point is not None:
        pred_curve_poly = [
            (int(pt["x"]), int(pt["y"]))
            for pt in (point.get("future_curve") or [])
            if pt.get("x") is not None and pt.get("y") is not None
        ][:pred_points]

    glow_layer = np.zeros_like(canvas)
    core_layer = np.zeros_like(canvas)
    pred_layer = np.zeros_like(canvas)

    if len(unified_poly) >= 2:
        n = max(2, len(unified_poly))
        glow_thickness = 3 if fast_mode else 4
        core_thickness = 1 if fast_mode else 2
        for i in range(1, n):
            p1 = unified_poly[i - 1]
            p2 = unified_poly[i]
            if math.hypot(p2[0] - p1[0], p2[1] - p1[1]) > 120.0:
                continue
            ratio = float(i / max(1, n - 1))
            glow_color = (
                int(18 + 22 * ratio),    # B
                int(20 + 36 * ratio),    # G
                int(128 + 112 * ratio),  # R
            )
            core_color = (
                int(26 + 16 * ratio),    # B
                int(30 + 24 * ratio),    # G
                int(176 + 74 * ratio),   # R
            )
            cv2.line(glow_layer, p1, p2, glow_color, glow_thickness, cv2.LINE_AA)
            cv2.line(core_layer, p1, p2, core_color, core_thickness, cv2.LINE_AA)

    if not fast_mode and len(unified_poly) >= 3:
        glow_layer = cv2.GaussianBlur(glow_layer, (7, 7), 0)
    elif len(unified_poly) >= 3:
        glow_layer = cv2.GaussianBlur(glow_layer, (5, 5), 0)

    if phase == "pre_shot":
        cv2.addWeighted(glow_layer, 0.20 if fast_mode else 0.24, canvas, 1.0, 0.0, canvas)
        cv2.addWeighted(core_layer, 0.34 if fast_mode else 0.40, canvas, 1.0, 0.0, canvas)
    else:
        cv2.addWeighted(glow_layer, 0.34 if fast_mode else 0.40, canvas, 1.0, 0.0, canvas)
        cv2.addWeighted(core_layer, 0.70 if fast_mode else 0.78, canvas, 1.0, 0.0, canvas)

    if point is not None and pred_curve_poly and phase == "flight":
        anchor = (int(point["x"]), int(point["y"]))
        for i, nxt in enumerate(pred_curve_poly):
            if math.hypot(nxt[0] - anchor[0], nxt[1] - anchor[1]) > 140.0:
                break
            if i % 2 == 0 or i <= 2:
                ratio = float(i / max(1, len(pred_curve_poly) - 1))
                pred_color = (
                    int(22 + 26 * ratio),   # B
                    int(176 + 74 * ratio),  # G
                    int(24 + 28 * ratio),   # R
                )
                cv2.line(pred_layer, anchor, nxt, pred_color, 1 if fast_mode else 2, cv2.LINE_AA)
            anchor = nxt
    if np.any(pred_layer):
        if not fast_mode:
            pred_layer = cv2.GaussianBlur(pred_layer, (5, 5), 0)
        cv2.addWeighted(pred_layer, 0.58 if fast_mode else 0.68, canvas, 1.0, 0.0, canvas)

    for p in points[-8:]:
        px = p.get("fit_x") if p.get("fit_valid") and p.get("fit_x") is not None else p.get("raw_x")
        py = p.get("fit_y") if p.get("fit_valid") and p.get("fit_y") is not None else p.get("raw_y")
        if px is None or py is None:
            continue
        cv2.circle(canvas, (int(px), int(py)), 1, (245, 205, 132), -1, cv2.LINE_AA)
    draw_center = point is not None
    if draw_center and phase == "pre_shot":
        try:
            center_conf = float(point.get("confidence") or 0.0)
        except Exception:
            center_conf = 0.0
        if point.get("source") != "yolo" or center_conf < 0.55:
            draw_center = False
    if draw_center and point is not None:
        center = (int(point["x"]), int(point["y"]))
        layer = canvas.copy()
        cv2.circle(layer, center, 7 if fast_mode else 8, (248, 184, 76), 1 if fast_mode else 2, cv2.LINE_AA)
        cv2.circle(layer, center, 2, (255, 244, 196), -1, cv2.LINE_AA)
        cv2.addWeighted(layer, 0.66 if fast_mode else 0.72, canvas, 0.34 if fast_mode else 0.28, 0.0, canvas)

    def _put_text_with_outline(text: str, org: Tuple[int, int], color: Tuple[int, int, int], scale: float = 0.58) -> None:
        cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (18, 18, 18), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    phase_map = {"pre_shot": "Pre-shot", "flight": "Flight", "post_shot": "Post-shot"}
    mode_label = "YOLO" if detector_mode == "yolo" else "OpenCV fallback"
    _put_text_with_outline(f"Mode: {mode_label}", (18, 30), (244, 244, 244), 0.56)
    _put_text_with_outline(f"Frame: {frame_idx}", (18, 52), (226, 226, 226), 0.54)
    _put_text_with_outline(f"Phase: {phase_map.get(phase, phase)}", (212, 52), (226, 226, 226), 0.54)
    _put_text_with_outline(f"Lag: {lag_frames}f", (18, 74), (255, 226, 130), 0.54)
    return canvas

def _normalize_rim_calibration(rim_calibration: Optional[Dict[str, Any]], width: int, height: int) -> Optional[RimCalibration]:
    if not rim_calibration:
        return None
    try:
        cx = float(rim_calibration.get("cx"))
        cy = float(rim_calibration.get("cy"))
        r = float(rim_calibration.get("r"))
    except Exception:
        return None
    if width <= 0 or height <= 0:
        return None
    if not math.isfinite(cx) or not math.isfinite(cy) or not math.isfinite(r):
        return None
    if r <= 2:
        return None
    cx = float(np.clip(cx, 0, width - 1))
    cy = float(np.clip(cy, 0, height - 1))
    max_r = max(6.0, min(width, height) * 0.45)
    r = float(np.clip(r, 6.0, max_r))
    return RimCalibration(cx=cx, cy=cy, r=r)


def _predict_shot_by_rim(points: List[Dict], rim: Optional[RimCalibration], fps: float) -> Dict[str, Any]:
    if rim is None:
        return {
            "label": "Unknown",
            "confidence": 0.0,
            "crossing_frame": None,
            "crossing_time": None,
            "reason": "rim_calibration_missing",
        }
    if not points:
        return {
            "label": "Unknown",
            "confidence": 0.0,
            "crossing_frame": None,
            "crossing_time": None,
            "reason": "no_trajectory_points",
        }

    def point_xy(point: Dict[str, Any]) -> Tuple[float, float]:
        if point.get("fit_valid") and point.get("fit_x") is not None and point.get("fit_y") is not None:
            return float(point["fit_x"]), float(point["fit_y"])
        if point.get("raw_x") is not None and point.get("raw_y") is not None:
            return float(point["raw_x"]), float(point["raw_y"])
        return float(point.get("x", 0.0)), float(point.get("y", 0.0))

    upper = rim.cy - rim.r * 0.78
    lower = rim.cy + rim.r * 0.50
    x_gate = rim.r * 1.30
    near_gate = rim.r * 2.20

    nearest_dist = 1e9
    near_count = 0
    crossing_frame = None
    crossing_time = None

    for i in range(1, len(points)):
        prev = points[i - 1]
        curr = points[i]
        px, py = point_xy(prev)
        cx, cy = point_xy(curr)

        nearest_dist = min(nearest_dist, math.hypot(cx - rim.cx, cy - rim.cy))
        if math.hypot(cx - rim.cx, cy - rim.cy) <= near_gate:
            near_count += 1

        desc = (cy - py) > max(0.8, rim.r * 0.015)
        cross_vertical = py <= upper and cy >= lower
        near_center_x = abs(px - rim.cx) <= x_gate and abs(cx - rim.cx) <= x_gate
        if desc and cross_vertical and near_center_x:
            crossing_frame = int(curr.get("frame", i))
            crossing_time = float(curr.get("time", crossing_frame / max(1.0, fps)))
            break

    if crossing_frame is not None:
        center_penalty = min(1.0, nearest_dist / max(1.0, rim.r * 1.5))
        confidence = max(0.52, min(0.98, 0.92 - 0.28 * center_penalty + min(0.12, near_count * 0.008)))
        return {
            "label": "Basket",
            "confidence": round(float(confidence), 4),
            "crossing_frame": int(crossing_frame),
            "crossing_time": round(float(crossing_time), 3),
            "reason": "rim_window_crossing",
        }

    if near_count >= 2:
        confidence = max(0.55, min(0.95, 0.64 + min(0.26, near_count * 0.01)))
        return {
            "label": "No Basket",
            "confidence": round(float(confidence), 4),
            "crossing_frame": None,
            "crossing_time": None,
            "reason": "near_rim_without_crossing",
        }

    return {
        "label": "Unknown",
        "confidence": round(float(max(0.0, min(0.45, 0.2 + (1.0 / max(1.0, nearest_dist)) * 18.0))), 4),
        "crossing_frame": None,
        "crossing_time": None,
        "reason": "trajectory_far_from_rim",
    }

def _probe_codec(output_dir: str, fourcc_name: str, fps: float, frame_size: Tuple[int, int]) -> bool:
    w, h = frame_size
    if w <= 0 or h <= 0:
        return False
    path = os.path.join(output_dir, f"_codec_probe_{fourcc_name}.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc_name), fps, frame_size)
    if not writer.isOpened():
        writer.release()
        return False
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    for _ in range(3):
        writer.write(blank)
    writer.release()
    cap = cv2.VideoCapture(path)
    ok = cap.isOpened()
    read_ok, _ = cap.read()
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    try:
        os.remove(path)
    except OSError:
        pass
    return bool(ok and (read_ok or count > 0))


def _create_video_writer(output_path: str, fps: float, frame_size: Tuple[int, int]) -> Tuple[Optional[cv2.VideoWriter], str]:
    out_dir = os.path.dirname(output_path)
    fps = max(12.0, min(120.0, float(fps)))
    for fourcc_name, label in [("avc1", "h264"), ("H264", "h264"), ("mp4v", "mp4v")]:
        if not _probe_codec(out_dir, fourcc_name, fps, frame_size):
            continue
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*fourcc_name), fps, frame_size)
        if writer.isOpened():
            return writer, label
        writer.release()
    return None, "unknown"


def process_trajectory_video(
    video_path: str,
    output_dir: str,
    rim_calibration: Optional[Dict[str, Any]] = None,
    cancel_check: CancelCheck = lambda: False,
    progress_callback: ProgressCallback = lambda p: None,
    state_callback: StateCallback = lambda payload: None,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Cannot open video: {video_path}"}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if total_frames <= 0 or width <= 0 or height <= 0:
        cap.release()
        return {"error": "Invalid video metadata"}

    backend_dir = os.path.dirname(__file__)
    if not os.getenv("YOLO_CONFIG_DIR"):
        yolo_cfg_root = os.path.join(backend_dir, ".ultralytics_cfg")
        os.makedirs(yolo_cfg_root, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = yolo_cfg_root
    model_hint = os.getenv("TRAJ_BALL_MODEL", "yolo11l.pt")
    model_path = _resolve_ball_model_path(model_hint=model_hint, backend_dir=backend_dir)
    det_stride = _read_int_from_env("TRAJ_DET_STRIDE", 1, min_value=1, max_value=6)
    detector = CompositeBallDetector(model_path=model_path)
    rim = _normalize_rim_calibration(rim_calibration=rim_calibration, width=width, height=height)
    pose_trigger = PoseShotTrigger(frame_width=width, frame_height=height)
    tracker = TrajectoryTracker(width, height, fps=fps, rim=rim, pose_trigger=pose_trigger)
    output_filename = f"trajectory_{uuid.uuid4().hex[:10]}.mp4"
    output_path = os.path.join(output_dir, output_filename)
    writer, output_codec = _create_video_writer(output_path, fps=fps, frame_size=(width, height))
    if writer is None:
        cap.release()
        return {"error": "Cannot create output video writer"}
    progress_callback(0)
    processed_frames = 0
    failed_frames = 0
    first_frame_error = ""
    while True:
        if cancel_check():
            cap.release()
            writer.release()
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
            except OSError:
                pass
            return {"cancelled": True}
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx = processed_frames
        t = frame_idx / max(1.0, fps)
        try:
            detections = detector.detect(frame) if (frame_idx % det_stride == 0) else []
            lowlight_active = bool((detector.last_frame_meta or {}).get("lowlight_active", False))
            point = tracker.update(
                frame_idx=frame_idx,
                time_s=t,
                detections=detections,
                frame_bgr=frame,
                lowlight_active=lowlight_active,
            )
            live_state = tracker.get_live_state(frame_idx)
            writer.write(_draw_overlay(frame, point, tracker.points, detector.mode_name, frame_idx, live_state.get("phase", "pre_shot"), int(live_state.get("lag_frames", 0))))
            progress_callback(int(min(99, ((frame_idx + 1) / max(1, total_frames)) * 100)))
            state_callback(live_state)
        except Exception as exc:
            failed_frames += 1
            if not first_frame_error:
                first_frame_error = str(exc)
            writer.write(frame)
        processed_frames += 1
    cap.release()
    writer.release()
    progress_callback(100)
    if processed_frames == 0:
        return {"error": "No frame processed"}
    if tracker.current_event is not None:
        end_frame = max(0, processed_frames - 1)
        end_time = float(end_frame / max(1.0, fps))
        tracker._terminate_shot_event(frame_idx=end_frame, time_s=end_time, stop_reason="TIMEOUT_STOP")
    tracking = tracker.finalize_metrics(total_frames=processed_frames)
    tracking["detector_mode"] = detector.mode_name
    tracking["detector_model"] = model_path
    tracking["detector_model_hint"] = model_hint
    tracking["detector_yolo_conf_min"] = _read_float_from_env("TRAJ_YOLO_CONF_MIN", 0.08, min_value=0.01, max_value=0.95)
    tracking["detector_yolo_imgsz"] = _read_int_from_env("TRAJ_YOLO_IMGSZ", 960, min_value=320, max_value=1600)
    tracking["detector_stride"] = int(det_stride)
    tracking["detector_device"] = os.getenv("TRAJ_YOLO_DEVICE", "0" if torch.cuda.is_available() else "cpu")
    tracking["detector_takeover_missing"] = _read_int_from_env("TRAJ_YOLO_TAKEOVER_MISSING", 8, min_value=1, max_value=20)
    tracking["detector_fallback_min_conf"] = _read_float_from_env("TRAJ_FALLBACK_MIN_CONF", 0.24, min_value=0.0, max_value=1.0)
    tracking["detector_fallback_min_motion"] = _read_float_from_env("TRAJ_FALLBACK_MIN_MOTION", 0.030, min_value=0.0, max_value=1.0)
    tracking["detector_fallback_require_both"] = _read_bool_from_env("TRAJ_FALLBACK_REQUIRE_BOTH", True)
    tracking["detector_motion_rescue_enabled"] = _read_bool_from_env("TRAJ_ENABLE_MOTION_RESCUE", False)
    tracking["detector_fallback_reason"] = detector.fallback_reason
    tracking["detector_lowlight_enable"] = detector.lowlight_enable
    tracking["detector_lowlight_luma_th"] = detector.lowlight_luma_th
    tracking["detector_lowlight_conf_ratio"] = detector.lowlight_conf_ratio
    tracking["detector_lowlight_aux_strict_ratio"] = detector.lowlight_aux_strict_ratio
    tracking["detector_lowlight_dual_branch"] = detector.lowlight_dual_branch
    tracking["shot_trigger_mode"] = tracker.shot_trigger_mode
    tracking["shot_gate_strict"] = tracker.shot_gate_strict
    tracking["shot_timeout_s"] = tracker.shot_timeout_s
    tracking["event_reset_frames"] = tracker.event_reset_frames
    tracking["shot_min_event_gap_s"] = tracker.shot_min_event_gap_s
    tracking["shot_post_cooldown_s"] = tracker.shot_post_cooldown_s
    tracking["shot_require_release_transition"] = tracker.require_release_transition
    tracking["shot_release_block_fallback_th"] = tracker.release_block_fallback_th
    tracking["pose_model"] = pose_trigger.model_hint
    tracking["pose_model_available"] = pose_trigger.available
    tracking["pose_model_error"] = pose_trigger.model_error
    tracking["pose_stride"] = int(pose_trigger.infer_stride)
    tracking["pose_device"] = pose_trigger.device
    tracking["frame_failures"] = failed_frames
    tracking["frame_error_sample"] = first_frame_error
    tracking["max_missing"] = tracker.max_missing
    final_live = tracker.get_live_state(max(0, processed_frames - 1))
    shot_prediction = tracker.get_latest_shot_prediction(fps=fps)
    shot_events = tracker.get_shot_events()
    state_callback(final_live)
    return {
        "task_type": "trajectory",
        "created_at": datetime.now().isoformat(),
        "status": "completed",
        "video_info": {
            "filename": os.path.basename(video_path),
            "total_frames": int(processed_frames),
            "fps": round(float(fps), 2),
            "width": int(width),
            "height": int(height),
            "duration": round(float(processed_frames / max(1.0, fps)), 2),
            "output_codec": output_codec,
        },
        "analysis": {
            "tracking": tracking,
            "trajectory_points": tracker.points,
            "fit_trajectory_points": [{"frame": p["frame"], "time": p["time"], "x": p["fit_x"], "y": p["fit_y"]} for p in tracker.points if p.get("fit_valid")],
            "predicted_trajectory_points": tracker.get_predicted_trajectory_points(),
            "shot_events": shot_events,
            "live_summary": final_live,
            "rim_calibration": (
                {"cx": round(float(rim.cx), 2), "cy": round(float(rim.cy), 2), "r": round(float(rim.r), 2)}
                if rim is not None
                else None
            ),
            "shot_prediction": shot_prediction,
        },
        "metrics": {
            "detected_points": tracking["detected_points"],
            "predicted_points": tracking["predicted_points"],
            "coverage_percent": tracking["coverage_percent"],
            "continuity_percent": tracking["continuity_percent"],
            "suppressed_false_positives": tracking["suppressed_false_positives"],
            "head_false_positive_suppressed": tracking.get("head_false_positive_suppressed", 0),
            "false_joint_suppressed": tracking.get("false_joint_suppressed", 0),
            "body_zone_suppressed": tracking.get("body_zone_suppressed", 0),
            "hsv_assist_hits": tracking.get("hsv_assist_hits", 0),
            "fit_quality_score": tracking.get("fit_quality_score", 0),
            "predicted_trajectory_points": len(tracker.get_predicted_trajectory_points()),
            "source_ratio": tracking.get("source_ratio", {}),
            "static_hotspot_topk": tracking.get("static_hotspot_topk", []),
            "reject_reason_histogram": tracking.get("reject_reason_histogram", {}),
            "shot_count": tracking.get("shot_count", 0),
            "made_count": tracking.get("made_count", 0),
            "miss_count": tracking.get("miss_count", 0),
            "event_retrigger_count": tracking.get("event_retrigger_count", 0),
            "max_consecutive_miss_in_flight": tracking.get("max_consecutive_miss_in_flight", 0),
            "blocked_release_count": tracking.get("blocked_release_count", 0),
            "release_block_max_streak": tracking.get("release_block_max_streak", 0),
            "release_candidate_count": tracking.get("release_candidate_count", 0),
            "pose_person_found_ratio": tracking.get("pose_person_found_ratio", 0),
        },
        "artifacts": {
            "annotated_video": output_filename,
            "annotated_video_url": f"/uploads/trajectory/{output_filename}",
        },
    }

