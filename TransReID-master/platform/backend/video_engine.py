"""
BallShow 平台 - 视频分析引擎
YOLOv8 人体检测 + ReID 匹配，输出目标球员出现的时间段
"""
import os
import cv2
import json
import uuid
import numpy as np
from PIL import Image
from datetime import datetime


def _ensure_ultralytics_writable_config():
    cfg_dir = os.getenv("YOLO_CONFIG_DIR", "").strip()
    if cfg_dir and os.path.isdir(cfg_dir) and os.access(cfg_dir, os.W_OK):
        return
    local_cfg = os.path.join(os.path.dirname(__file__), ".ultralytics_cfg")
    os.makedirs(local_cfg, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = local_cfg


def process_video(video_path: str, query_image_path: str, reid_engine,
                  similarity_threshold: float = 0.70,
                  frame_sample_interval: int = 15,
                  min_segment_frames: int = 10,
                  cancel_check=lambda: False,
                  progress_callback=lambda p: None) -> dict:
    """
    处理视频：逐帧检测人体 → 裁剪 → 与 query 特征比对 → 输出匹配片段

    Args:
        video_path: 视频文件路径
        query_image_path: 查询图片路径
        reid_engine: ReIDEngine 实例
        similarity_threshold: 相似度阈值（超过则认为是同一人）
        frame_sample_interval: 每隔多少帧采样一次（降低计算量）
        min_segment_frames: 最少连续匹配帧数（过滤闪烁噪声）

    Returns:
        dict: 分析结果，包含匹配片段列表和统计信息
    """
    # 提取 Query 特征
    query_img = Image.open(query_image_path).convert("RGB")
    query_feat = reid_engine.extract_feature(query_img)

    # 尝试加载 YOLOv8（如果可用）
    yolo_model = None
    try:
        _ensure_ultralytics_writable_config()
        from ultralytics import YOLO
        model_hint = os.getenv("VIDEO_PERSON_MODEL", "yolo11m.pt").strip() or "yolo11m.pt"
        yolo_weights = model_hint if os.path.isabs(model_hint) else os.path.join(os.path.dirname(__file__), model_hint)
        if not os.path.exists(yolo_weights):
            yolo_weights = "yolov8n.pt"
        yolo_model = YOLO(yolo_weights)
    except ImportError:
        print("[VideoEngine] ultralytics 未安装，将使用 OpenCV 人体检测作为回退方案")

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"无法打开视频: {video_path}"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    progress_callback(0)

    matched_frames = []  # 记录所有匹配的帧号
    frame_details = []   # 每个匹配帧的详细信息
    frame_idx = 0
    
    print(f"\n[VideoEngine] 开始分析视频: {os.path.basename(video_path)}", flush=True)
    print(f"[VideoEngine] 视频信息: FPS {fps:.1f}, 总帧数 {total_frames}, 分辨率 {width}x{height}", flush=True)
    print(f"[VideoEngine] 配置参数: 采样间隔 {frame_sample_interval} 帧, 相似度阈值 {similarity_threshold}", flush=True)
    print("-" * 50, flush=True)

    while True:
        if cancel_check():
            cap.release()
            return {"error": "任务已被用户取消", "cancelled": True}
        
        ret, frame = cap.read()
        if not ret:
            break

        # 按间隔采样
        if frame_idx % frame_sample_interval != 0:
            frame_idx += 1
            continue
            
        # 每处理 150 帧打印一次进度日志
        if frame_idx % (frame_sample_interval * 10) == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"[VideoEngine] 正在分析进度: {progress:.1f}% ({frame_idx}/{total_frames} 帧) ...", flush=True)
            progress_callback(int(progress))

        detections = _detect_persons(frame, yolo_model)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            # 安全裁剪
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 - x1 < 20 or y2 - y1 < 40:
                continue

            crop = frame[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)

            feat = reid_engine.extract_feature(crop_pil)
            sim = float(np.dot(query_feat, feat) / (
                np.linalg.norm(query_feat) * np.linalg.norm(feat) + 1e-8
            ))

            if sim >= similarity_threshold:
                time_sec = frame_idx / fps
                matched_frames.append(frame_idx)
                print(f"[Engine Match] Found target at {_format_time(time_sec)} (frame {frame_idx}), similarity: {sim:.2%}", flush=True)
                frame_details.append({
                    "frame": frame_idx,
                    "time": round(time_sec, 2),
                    "time_str": _format_time(time_sec),
                    "similarity": round(sim, 4),
                    "bbox": [x1, y1, x2, y2],
                })

        frame_idx += 1

    cap.release()
    progress_callback(100)
    print("-" * 50, flush=True)
    print(f"[VideoEngine] 视频帧遍历完成。共检测到 {len(matched_frames)} 个符合的主角帧。", flush=True)
    print("[VideoEngine] 正在融合时序特征轨迹 (Tracklets)...", flush=True)

    # 将零散帧聚合为连续片段
    segments = _merge_to_segments(frame_details, fps, frame_sample_interval, min_segment_frames)
    print(f"[VideoEngine] 轨迹融合完毕，聚合生成 {len(segments)} 个连续出场片段。", flush=True)

    # 拼接高光视频
    highlight_filename = None
    if segments:
        try:
            # 延迟导入以加快初始加载
            from moviepy.editor import VideoFileClip, concatenate_videoclips
            clips = []
            original_clip = VideoFileClip(video_path)
            for i, seg in enumerate(segments):
                # 确保不会超出视频总时长
                start_t = max(0, seg["start_time"])
                end_t = min(original_clip.duration, seg["end_time"])
                if end_t > start_t:
                    sub_clip = original_clip.subclip(start_t, end_t)
                    clips.append(sub_clip)
                    
            if clips:
                final_clip = concatenate_videoclips(clips)
                highlight_filename = f"highlight_{uuid.uuid4().hex[:8]}.mp4"
                highlight_path = os.path.join(os.path.dirname(video_path), highlight_filename)
                final_clip.write_videofile(highlight_path, codec="libx264", audio_codec="aac", logger=None, preset="fast")
                original_clip.close()
                final_clip.close()
                for c in clips:
                    c.close()
        except Exception as e:
            print(f"[VideoEngine] 生成拼接高光视频失败: {e}", flush=True)

    result = {
        "video_info": {
            "filename": os.path.basename(video_path),
            "fps": round(fps, 1),
            "total_frames": total_frames,
            "duration": round(total_frames / fps, 1),
            "resolution": f"{width}x{height}",
        },
        "analysis": {
            "sampled_frames": total_frames // frame_sample_interval,
            "matched_frames": len(matched_frames),
            "segments": segments,
            "highlight_video": highlight_filename,
            "total_appearance_time": round(
                sum(s["duration"] for s in segments), 1
            ),
        },
    }
    return result


def _detect_persons(frame, yolo_model=None):
    """使用 YOLO 或 OpenCV 检测画面中的人体"""
    detections = []

    if yolo_model is not None:
        results = yolo_model(frame, verbose=False, conf=0.3)
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                # COCO 类别 0 = person
                if cls_id == 0:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    detections.append({
                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": float(box.conf[0]),
                    })
    else:
        # OpenCV HOG 人体检测（回退方案）
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(4, 4), scale=1.05)
        for (x, y, w, h), weight in zip(boxes, weights):
            detections.append({
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "confidence": float(weight),
            })

    return detections


def _merge_to_segments(frame_details, fps, sample_interval, min_segment_frames):
    """将零散匹配帧聚合为连续时间片段"""
    if not frame_details:
        return []

    segments = []
    current_segment_frames = [frame_details[0]]

    for i in range(1, len(frame_details)):
        gap = frame_details[i]["frame"] - frame_details[i - 1]["frame"]
        # 如果两帧之间的间隔不超过 3 倍采样间隔，视为同一段
        if gap <= sample_interval * 3:
            current_segment_frames.append(frame_details[i])
        else:
            if len(current_segment_frames) >= max(1, min_segment_frames // sample_interval):
                segments.append(_build_segment(current_segment_frames, fps))
            current_segment_frames = [frame_details[i]]

    # 最后一段
    if len(current_segment_frames) >= max(1, min_segment_frames // sample_interval):
        segments.append(_build_segment(current_segment_frames, fps))

    return segments


def _build_segment(frames, fps):
    """构建单个片段对象"""
    # 前后各扩展 1 秒作为缓冲
    buffer_sec = 1.0
    start_time = max(0, frames[0]["time"] - buffer_sec)
    end_time = frames[-1]["time"] + buffer_sec
    avg_sim = sum(f["similarity"] for f in frames) / len(frames)
    return {
        "start_time": round(start_time, 2),
        "end_time": round(end_time, 2),
        "start_str": _format_time(start_time),
        "end_str": _format_time(end_time),
        "duration": round(end_time - start_time, 1),
        "avg_similarity": round(avg_sim, 4),
        "frame_count": len(frames),
        "best_similarity": max(f["similarity"] for f in frames),
    }


def _format_time(seconds):
    """将秒数格式化为 MM:SS"""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"
