"""
视频分析路由：视频上传 → YOLOv8 检测 + ReID 匹配
"""
import os
import uuid
import json
import threading
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Request
from ..auth import get_current_user
from ..reid_engine import get_engine
from ..runtime_paths import QUERY_DIR, VIDEOS_DIR, ensure_runtime_dirs
from ..video_engine import process_video
from ..database import create_video_task, update_video_task, get_video_task, get_user_video_history

router = APIRouter(prefix="/api/video", tags=["视频分析"])

ensure_runtime_dirs()

# 全局任务取消标志
active_tasks = {}


def _run_analysis(task_id: int, video_path: str, query_path: str):
    """后台线程：执行视频分析"""
    try:
        update_video_task(task_id, status="processing")
        engine = get_engine()
        
        # 传入可由路由随时停止的回调函数
        result = process_video(
            video_path, 
            query_path, 
            engine,
            cancel_check=lambda: active_tasks.get(task_id, {}).get("cancelled", False),
            progress_callback=lambda p: active_tasks.setdefault(task_id, {}).update({"progress": p})
        )
        
        # 如果提早返回并且标记了已被取消
        if result.get("cancelled"):
            update_video_task(task_id, status="cancelled", result_json=json.dumps({"error": "由于用户终止，任务已取消"}))
        else:
            update_video_task(
                task_id,
                status="completed",
                total_frames=result.get("video_info", {}).get("total_frames", 0),
                matched_segments=len(result.get("analysis", {}).get("segments", [])),
                result_json=json.dumps(result, ensure_ascii=False),
                finished_at=__import__("datetime").datetime.now().isoformat(),
            )
    except Exception as e:
        update_video_task(task_id, status="failed", result_json=json.dumps({"error": str(e)}))
    finally:
        # 清理内存中的状态
        if task_id in active_tasks:
            del active_tasks[task_id]


@router.post("/query/upload")
async def upload_video_query(
    query: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    """单独上传查询目标（适用于微信小程序限制单文件的环境）"""
    q_ext = os.path.splitext(query.filename or "q.jpg")[1] or ".jpg"
    q_name = f"{uuid.uuid4().hex}{q_ext}"
    q_path = os.path.join(QUERY_DIR, q_name)
    with open(q_path, "wb") as f:
        content = await query.read()
        f.write(content)
    return {"query_filename": q_name}


@router.post("/analyze")
async def analyze_video(
    request: Request,
    user: dict = Depends(get_current_user),
):
    """上传视频和查询图片标志，启动异步分析任务"""
    form = await request.form()
    video = form.get("video")
    query_filename = form.get("query_filename")
    
    if not video or not query_filename:
        raise HTTPException(400, "必须上传 video 字段以及提供 query_filename")
    # 保存视频
    v_ext = os.path.splitext(video.filename or "v.mp4")[1] or ".mp4"
    v_name = f"{uuid.uuid4().hex}{v_ext}"
    v_path = os.path.join(VIDEOS_DIR, v_name)
    with open(v_path, "wb") as f:
        content = await video.read()
        f.write(content)

    # 获取已上传的图片
    q_name = query_filename
    q_path = os.path.join(QUERY_DIR, q_name)
    
    if not os.path.exists(q_path):
        raise HTTPException(422, "提供的 query_filename 不存在或已过期")

    # 创建任务记录
    task_id = create_video_task(user["user_id"], v_name, q_name)

    # 注册运行状态（默认不取消与初始进度）
    active_tasks[task_id] = {"cancelled": False, "progress": 0}

    # 启动后台分析线程
    t = threading.Thread(target=_run_analysis, args=(task_id, v_path, q_path), daemon=True)
    t.start()

    return {
        "message": "视频分析任务已提交",
        "task_id": task_id,
        "video_filename": v_name,
    }


@router.get("/task/{task_id}")
async def get_task_status(task_id: int, user: dict = Depends(get_current_user)):
    """查询视频分析任务状态"""
    task = get_video_task(task_id)
    if not task:
        raise HTTPException(404, "任务不存在")

    result = {
        "task_id": task["id"],
        "status": task["status"],
        "video_filename": task["video_filename"],
        "total_frames": task["total_frames"],
        "matched_segments": task["matched_segments"],
        "created_at": task["created_at"],
        "finished_at": task["finished_at"],
    }
    
    if task["status"] == "processing" and task_id in active_tasks:
        result["progress"] = active_tasks[task_id].get("progress", 0)

    if task["status"] == "completed":
        result["analysis"] = json.loads(task["result_json"])
    elif task["status"] in ["failed", "cancelled"]:
        result["error"] = json.loads(task["result_json"]).get("error", "未知错误")

    return result

@router.post("/cancel/{task_id}")
async def cancel_task(task_id: int, user: dict = Depends(get_current_user)):
    """标记某个运行中的视频任务为取消"""
    task = get_video_task(task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    if task["user_id"] != user["user_id"]:
        raise HTTPException(403, "无权操作此任务")
        
    if task["status"] == "processing":
        if task_id not in active_tasks:
            active_tasks[task_id] = {"cancelled": False, "progress": 0}
        active_tasks[task_id]["cancelled"] = True
        return {"message": "取消指令已发送"}
    else:
        return {"message": f"当前任务状态为 {task['status']}，无法取消"}


@router.get("/history")
async def get_history(user: dict = Depends(get_current_user)):
    """获取用户所有历史视频任务"""
    history = get_user_video_history(user["user_id"])
    for item in history:
        if item["status"] == "completed":
            item["analysis"] = json.loads(item["result_json"])
        elif item["status"] in ["failed", "cancelled"]:
            item["error"] = json.loads(item["result_json"]).get("error", "未知错误")
        # 清除冗余字段
        item.pop("result_json", None)
    return history
