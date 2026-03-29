"""
ReID 检索路由：图片上传 → 特征提取 → Top-K 匹配
"""
import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, Depends, Query, Form
from ..auth import get_current_user
from ..reid_engine import get_engine
from ..database import add_search_history
from ..runtime_paths import QUERY_DIR, ensure_runtime_dirs
from PIL import Image

router = APIRouter(prefix="/api/reid", tags=["ReID 检索"])

ensure_runtime_dirs()
UPLOAD_DIR = QUERY_DIR


@router.post("/search")
async def search_by_image(
    file: UploadFile = File(...),
    topk: int = Form(default=10, ge=1, le=150),
    user: dict = Depends(get_current_user),
):
    """上传一张查询图片，返回 Gallery 中最相似的 Top-K 结果"""
    # 保存上传文件
    ext = os.path.splitext(file.filename or "img.jpg")[1] or ".jpg"
    save_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, save_name)

    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # 提取特征并检索
    engine = get_engine()
    query_img = Image.open(save_path).convert("RGB")
    query_feat = engine.extract_feature(query_img)
    results = engine.search(query_feat, topk=topk)

    top1_score = results[0]["score"] if results else 0.0
    add_search_history(user["user_id"], save_name, len(results), top1_score, "image")

    return {
        "query_image": f"/uploads/query/{save_name}",
        "topk": topk,
        "result_count": len(results),
        "results": results,
    }


@router.get("/gallery/stats")
async def gallery_stats(user: dict = Depends(get_current_user)):
    """获取 Gallery 图库统计信息"""
    engine = get_engine()
    total = len(engine.gallery_paths)

    # 统计不同 person_id 的数量
    person_ids = set()
    for p in engine.gallery_paths:
        fname = os.path.basename(p)
        pid = fname.split("_")[0] if "_" in fname else "unknown"
        person_ids.add(pid)

    return {
        "total_images": total,
        "total_identities": len(person_ids),
        "feature_dim": engine.gallery_feats.shape[1] if engine.gallery_feats is not None else 0,
    }
