"""
认证路由：注册、登录
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..auth import hash_password, verify_password, create_access_token
from ..database import create_user, get_user_by_username

router = APIRouter(prefix="/api", tags=["认证"])


class RegisterRequest(BaseModel):
    username: str
    password: str
    nickname: str = ""


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/register")
async def register(req: RegisterRequest):
    username = (req.username or "").strip()
    password = req.password or ""
    nickname = (req.nickname or "").strip()

    if not username:
        raise HTTPException(400, "用户名不能为空")
    if not password:
        raise HTTPException(400, "密码不能为空")

    existing = get_user_by_username(username)
    if existing:
        raise HTTPException(400, "用户名已存在")

    hashed = hash_password(password)
    user_id = create_user(username, hashed, nickname or username)

    token = create_access_token({"user_id": user_id, "username": username})
    return {
        "message": "注册成功",
        "token": token,
        "user": {"id": user_id, "username": username, "nickname": nickname or username},
    }


@router.post("/login")
async def login(req: LoginRequest):
    username = (req.username or "").strip()
    password = req.password or ""
    if not username or not password:
        raise HTTPException(401, "用户名或密码错误")

    user = get_user_by_username(username)
    if not user:
        raise HTTPException(401, "用户名或密码错误")

    if not verify_password(password, user["password_hash"]):
        raise HTTPException(401, "用户名或密码错误")

    token = create_access_token({"user_id": user["id"], "username": user["username"]})
    return {
        "message": "登录成功",
        "token": token,
        "user": {"id": user["id"], "username": user["username"], "nickname": user["nickname"]},
    }
