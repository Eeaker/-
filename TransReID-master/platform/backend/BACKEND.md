# BallShow 后端服务层文档 (Backend Layer)

## 1. 架构与职能
后端服务采用 **FastAPI** 框架，作为连接 AI 模型引擎（ReID + YOLOv8）与各个前端（Web / APP / 小程序）的中心枢纽。它负责图像/视频接收、任务调度、数据库持久化及认证鉴权。

### 核心技术栈
- **Web 框架**: FastAPI (异步高性能，自动生成 OpenAPI 文档)
- **数据库**: SQLite (通过官方 `sqlite3` 模块进行轻量级读写)
- **认证**: JWT (JSON Web Token) + `bcrypt` 密码哈希
- **视频与图像**: `opencv-python` (逐帧读取), `moviepy` (视频切割与拼接)
- **AI 引擎对接**: 动态加载 `ultralytics` (YOLO) 和 `torch` (TransReID)

## 2. 核心模块说明

| 文件 | 描述 |
|---|---|
| `app.py` | FastAPI 主入口，挂载跨域中间件(CORS)、静态文件目录(StaticFiles)，并在启动时初始化 DB 和 ReID 引擎。 |
| `database.py` | 封装了所有对 `ballshow.db` 的 CRUD 操作。包含 `users`, `search_history`, `video_tasks` 三张业务表。 |
| `auth.py` | JWT 签发、校验及密码加密逻辑，提供 `get_current_user` 依赖注入用于保护路由。 |
| `reid_engine.py`| TransReID 双模型融合推理引擎。启动时加载权重构建 Gallery 特征库，运行中提供实时的图片 Top-K 检索。 |
| `video_engine.py`| 视频分析引擎。使用 YOLOv8 逐帧寻找人体 BBox -> 裁剪 -> TransReID 匹配 -> 时间轴聚合融合 -> MoviePy 生成最终 Highlight 视频。 |

## 3. 路由设计 (Routes)

所有路由前缀均为 `/api`：

- **认证** (`routes/auth_routes.py`):
  - `POST /api/login`: 登录获取 Token。
  - `POST /api/register`: 注册新用户。
- **大屏监控** (`routes/dashboard_routes.py`):
  - `GET /api/dashboard`: 获取系统状态、GPU 在线情况、模型精度及使用量统计。
- **图片检索** (`routes/reid_routes.py`):
  - `POST /api/reid/search`: 接收截图，调用 `ReIDEngine` 返回 Top-K 匹配列表。
- **视频分析 (异步轮询机制)** (`routes/video_routes.py`):
  - *采用了长任务的两阶段断点上传和轮询策略应对云端和移动端的超时限制。*
  - `POST /api/video/query/upload`: 第一阶段，上传查询图片。
  - `POST /api/video/analyze`: 第二阶段，上传庞大的视频文件，后台启动 `video_engine` 开始逐帧分析。
  - `GET /api/video/task/{id}`: 客户端轮询获取实时分析进度百分比。
  - `GET /api/video/history`: 获取当前用户的视频分析历史。

## 4. 目录结构
```text
platform/backend/
├── app.py                  # 服务入口
├── auth.py                 # JWT 安全模块
├── database.py             # SQLite 交互
├── reid_engine.py          # ViT 特征引擎
├── video_engine.py         # YOLO 视频逻辑
├── ballshow.db             # 数据库文件
├── routes/                 # 接口路由层
├── uploads/
│   ├── query/              # 查询图片临时缓存
│   └── videos/             # 原始视频与生成的高光视频 (highlight_*.mp4)
└── weights/                # 模型依赖权重 (yolov8n.pt 等)
```

## 5. 启动方式
在配置好 Python 虚拟环境并安装 `requirements_platform.txt` 后：
```bash
# Windows
python app.py
```
服务默认运行在 `http://127.0.0.1:8000` (允许局域网通过 IPv4 访问)。
自带 Swagger UI 调试文档：`http://127.0.0.1:8000/docs`
