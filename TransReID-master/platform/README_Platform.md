# BallShow Pro 智能球员检索与高光生成平台

本项目是基于超越国一指标的 **Dual-ViT TransReID (Rank-1 94.4%, mAP 91.8%)** 算法打造的商业级 Web 演示平台。采用轻量化、零构建的前后端分离架构，特别适合用于路演答辩与演示。

## 🚀 架构特性 (The Four-End Ecosystem)

1. **AI 中控大脑 (FastAPI)**：全异步非阻塞，内置 SQLite 数据库，双 ViT 模型常驻显存，支持毫秒级图解图 (ImageSearch) 和异步视频逐帧解析 (VideoAnalysis)。
2. **教练大屏前端 (Vue 3 + Element Plus)**：基于 CDN ESM 规范编写，**无需安装 Node.js，无需 npm install，无需打包**。Python 启动后直接可在浏览器流畅运行带有暗黑主题、ECharts 动态图表的高级界面。
3. **引擎加持**：融合了 Ultralytics YOLOv8n 的极速人体检测和 TransReID 强大特征提取。

---

## 🛠️ 环境依赖与安装

请确保你已经拥有了最初跑 TransReID 算法的 Conda 环境（包含 `torch`, `torchvision`, `timm`）。
然后，在此环境中补充平台所需的 Web 依赖：

```bash
cd platform
pip install -r requirements_platform.txt
```

---

## 📂 目录结构与模型权重初始化

在启动前，你需要把云端训练好的两个 `.pth` 模型权重放到本地电脑的指定目录：

```text
TransReID-master/
├── data/BallShow/bounding_box_test/  <-- 这是平台的默认 Gallery 图库来源
├── logs/                             <-- 默认从这里读取模型权重
│   ├── BallShow_4090_opt5/transformer_120.pth  (256x128 ViT)
│   └── BallShow_4090_opt/transformer_120.pth   (384x128 ViT)
└── platform/
    ├── backend/
    │   ├── weights/             <-- 或者你可以把模型移动到这个文件夹下
    │   │   ├── yolov8n.pt       <-- YOLOv8 权重 (如果本地没有，首次运行视频分析时会自动下载)
    │   ├── app.py               <-- FastAPI 核心启动文件
    │   ├── reid_engine.py       <-- AI 引擎
    │   ├── video_engine.py      <-- 视频高光分析引擎
    │   └── database.py          <-- SQLite 数据库
    ├── frontend/                <-- ES Module Vue 3 前端 (没有任何构建工具)
    └── requirements_platform.txt
```

*提示：`backend/app.py` 中有路径自动检测逻辑，只要权重在 `logs` 目录或者 `platform/backend/weights/`，系统都能自动找得到。*

---

## 🏎️ 启动平台

在命令行中执行以下命令（注意是在 `TransReID-master` 根目录下，由 uvicorn 引导 `app.py` 相对路径运行）：

```bash
# 激活你的环境
conda activate YOUR_ENV_NAME

# 返回 TransReID 源码根目录
cd /path/to/TransReID-master

# 启动服务端 (Uvicorn)
python platform/backend/app.py
```

### 启动成功标志
如果看到以下字样，说明不仅 Web 挂载成功，双 ViT 引擎也已经全部读入显存就绪：

```
============================================================
  BallShow 智能球员检索平台 v1.0
  Powered by TransReID Dual-ViT Ensemble
============================================================
[DB] 数据库初始化完毕 ✓
[ReIDEngine] 正在初始化双模型 Ensemble...
...
[ReIDEngine] Gallery 构建完毕: 4858 张图, 特征维度 (4858, 768)
[ReIDEngine] 初始化完毕 ✓
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

---

## 💻 访问与使用

1. 打开浏览器（推荐 Chrome 或 Edge），访问：[http://localhost:8080](http://localhost:8080)
2. **注册账号**：随意注册一个 admin 账号进入系统。
3. **图像检索 (ImageSearch)**：进入菜单上传一张球员的图片，系统会在几十毫秒内比对显存矩阵，返回 Top-10 最相似的结果。
4. **视频分析 (VideoAnalysis)**：
   - 上传一张 Query 图。
   - 上传一段 `.mp4` 录像（推荐 10-20秒 的短视频用于路演）。
   - 提交任务后，后台会触发异步 YOLO 检测，前端会轮询展示“处理进度”。
   - 任务完成后，将在右侧时间轴精准标注主角出现的具体时间段！

---
*Developed by Team Antigravity.*
