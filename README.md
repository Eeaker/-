# BallShow Project Recovery Guide

This repository uses a unified entry for analysis:
- `Basketball Intelligence (game-analysis)` is the primary feature.
- Legacy `video-analysis` and `trajectory-analysis` pages were merged into the unified workflow.

---

## One-click environment recovery (Windows)

### 1) Open project root
```powershell
cd C:\Users\23159\Downloads\TransReID-master
```

### 2) Pull submodule
```powershell
git submodule update --init --recursive
```

### 3) Run bootstrap script
```powershell
powershell -ExecutionPolicy Bypass -File .\TransReID-master\platform\backend\tools\bootstrap_env.ps1 -CondaEnv deepstudy -InstallDeps
```

What it does:
- Sets writable `YOLO_CONFIG_DIR`
- Sets default `REID_SINGLE_WEIGHT_PATH`
- Sets default `BA_REPO_PATH` (third-party basketball repo path)
- Installs `platform/requirements_platform.txt` (optional with `-InstallDeps`)
- Initializes runtime upload directories
- Runs asset checks (`tools/check_assets.py`)

### 4) Start backend
```powershell
conda run -n deepstudy python .\TransReID-master\platform\backend\app.py
```

Then open:
- [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## Required assets checklist

The checker is:
```powershell
conda run -n deepstudy python .\TransReID-master\platform\backend\tools\check_assets.py
```

Required:
- ReID single weight:
  - default: `TransReID-master\logs\92.6\transformer_120.pth`
  - override with env: `REID_SINGLE_WEIGHT_PATH`
- Gallery directory:
  - default: `data\BallShow\bounding_box_test`

Optional (recommended):
- YOLO model files in `TransReID-master\platform\backend\` (small runtime models are now committed)
- Third-party basketball repo (tracked as submodule): `third_party\basketball_analysis`
- Third-party models in `_tmp_repo_basketball_analysis\models\` (or custom `BA_REPO_PATH`)

---

## About uploading model files / third-party assets to GitHub

Short answer: possible, but not all in normal Git history.

Current practical constraints:
- GitHub normal Git has a hard 100MB per-file limit.
- Your key files include:
  - `transformer_120.pth` ~455MB
  - `court_keypoint_detector.pt` ~418MB
  - other `.pt` files ~170MB+

Current status in this repo:
- Uploaded:
  - `yolo11l.pt`, `yolo11m.pt`, `yolo11s-pose.pt`, `yolov8n.pt`
  - third-party code via submodule `third_party/basketball_analysis`
- Not uploaded (too large for normal Git):
  - `logs/92.6/transformer_120.pth`
  - third-party heavy models under `_tmp_repo_basketball_analysis/models/`

Recommended strategy:
1. Keep code in Git (already done).
2. Put large model files into **Git LFS** or **GitHub Release assets**.
3. Keep dataset folders (`data/`) out of Git history.

If you want, next step can be:
- add `.gitattributes` for LFS patterns
- upload selected model pack (for example only runtime-needed models)
- keep download links in `tools/check_assets.py` and README.

---

## Trajectory module: delete or modify?

Recommendation: **modify, do not hard-delete core engine**.

Use this structure:
- Keep `trajectory_engine.py` as algorithm module (library role)
- Keep old trajectory/video APIs as compatibility layer (deprecated)
- Continue feature development only in unified `game-analysis`
- Use possession tracking ball points as primary source for trajectory rendering and future curve prediction

Why:
- avoids breaking old clients/scripts
- avoids duplicated logic
- keeps one source of truth for future improvements
