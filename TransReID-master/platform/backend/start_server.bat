@echo off
setlocal

if "%BALLSHOW_CONDA_ENV%"=="" (
    set "BALLSHOW_CONDA_ENV=deepstudy"
)

if "%YOLO_CONFIG_DIR%"=="" (
    set "YOLO_CONFIG_DIR=%~dp0.ultralytics_cfg"
)

if "%REID_SINGLE_WEIGHT_PATH%"=="" (
    set "REID_SINGLE_WEIGHT_PATH=%~dp0..\..\logs\92.6\transformer_120.pth"
)

if "%BA_REPO_PATH%"=="" (
    set "BA_REPO_PATH=%~dp0..\..\..\_tmp_repo_basketball_analysis"
)

if not exist "%YOLO_CONFIG_DIR%" (
    mkdir "%YOLO_CONFIG_DIR%"
)

if "%TRAJ_BALL_MODEL%"=="" (
    set "TRAJ_BALL_MODEL=yolo11m.pt"
)

if "%TRAJ_YOLO_IMGSZ%"=="" (
    set "TRAJ_YOLO_IMGSZ=640"
)

if "%TRAJ_YOLO_CONF_MIN%"=="" (
    set "TRAJ_YOLO_CONF_MIN=0.08"
)

if "%TRAJ_YOLO_TAKEOVER_MISSING%"=="" (
    set "TRAJ_YOLO_TAKEOVER_MISSING=8"
)

if "%TRAJ_TAKEOVER_AUX_LIMIT%"=="" (
    set "TRAJ_TAKEOVER_AUX_LIMIT=1"
)

if "%TRAJ_ENABLE_HSV_WHITE%"=="" (
    set "TRAJ_ENABLE_HSV_WHITE=0"
)

if "%TRAJ_SHOT_TRIGGER_MODE%"=="" (
    set "TRAJ_SHOT_TRIGGER_MODE=pose_track"
)

if "%TRAJ_POSE_MODEL%"=="" (
    set "TRAJ_POSE_MODEL=yolo11s-pose.pt"
)

if "%TRAJ_YOLO_DEVICE%"=="" (
    set "TRAJ_YOLO_DEVICE=0"
)

if "%TRAJ_POSE_DEVICE%"=="" (
    set "TRAJ_POSE_DEVICE=0"
)

if "%TRAJ_YOLO_HALF%"=="" (
    set "TRAJ_YOLO_HALF=1"
)

if "%TRAJ_POSE_HALF%"=="" (
    set "TRAJ_POSE_HALF=1"
)

if "%TRAJ_POSE_STRIDE%"=="" (
    set "TRAJ_POSE_STRIDE=2"
)

if "%TRAJ_DET_STRIDE%"=="" (
    set "TRAJ_DET_STRIDE=2"
)

if "%TRAJ_POSE_CONF%"=="" (
    set "TRAJ_POSE_CONF=0.16"
)

if "%TRAJ_SHOT_GATE_STRICT%"=="" (
    set "TRAJ_SHOT_GATE_STRICT=high_precision"
)

if "%TRAJ_LOWLIGHT_ENABLE%"=="" (
    set "TRAJ_LOWLIGHT_ENABLE=1"
)

if "%TRAJ_LOWLIGHT_LUMA_TH%"=="" (
    set "TRAJ_LOWLIGHT_LUMA_TH=75"
)

if "%TRAJ_LOWLIGHT_DUAL_BRANCH%"=="" (
    set "TRAJ_LOWLIGHT_DUAL_BRANCH=0"
)

if "%TRAJ_LOWLIGHT_CONF_RATIO%"=="" (
    set "TRAJ_LOWLIGHT_CONF_RATIO=0.78"
)

if "%TRAJ_LOWLIGHT_AUX_STRICT_RATIO%"=="" (
    set "TRAJ_LOWLIGHT_AUX_STRICT_RATIO=1.25"
)

if "%TRAJ_SHOT_TIMEOUT_S%"=="" (
    set "TRAJ_SHOT_TIMEOUT_S=1.5"
)

if "%TRAJ_EVENT_RESET_FRAMES%"=="" (
    set "TRAJ_EVENT_RESET_FRAMES=8"
)

if "%TRAJ_SHOT_MIN_EVENT_GAP_S%"=="" (
    set "TRAJ_SHOT_MIN_EVENT_GAP_S=0.45"
)

if "%TRAJ_SHOT_POST_COOLDOWN_S%"=="" (
    set "TRAJ_SHOT_POST_COOLDOWN_S=0.35"
)

if "%TRAJ_SHOT_REQUIRE_RELEASE_TRANSITION%"=="" (
    set "TRAJ_SHOT_REQUIRE_RELEASE_TRANSITION=1"
)

if "%TRAJ_RELEASE_BLOCK_FALLBACK_TH%"=="" (
    set "TRAJ_RELEASE_BLOCK_FALLBACK_TH=6"
)

call conda activate %BALLSHOW_CONDA_ENV%
python app.py
pause
