param(
    [string]$CondaEnv = "deepstudy",
    [switch]$InstallDeps,
    [switch]$RunAssetCheck = $true
)

$ErrorActionPreference = "Stop"

$BackendDir = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $BackendDir)
$OuterRoot = Split-Path -Parent $ProjectRoot

$env:YOLO_CONFIG_DIR = Join-Path $BackendDir ".ultralytics_cfg"
if (-not (Test-Path $env:YOLO_CONFIG_DIR)) {
    New-Item -ItemType Directory -Path $env:YOLO_CONFIG_DIR | Out-Null
}

if (-not $env:REID_SINGLE_WEIGHT_PATH) {
    $env:REID_SINGLE_WEIGHT_PATH = Join-Path $ProjectRoot "logs\92.6\transformer_120.pth"
}
if (-not $env:BA_REPO_PATH) {
    $env:BA_REPO_PATH = Join-Path $OuterRoot "_tmp_repo_basketball_analysis"
}

Write-Host "== BallShow bootstrap ==" -ForegroundColor Cyan
Write-Host "Conda env             : $CondaEnv"
Write-Host "YOLO_CONFIG_DIR       : $env:YOLO_CONFIG_DIR"
Write-Host "REID_SINGLE_WEIGHT    : $env:REID_SINGLE_WEIGHT_PATH"
Write-Host "BA_REPO_PATH          : $env:BA_REPO_PATH"

if ($InstallDeps) {
    Write-Host ""
    Write-Host "[1/3] Installing Python dependencies..." -ForegroundColor Yellow
    conda run -n $CondaEnv python -m pip install -r (Join-Path (Split-Path -Parent $BackendDir) "requirements_platform.txt")
}

Write-Host ""
Write-Host "[2/3] Initializing runtime directories..." -ForegroundColor Yellow
conda run -n $CondaEnv python (Join-Path $BackendDir "tools\project_maintenance.py") --clear-uploads

if ($RunAssetCheck) {
    Write-Host ""
    Write-Host "[3/3] Checking assets..." -ForegroundColor Yellow
    conda run -n $CondaEnv python (Join-Path $BackendDir "tools\check_assets.py")
}

Write-Host ""
Write-Host "Done. Start backend with:" -ForegroundColor Green
Write-Host "  conda run -n $CondaEnv python $BackendDir\app.py"

