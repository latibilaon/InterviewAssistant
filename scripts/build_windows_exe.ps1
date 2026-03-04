$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Output "[build] Host python:"
python --version

# CI 中必须使用 setup-python 注入的 python，而不是 py launcher（它可能选到 3.14）
$pyVer = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
Write-Output "[build] Using Python minor: $pyVer"
if ([version]$pyVer -ge [version]"3.13") {
  Write-Error "Python $pyVer is too new for current dependency stack. Please use Python 3.11/3.12."
  exit 1
}

if (!(Test-Path ".venv")) {
  python -m venv .venv
}
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install pyinstaller
if (Test-Path "requirements-packaging.txt") {
  python -m pip install -r requirements-packaging.txt
}
else {
  python -m pip install -r requirements.txt
}

if (Test-Path build) { Remove-Item build -Recurse -Force }
if (Test-Path dist) { Remove-Item dist -Recurse -Force }
if (Test-Path release) { Remove-Item release -Recurse -Force }
New-Item -ItemType Directory -Force -Path release | Out-Null

# 记录完整构建日志，便于 CI 排查
$buildLog = "release\\pyinstaller-windows.log"
$pyiArgs = @(
  "--noconfirm",
  "--clean",
  "--windowed",
  "--onedir",
  "--name", "InterviewAssistant",
  "app/main.py"
)

Write-Output "[build] Running: pyinstaller $($pyiArgs -join ' ')"
& pyinstaller @pyiArgs 2>&1 | Tee-Object -FilePath $buildLog
if ($LASTEXITCODE -ne 0) {
  Write-Error "PyInstaller failed with exit code $LASTEXITCODE. See $buildLog"
  exit $LASTEXITCODE
}

$zipPath = "release\\InterviewAssistant-windows.zip"
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }

$oneDir = "dist\\InterviewAssistant"
$oneFile = "dist\\InterviewAssistant.exe"

if (Test-Path $oneDir) {
  # 压缩目录本体，避免通配符在空目录/特殊场景下报“路径无效”
  Compress-Archive -Path $oneDir -DestinationPath $zipPath -Force
}
elseif (Test-Path $oneFile) {
  Compress-Archive -Path $oneFile -DestinationPath $zipPath -Force
}
else {
  # 兜底：扫描 dist 下任意可发布目标
  $candidates = @()
  if (Test-Path "dist") {
    $candidates += Get-ChildItem dist -Directory -ErrorAction SilentlyContinue
    $candidates += Get-ChildItem dist -File -Filter *.exe -ErrorAction SilentlyContinue
  }
  if ($candidates.Count -gt 0) {
    $target = $candidates[0].FullName
    Write-Output "[build] Fallback packaging target: $target"
    Compress-Archive -Path $target -DestinationPath $zipPath -Force
  }
  else {
    Write-Error "PyInstaller output not found. Expected '$oneDir' or '$oneFile'. See $buildLog"
  }
  if (Test-Path "dist") {
    Write-Output "dist content:"
    Get-ChildItem -Recurse dist | Select-Object FullName
  }
  if (!(Test-Path $zipPath)) { exit 1 }
}

Write-Output "[OK] Windows package created: release\InterviewAssistant-windows.zip"
