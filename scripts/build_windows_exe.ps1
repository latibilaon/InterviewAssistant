$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (!(Test-Path ".venv")) {
  py -3 -m venv .venv
}
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install pyinstaller
python -m pip install -r requirements.txt

if (Test-Path build) { Remove-Item build -Recurse -Force }
if (Test-Path dist) { Remove-Item dist -Recurse -Force }
if (Test-Path release) { Remove-Item release -Recurse -Force }

pyinstaller --noconfirm --clean --windowed --name InterviewAssistant app/main.py

New-Item -ItemType Directory -Force -Path release | Out-Null
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
  Write-Error "PyInstaller output not found. Expected '$oneDir' or '$oneFile'."
  if (Test-Path "dist") {
    Write-Output "dist content:"
    Get-ChildItem -Recurse dist | Select-Object FullName
  }
  exit 1
}

Write-Output "[OK] Windows package created: release\InterviewAssistant-windows.zip"
