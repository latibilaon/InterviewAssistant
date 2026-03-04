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
Compress-Archive -Path dist\InterviewAssistant\* -DestinationPath release\InterviewAssistant-windows.zip -Force
Write-Output "[OK] Windows package created: release\InterviewAssistant-windows.zip"
