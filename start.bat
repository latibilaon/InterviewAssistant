@echo off
setlocal ENABLEDELAYEDEXPANSION
cd /d %~dp0

if not exist app\.env (
  copy app\.env.example app\.env >nul
  echo [init] Created app\.env. Fill API key in settings or file.
)

if not exist .venv (
  py -3 -m venv .venv
)

call .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt

set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set TOKENIZERS_PARALLELISM=false

cd app
python main.py --live %*
