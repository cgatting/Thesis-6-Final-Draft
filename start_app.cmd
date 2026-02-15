@echo off
setlocal
title RefScore Launcher
echo ==========================================
echo       RefScore Development Launcher
echo ==========================================

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
  echo [INFO] Creating Python virtual environment...
  py -3 -m venv .venv 2>nul || python -m venv .venv
)

if not exist ".venv\Scripts\activate.bat" (
  echo [ERROR] Failed to create Python venv.
  pause
  exit /b 1
)

call ".venv\Scripts\activate"

python -c "import importlib.util as u, sys; mods=['fastapi','uvicorn','pydantic','starlette','customtkinter','nltk','yake','sentence_transformers','transformers','sklearn','aiohttp','torch']; sys.exit(0 if all(u.find_spec(m) for m in mods) else 1)"
if errorlevel 1 (
  echo [INFO] Installing Python dependencies...
  python -m pip install --upgrade pip wheel
  pip install fastapi "uvicorn[standard]" pydantic starlette customtkinter nltk yake sentence-transformers transformers scikit-learn aiohttp
  pip install --index-url https://download.pytorch.org/whl/cpu torch
  python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"
)

if exist "deepsearch_api.py" (
  echo [INFO] Starting DeepSearch API at http://127.0.0.1:8000
  start "DeepSearch API" cmd /k ".venv\Scripts\activate && uvicorn deepsearch_api:app --host 127.0.0.1 --port 8000 --reload"
)

if not exist "node_modules" (
  echo [INFO] Installing Node dependencies...
  call npm install
  if %errorlevel% neq 0 (
    echo [ERROR] Failed to install Node dependencies.
    pause
    exit /b %errorlevel%
  )
)

echo [INFO] Starting development server...
echo [INFO] The application will be available at http://localhost:3000
if exist "deepsearch_api.py" echo [INFO] DeepSearch API available at http://127.0.0.1:8000
echo.

call npm run dev

if %errorlevel% neq 0 (
  echo [ERROR] Server stopped unexpectedly.
  pause
)
