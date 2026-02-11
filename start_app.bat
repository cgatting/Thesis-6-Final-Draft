@echo off
title RefScore Launcher
echo ==========================================
echo       RefScore Development Launcher
echo ==========================================

cd /d "%~dp0"

if not exist "node_modules" (
    echo [INFO] node_modules not found. Installing dependencies...
    call npm install
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b %errorlevel%
    )
)

echo [INFO] Starting development server...
echo [INFO] The application will be available at http://localhost:3000 (or similar)
echo.

call npm run dev

if %errorlevel% neq 0 (
    echo [ERROR] Server stopped unexpectedly.
    pause
)
