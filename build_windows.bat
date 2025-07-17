
@echo off
echo 🚀 HuggingDrive Windows Build Script
echo ======================================

echo 🔍 Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

echo ✅ Python found

echo 📦 Installing build dependencies...
pip install -r requirements_build.txt
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo ✅ Dependencies installed

echo 🏗️ Building Windows executable...
python build_executables.py
if %errorlevel% neq 0 (
    echo ❌ Build failed
    pause
    exit /b 1
)

echo.
echo 🎉 Build completed successfully!
echo 📁 Check the dist/ folder for HuggingDrive.exe
echo.
pause 