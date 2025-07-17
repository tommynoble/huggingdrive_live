#!/bin/bash

echo "🚀 HuggingDrive macOS Build Script"
echo "=================================="

# Check if Python is installed
echo "🔍 Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.9+"
    exit 1
fi

python3 --version
echo "✅ Python found"

# Check if pip is installed
echo "📦 Installing build dependencies..."
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 not found. Please install pip"
    exit 1
fi

pip3 install -r requirements_build.txt
if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✅ Dependencies installed"

# Build the executable
echo "🏗️ Building macOS application..."
python3 build_executables.py
if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    exit 1
fi

echo ""
echo "🎉 Build completed successfully!"
echo "📁 Check the dist/ folder for HuggingDrive"
echo "" 