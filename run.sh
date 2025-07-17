#!/bin/bash
# HuggingDrive Launcher Script

echo "🚀 Starting HuggingDrive..."

# Check if python3 is available
if command -v python3 &> /dev/null; then
    echo "✅ Python 3 found: $(python3 --version)"
    
    # Check if required packages are installed
    if python3 -c "import PyQt6" 2>/dev/null; then
        echo "✅ PyQt6 is installed"
        echo "🎯 Launching HuggingDrive GUI..."
        python3 huggingdrive_cli.py
    else
        echo "❌ PyQt6 is not installed"
        echo "📦 Installing dependencies..."
        pip3 install -r requirements.txt
        echo "🎯 Launching HuggingDrive GUI..."
        python3 huggingdrive_cli.py
    fi
else
    echo "❌ Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi 