#!/bin/bash

# HuggingDrive Virtual Environment Activation Script

echo "🚀 Activating HuggingDrive Virtual Environment..."

# Check if virtual environment exists
if [ ! -d "huggingdrive_env" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python3 -m venv huggingdrive_env
    source huggingdrive_env/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "✅ Virtual environment found. Activating..."
    source huggingdrive_env/bin/activate
fi

echo "✅ Virtual environment activated!"
echo "📦 Installed packages:"
pip list | grep -E "(PyQt6|transformers|torch|gradio|psutil)"

echo ""
echo "🎯 To run HuggingDrive:"
echo "   python3 huggingdrive.py"
echo ""
echo "🔧 To deactivate:"
echo "   deactivate"
echo ""
echo "📝 To install new packages:"
echo "   pip install package_name"
echo ""
echo "💾 To save current packages:"
echo "   pip freeze > requirements.txt" 