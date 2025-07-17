#!/bin/bash

# HuggingDrive Installation Script
# This script installs HuggingDrive and its dependencies

set -e

echo "🚀 Installing HuggingDrive..."

# Check if Python 3.8+ is installed
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8 or higher is required. Found version: $python_version"
    exit 1
fi

echo "✅ Python version $python_version detected"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ Error: pip3 is not installed. Please install pip first."
    exit 1
fi

echo "📦 Installing dependencies..."

# Install dependencies
pip3 install -r requirements.txt

echo "✅ Dependencies installed successfully"

# Make scripts executable
chmod +x huggingdrive.py
chmod +x huggingdrive_cli.py

echo "✅ Scripts made executable"

# Create symlinks for easy access (optional)
if command -v ln &> /dev/null; then
    echo "🔗 Creating symlinks..."
    
    # Create symlink for CLI
    if [ -w /usr/local/bin ]; then
        sudo ln -sf "$(pwd)/huggingdrive_cli.py" /usr/local/bin/huggingdrive
        echo "✅ CLI symlink created: /usr/local/bin/huggingdrive"
    else
        echo "⚠️  Could not create CLI symlink (requires sudo access)"
    fi
    
    # Create symlink for GUI
    if [ -w /usr/local/bin ]; then
        sudo ln -sf "$(pwd)/huggingdrive.py" /usr/local/bin/huggingdrive-gui
        echo "✅ GUI symlink created: /usr/local/bin/huggingdrive-gui"
    else
        echo "⚠️  Could not create GUI symlink (requires sudo access)"
    fi
fi

echo ""
echo "🎉 HuggingDrive installation completed!"
echo ""
echo "Usage:"
echo "  GUI Version: python3 huggingdrive.py"
echo "  CLI Version: python3 huggingdrive_cli.py --help"
echo ""
echo "If symlinks were created successfully, you can also use:"
echo "  huggingdrive --help"
echo "  huggingdrive-gui"
echo ""
echo "For more information, see README.md" 