#!/usr/bin/env python3
"""
Build script for creating HuggingDrive executables
Supports Windows (.exe) and macOS (.app)
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    # Check PyInstaller
    try:
        import PyInstaller
        print("✅ PyInstaller found")
    except ImportError:
        print("❌ PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # Check if we're on the right platform
    system = platform.system()
    print(f"✅ Building for {system}")

def clean_build_dirs():
    """Clean previous build directories"""
    print("🧹 Cleaning previous builds...")
    
    dirs_to_clean = ["build", "dist", "__pycache__"]
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"   Removed {dir_name}/")

def build_windows_exe():
    """Build Windows executable"""
    print("🪟 Building Windows executable...")
    
    # PyInstaller command for Windows
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--windowed",
        "--name=HuggingDrive",
        "--icon=assets/icon.ico" if os.path.exists("assets/icon.ico") else "",
        "--add-data=huggingdrive;huggingdrive",
        "--hidden-import=PyQt6",
        "--hidden-import=huggingface_hub",
        "--hidden-import=transformers",
        "--hidden-import=torch",
        "--hidden-import=psutil",
        "--hidden-import=requests",
        "--hidden-import=json",
        "--hidden-import=pathlib",
        "--hidden-import=platform",
        "--hidden-import=subprocess",
        "--hidden-import=shutil",
        "--hidden-import=tempfile",
        "--hidden-import=random",
        "--hidden-import=socket",
        "--hidden-import=sys",
        "--hidden-import=os",
        "--hidden-import=time",
        "--hidden-import=threading",
        "huggingdrive_cli.py"
    ]
    
    # Remove empty strings
    cmd = [arg for arg in cmd if arg]
    
    try:
        subprocess.check_call(cmd)
        print("✅ Windows executable built successfully!")
        print("   Location: dist/HuggingDrive.exe")
    except subprocess.CalledProcessError as e:
        print(f"❌ Windows build failed: {e}")
        return False
    
    return True

def build_macos_app():
    """Build macOS application bundle"""
    print("🍎 Building macOS application...")
    
    # PyInstaller command for macOS
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onedir",
        "--windowed",
        "--name=HuggingDrive",
        "--icon=assets/logo.icns" if os.path.exists("assets/logo.icns") else "",
        "--add-data=huggingdrive:huggingdrive",
        "--hidden-import=PyQt6",
        "--hidden-import=huggingface_hub",
        "--hidden-import=transformers",
        "--hidden-import=torch",
        "--hidden-import=psutil",
        "--hidden-import=requests",
        "--hidden-import=json",
        "--hidden-import=pathlib",
        "--hidden-import=platform",
        "--hidden-import=subprocess",
        "--hidden-import=shutil",
        "--hidden-import=tempfile",
        "--hidden-import=random",
        "--hidden-import=socket",
        "--hidden-import=sys",
        "--hidden-import=os",
        "--hidden-import=time",
        "--hidden-import=threading",
        "huggingdrive_cli.py"
    ]
    
    # Remove empty strings
    cmd = [arg for arg in cmd if arg]
    
    try:
        subprocess.check_call(cmd)
        print("✅ macOS application built successfully!")
        print("   Location: dist/HuggingDrive")
    except subprocess.CalledProcessError as e:
        print(f"❌ macOS build failed: {e}")
        return False
    
    return True

def create_assets_directory():
    """Create assets directory and placeholder icons"""
    print("📁 Creating assets directory...")
    
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    
    # Create placeholder icon files if they don't exist
    if not (assets_dir / "icon.ico").exists():
        print("   Note: assets/icon.ico not found - using default icon")
    
    if not (assets_dir / "icon.icns").exists():
        print("   Note: assets/icon.icns not found - using default icon")

def main():
    """Main build function"""
    print("🚀 HuggingDrive Build Script")
    print("=" * 40)
    
    # Check dependencies
    check_dependencies()
    
    # Create assets directory
    create_assets_directory()
    
    # Clean previous builds
    clean_build_dirs()
    
    # Build based on platform
    system = platform.system()
    
    if system == "Windows":
        success = build_windows_exe()
        if success:
            print("\n🎉 Windows build completed!")
            print("   Executable: dist/HuggingDrive.exe")
            print("   Size: Check the dist/ folder")
    elif system == "Darwin":  # macOS
        success = build_macos_app()
        if success:
            print("\n🎉 macOS build completed!")
            print("   Application: dist/HuggingDrive")
            print("   Size: Check the dist/ folder")
    else:
        print(f"❌ Unsupported platform: {system}")
        print("   This script supports Windows and macOS only")
        return
    
    print("\n📦 Build Summary:")
    print("   - Executable created in dist/ folder")
    print("   - Test the executable to ensure it works")
    print("   - Consider code signing for distribution")
    print("   - Package with installer for easier distribution")

if __name__ == "__main__":
    main() 