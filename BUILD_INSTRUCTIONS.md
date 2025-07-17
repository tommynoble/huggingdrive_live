# HuggingDrive Build Instructions

This guide will help you create standalone executables for HuggingDrive on Windows and macOS.

## Prerequisites

### For Windows:
- Python 3.9 or higher
- pip (usually comes with Python)
- At least 2GB free disk space

### For macOS:
- Python 3.9 or higher
- pip3 (usually comes with Python)
- At least 2GB free disk space

## Quick Build

### Windows
1. Open Command Prompt in the HuggingDrive directory
2. Run: `build_windows.bat`
3. Wait for the build to complete
4. Find your executable in `dist/HuggingDrive.exe`

### macOS
1. Open Terminal in the HuggingDrive directory
2. Run: `./build_macos.sh`
3. Wait for the build to complete
4. Find your application in `dist/HuggingDrive`

## Manual Build

If the quick build scripts don't work, you can build manually:

### 1. Install Dependencies
```bash
# Windows
pip install -r requirements_build.txt

# macOS
pip3 install -r requirements_build.txt
```

### 2. Run Build Script
```bash
# Windows
python build_executables.py

# macOS
python3 build_executables.py
```

## Build Output

After a successful build, you'll find:

### Windows
- `dist/HuggingDrive.exe` - Standalone executable
- Size: ~100-200MB (includes Python runtime and all dependencies)

### macOS
- `dist/HuggingDrive` - Standalone application
- Size: ~100-200MB (includes Python runtime and all dependencies)

## Testing Your Build

1. **Test the executable**:
   - Windows: Double-click `HuggingDrive.exe`
   - macOS: Double-click `HuggingDrive` or run `./dist/HuggingDrive`

2. **Check functionality**:
   - Verify the GUI opens
   - Test external drive detection
   - Try downloading a small model

## Troubleshooting

### Common Issues

**"Python not found"**
- Install Python 3.9+ from python.org
- Make sure Python is in your PATH

**"pip not found"**
- Python usually comes with pip
- Try: `python -m pip install -r requirements_build.txt`

**"PyInstaller not found"**
- Install manually: `pip install pyinstaller`

**Build fails with import errors**
- The build script includes common hidden imports
- If you get specific import errors, add them to the `--hidden-import` list in `build_executables.py`

**Large file size**
- This is normal! The executable includes Python runtime and all dependencies
- Consider using `--onedir` instead of `--onefile` for smaller size (but requires distributing the entire folder)

### Windows Specific

**"Access denied" errors**
- Run Command Prompt as Administrator
- Make sure antivirus isn't blocking the build

**Missing Visual C++ Redistributable**
- Install Microsoft Visual C++ Redistributable

### macOS Specific

**"Permission denied"**
- Make sure the script is executable: `chmod +x build_macos.sh`
- Run with sudo if needed: `sudo ./build_macos.sh`

**"Gatekeeper" warnings**
- Right-click the app and select "Open"
- Or: `xattr -cr dist/HuggingDrive`

## Distribution

### Windows
- Distribute `HuggingDrive.exe` directly
- Consider creating an installer with NSIS or Inno Setup
- Package with required Visual C++ Redistributable

### macOS
- Distribute the `HuggingDrive` application
- Consider creating a `.dmg` file for easier distribution
- Code sign the application for better user experience

## Advanced Options

### Custom Icons
Place your icon files in the `assets/` directory:
- Windows: `assets/icon.ico`
- macOS: `assets/icon.icns`

### Smaller Build Size
Edit `build_executables.py` and change:
```python
"--onefile",  # Remove this line
"--onedir",   # Add this line instead
```

### Debug Build
Remove `--windowed` from the PyInstaller command to see console output.

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are installed
3. Try building manually step by step
4. Check PyInstaller documentation for advanced options 