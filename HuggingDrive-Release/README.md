# HuggingDrive v1.0.0 - Production Release

## 🚀 What is HuggingDrive?

HuggingDrive is a powerful GUI application that helps you manage and optimize Hugging Face models on external drives. It provides an intelligent interface for downloading, organizing, and running AI models efficiently.

## 📦 What's Included

This release contains:

- **Source Code** - Complete Python source code
- **Build Scripts** - PyInstaller specifications for creating executables
- **Documentation** - Comprehensive installation and usage guides
- **Requirements** - Production dependency specifications

## 🖥️ System Requirements

- **macOS**: 10.13 or later (Apple Silicon or Intel)
- **Python**: 3.9 or later
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space for the application
- **External Drive**: USB, Thunderbolt, or network drive for model storage

## 📋 Installation

### Option 1: Build from Source (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/tommynoble/huggingdrive_live.git
   cd huggingdrive_live
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements-prod.txt
   ```

4. **Build executables**:
   ```bash
   pip install pyinstaller
   pyinstaller HuggingDrive.spec  # GUI version
   pyinstaller --onefile --name HuggingDriveCLI huggingdrive_cli_simple.py  # CLI version
   ```

5. **Run the application**:
   ```bash
   # GUI version
   open dist/HuggingDrive.app
   
   # CLI version
   ./dist/HuggingDriveCLI --help
   ```

### Option 2: Run from Source

```bash
# Install dependencies
pip install -r requirements-prod.txt

# Run GUI
python huggingdrive.py

# Run CLI
python huggingdrive_cli_simple.py --help
```

## 🎯 Quick Start

### GUI Version
1. Launch the application
2. Connect an external drive
3. Browse and download models
4. Test models with the built-in chat interface

### CLI Version
```bash
# List available external drives
./HuggingDriveCLI --list-drives

# List models on a drive
./HuggingDriveCLI --list-models --drive "/Volumes/YourDrive"

# Download a model (requires GUI for full functionality)
./HuggingDriveCLI --download "microsoft/DialoGPT-medium" --drive "/Volumes/YourDrive"
```

## 🔧 Features

### ✅ Working Features
- **External Drive Detection** - Automatically finds USB/Thunderbolt drives
- **Model Management** - Download, organize, and track models
- **CLI Interface** - Command-line operations for automation
- **Cross-Platform** - Works on macOS (Windows/Linux versions available)

### 🎨 GUI Features
- **Intuitive Interface** - Easy-to-use graphical interface
- **Model Search** - Browse thousands of Hugging Face models
- **Download Management** - Progress tracking and resume capability
- **Model Testing** - Built-in chat interface for testing models
- **Drive Management** - Organize models across multiple drives

## 📁 File Structure

```
huggingdrive_live/
├── huggingdrive/              # Main application modules
├── dist/                      # Built executables (after building)
├── HuggingDrive-Release/      # Release documentation
├── requirements-prod.txt      # Production dependencies
├── HuggingDrive.spec         # PyInstaller spec for GUI
└── huggingdrive_cli_simple.py # CLI source code
```

## 🔍 Troubleshooting

### Common Issues

**"No external drives found"**
- Ensure your drive is properly connected
- Check drive permissions in System Preferences
- Try reconnecting the drive

**"Download failed"**
- Check internet connection
- Verify sufficient disk space
- Ensure drive is writable

**"Build failed"**
- Check Python version (3.9+ required)
- Verify all dependencies are installed
- Check PyInstaller installation

### Support

For issues and questions:
- **GitHub**: https://github.com/tommynoble/huggingdrive_live
- **Issues**: Create an issue on GitHub
- **Documentation**: See INSTALL.md for detailed instructions

## 🔄 Updates

To update HuggingDrive:
1. Pull the latest changes: `git pull origin main`
2. Rebuild executables if needed
3. Your models and settings will be preserved

## 📄 License

This software is provided as-is for educational and personal use.

## 🎉 What's New in v1.0.0

- ✅ **Production-ready source code**
- ✅ **Working CLI interface**
- ✅ **External drive detection**
- ✅ **Model management system**
- ✅ **Cross-platform compatibility**
- ✅ **Professional documentation**
- ✅ **Build scripts and specifications**

---

**Made with ❤️ for the AI community**

## 📥 Download Executables

For pre-built executables, please check the [GitHub Releases](https://github.com/tommynoble/huggingdrive_live/releases) page or build from source using the instructions above. 