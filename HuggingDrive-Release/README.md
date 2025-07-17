# HuggingDrive v1.0.0 - Production Release

## 🚀 What is HuggingDrive?

HuggingDrive is a powerful GUI application that helps you manage and optimize Hugging Face models on external drives. It provides an intelligent interface for downloading, organizing, and running AI models efficiently.

## 📦 What's Included

This release contains:

- **HuggingDrive.app** - macOS GUI application (156MB)
- **HuggingDriveCLISimple** - Command-line interface (142MB)
- **requirements-prod.txt** - Production dependencies
- **README.md** - This documentation

## 🖥️ System Requirements

- **macOS**: 10.13 or later (Apple Silicon or Intel)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space for the application
- **External Drive**: USB, Thunderbolt, or network drive for model storage

## 📋 Installation

### Option 1: GUI Application (Recommended)

1. **Download** the `HuggingDrive.app` file
2. **Drag and drop** it to your Applications folder
3. **Launch** from Applications or Spotlight
4. **First run**: The app will request permissions for external drive access

### Option 2: Command Line Interface

1. **Download** the `HuggingDriveCLISimple` file
2. **Make executable**: `chmod +x HuggingDriveCLISimple`
3. **Run**: `./HuggingDriveCLISimple --help`

## 🎯 Quick Start

### GUI Version
1. Launch `HuggingDrive.app`
2. Connect an external drive
3. Browse and download models
4. Test models with the built-in chat interface

### CLI Version
```bash
# List available external drives
./HuggingDriveCLISimple --list-drives

# List models on a drive
./HuggingDriveCLISimple --list-models --drive "/Volumes/YourDrive"

# Download a model (requires GUI for full functionality)
./HuggingDriveCLISimple --download "microsoft/DialoGPT-medium" --drive "/Volumes/YourDrive"
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
HuggingDrive-Release/
├── HuggingDrive.app/          # macOS GUI application
├── HuggingDriveCLISimple      # Command-line interface
├── requirements-prod.txt      # Production dependencies
└── README.md                  # This documentation
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

**"App won't launch"**
- Check macOS security settings
- Right-click and select "Open" for first launch
- Verify macOS version compatibility

### Support

For issues and questions:
- **GitHub**: https://github.com/tommynoble/huggingdrive_live
- **Issues**: Create an issue on GitHub
- **Documentation**: See the main repository README

## 🔄 Updates

To update HuggingDrive:
1. Download the latest release
2. Replace the existing application
3. Your models and settings will be preserved

## 📄 License

This software is provided as-is for educational and personal use.

## 🎉 What's New in v1.0.0

- ✅ **Production-ready executables**
- ✅ **Working CLI interface**
- ✅ **External drive detection**
- ✅ **Model management system**
- ✅ **Cross-platform compatibility**
- ✅ **Professional packaging**

---

**Made with ❤️ for the AI community** 