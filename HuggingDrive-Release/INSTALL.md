# HuggingDrive Installation Guide

## 🎯 Choose Your Installation Method

### For End Users (Recommended)

#### macOS Users
1. **Download** the release package
2. **Extract** the ZIP file
3. **Drag** `HuggingDrive.app` to your Applications folder
4. **Launch** from Applications or use Spotlight (Cmd+Space)
5. **First Launch**: Right-click and select "Open" if macOS blocks it

#### Command Line Users
1. **Download** the release package
2. **Extract** the ZIP file
3. **Open Terminal** and navigate to the extracted folder
4. **Make executable**: `chmod +x HuggingDriveCLISimple`
5. **Test**: `./HuggingDriveCLISimple --help`

### For Developers

#### From Source
```bash
# Clone the repository
git clone https://github.com/tommynoble/huggingdrive_live.git
cd huggingdrive_live

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements-prod.txt

# Run from source
python huggingdrive.py
```

#### Building Executables
```bash
# Install PyInstaller
pip install pyinstaller

# Build GUI version
pyinstaller HuggingDrive.spec

# Build CLI version
pyinstaller --onefile --name HuggingDriveCLI huggingdrive_cli_simple.py
```

## 🔧 System Requirements

### Minimum Requirements
- **OS**: macOS 10.13+ (High Sierra)
- **Architecture**: Intel or Apple Silicon
- **RAM**: 4GB
- **Storage**: 2GB free space
- **Network**: Internet connection for downloads

### Recommended Requirements
- **OS**: macOS 11+ (Big Sur) or later
- **RAM**: 8GB or more
- **Storage**: 10GB+ free space
- **External Drive**: USB 3.0, Thunderbolt, or SSD
- **Network**: High-speed internet for large model downloads

## 📋 Pre-Installation Checklist

### ✅ System Preparation
- [ ] macOS 10.13 or later
- [ ] Sufficient disk space (2GB+)
- [ ] External drive connected (optional but recommended)
- [ ] Internet connection available
- [ ] Admin privileges (for first installation)

### ✅ External Drive Setup
- [ ] Drive is properly connected and mounted
- [ ] Drive has sufficient free space (models can be 1-50GB+)
- [ ] Drive is writable (not read-only)
- [ ] Drive is formatted with a compatible filesystem (APFS, HFS+, exFAT)

## 🚀 Post-Installation Setup

### First Launch
1. **Launch** HuggingDrive.app
2. **Grant permissions** when prompted:
   - Full Disk Access (for external drives)
   - Network Access (for downloads)
3. **Connect external drive** if using one
4. **Test drive detection** in the app

### Configuration
1. **Set default drive** in preferences
2. **Configure download settings**:
   - Download location
   - Concurrent downloads
   - Cache settings
3. **Test with a small model** first

## 🔍 Verification

### GUI Version
1. Launch the app
2. Check that external drives are detected
3. Try searching for a model
4. Test the interface responsiveness

### CLI Version
```bash
# Test help
./HuggingDriveCLISimple --help

# Test drive detection
./HuggingDriveCLISimple --list-drives

# Test model listing (if you have models)
./HuggingDriveCLISimple --list-models --drive "/Volumes/YourDrive"
```

## 🛠️ Troubleshooting

### Installation Issues

**"App is damaged"**
```bash
# Remove quarantine attribute
xattr -d com.apple.quarantine HuggingDrive.app
```

**"Cannot be opened"**
- Right-click → Open
- Go to System Preferences → Security & Privacy → Allow

**"Permission denied"**
```bash
# Make executable
chmod +x HuggingDriveCLISimple
```

### Runtime Issues

**"No external drives found"**
- Check drive connection
- Verify drive permissions
- Try reconnecting the drive

**"Download failed"**
- Check internet connection
- Verify disk space
- Check drive permissions

**"App crashes on launch"**
- Check macOS version compatibility
- Verify system requirements
- Try reinstalling

## 📞 Support

### Getting Help
1. **Check this documentation** first
2. **Search existing issues** on GitHub
3. **Create a new issue** with:
   - macOS version
   - Error message
   - Steps to reproduce
   - System specifications

### Useful Commands
```bash
# Check macOS version
sw_vers

# Check available disk space
df -h

# Check external drives
ls /Volumes/

# Check app permissions
ls -la HuggingDrive.app
```

---

**Need more help?** Visit our [GitHub repository](https://github.com/tommynoble/huggingdrive_live) for additional resources and community support. 