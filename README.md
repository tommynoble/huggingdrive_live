# HuggingDrive 🚀

A robust GUI application for managing Hugging Face models on external drives with intelligent quantization selection and system-optimized recommendations.

## ✨ Features

- **Smart Model Management** - Download, organize, and manage Hugging Face models
- **Intelligent Quantization Selection** - AI-powered recommendations based on your system specs
- **External Drive Support** - Store models on external drives to save local storage
- **Advanced UI/UX** - Professional interface with dark theme and smart tooltips
- **Multi-Format Support** - Handles regular models, GGUF files, and custom quantizations
- **System Integration** - Detects drives, manages storage, and optimizes for your hardware

## 🛠️ Installation

### Prerequisites
- Python 3.9 or higher
- macOS, Windows, or Linux
- External drive (optional but recommended)

### Quick Start

1. **Clone or download the repository**
   ```bash
   git clone <repository-url>
   cd huggingdrive
   ```

2. **Set up virtual environment (Recommended)**
   ```bash
   # Option 1: Use the activation script
   ./activate_env.sh
   
   # Option 2: Manual setup
   python3 -m venv huggingdrive_env
   source huggingdrive_env/bin/activate  # On Windows: huggingdrive_env\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run HuggingDrive**
   ```bash
   python3 huggingdrive.py
   ```

## 🎯 Usage

### Basic Workflow

1. **Connect External Drive** - The app will automatically detect external drives
2. **Search for Models** - Use the search bar to find models on Hugging Face
3. **Select Quantization** - For GGUF models, choose the best quantization for your system
4. **Download Models** - Models will be downloaded to your external drive
5. **Test Models** - Use the built-in Gradio interface to test your models

### Smart Quantization Selection

The app analyzes your system specs and provides personalized recommendations:

- **32GB+ RAM**: Q4_K_M or Q5_K_M for high-quality models
- **16GB RAM**: Q4_K_M for optimal balance
- **8GB RAM**: Q4_K_S or Q4_0 for stability
- **<8GB RAM**: Q4_K_S or Q3_K_M for limited systems

### Features

- **🔍 Smart Search** - Search Hugging Face models with filters
- **💾 External Storage** - Store models on external drives
- **⚡ Quick Testing** - Built-in Gradio interface for model testing
- **📊 System Monitoring** - Real-time memory and storage monitoring
- **🔄 Model Management** - Organize, delete, and convert models
- **🔐 Authentication** - Hugging Face token support for gated models

## 📁 Project Structure

```
huggingdrive/
├── huggingdrive/
│   ├── gui.py              # Main GUI application
│   ├── downloader.py       # Model downloader
│   ├── gradio_manager.py   # Gradio interface manager
│   ├── auth_manager.py     # Hugging Face authentication
│   └── __init__.py
├── huggingdrive.py         # Main entry point
├── requirements.txt        # Python dependencies
├── activate_env.sh         # Virtual environment setup script
└── README.md              # This file
```

## 🔧 Configuration

### Environment Variables
- `HF_TOKEN` - Hugging Face authentication token (optional)
- `CACHE_DIR` - Custom cache directory (optional)

### Settings
- **Default Download Location**: External drive (if available)
- **Cache Directory**: `~/.cache/huggingface/`
- **Model Storage**: External drive in `huggingface_models/` folder

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure you're using the virtual environment
   - Run `pip install -r requirements.txt`

2. **Permission Errors**
   - Check external drive permissions
   - Ensure write access to the drive

3. **Memory Issues**
   - Close other applications
   - Use smaller model quantizations
   - Consider upgrading RAM

4. **Download Failures**
   - Check internet connection
   - Verify Hugging Face token (for gated models)
   - Ensure sufficient disk space

### Getting Help

- Check the status bar for error messages
- Review the console output for detailed error information
- Ensure all dependencies are properly installed

## 🚀 Advanced Features

### Model Conversion
- Convert models to GGUF format for better performance
- Support for various quantization levels
- Batch conversion capabilities

### API Integration
- Hugging Face Hub integration
- Model metadata fetching
- File size and compatibility checking

### Performance Optimization
- Multi-threaded downloads
- Memory-efficient model loading
- Smart caching strategies

## 📝 Development

### Setting Up Development Environment

1. **Clone the repository**
2. **Create virtual environment**
   ```bash
   python3 -m venv huggingdrive_env
   source huggingdrive_env/bin/activate
   pip install -r requirements.txt
   ```

3. **Install development dependencies**
   ```bash
   pip install pytest black flake8
   ```

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to functions and classes

### Testing
```bash
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review the console output for error details
- Ensure you're using the latest version

---

**HuggingDrive** - Making AI model management simple and efficient! 🎉
