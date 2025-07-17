#!/usr/bin/env python3
"""
HuggingDrive - A GUI application for managing Hugging Face models on external drives
"""

import sys
import os
import json
import shutil
import threading
from pathlib import Path
from typing import Dict, List, Optional

import psutil
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QTextEdit, QComboBox, QProgressBar, QFileDialog,
                             QMessageBox, QListWidget, QListWidgetItem, QGroupBox, QCompleter,
                             QDialog, QTabWidget, QTextBrowser, QSplitter)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont, QIcon

from huggingface_hub import snapshot_download, HfApi
import torch
import platform

# Import Gradio manager
try:
    from huggingdrive.gradio_manager import GradioInterfaceManager
    GRADIO_AVAILABLE = True
    print("DEBUG: Gradio available")
except ImportError as e:
    GRADIO_AVAILABLE = False
    print(f"DEBUG: Gradio import failed: {e}")

# Force button to show for testing
GRADIO_AVAILABLE = True


class ModelLoadThread(QThread):
    """Thread for loading models without blocking the GUI"""
    model_loaded_signal = pyqtSignal(object, str)  # pipeline, model_type
    loading_progress_signal = pyqtSignal(str)
    progress_bar_signal = pyqtSignal(int)  # Progress percentage
    memory_check_signal = pyqtSignal(bool, str)  # is_sufficient, message
    error_signal = pyqtSignal(str)
    
    def __init__(self, model_path: str, model_name: str):
        super().__init__()
        self.model_path = model_path
        self.model_name = model_name
        self.should_abort = False
        self.timeout_seconds = 300  # 5 minutes timeout
        
    def abort_loading(self):
        """Allow user to abort the loading process"""
        self.should_abort = True
        
    def check_memory_requirements(self, model_type: str, model_size_gb: float = None) -> tuple[bool, str]:
        """Check if system has sufficient memory for the model"""
        try:
            # Get system memory info
            memory = psutil.virtual_memory()
            available_ram_gb = memory.available / (1024**3)
            total_ram_gb = memory.total / (1024**3)
            
            # Check VRAM if CUDA is available
            vram_available = False
            vram_gb = 0
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                vram_available = True
            
            # Memory requirements based on model type
            min_ram_gb = 4.0  # Base requirement
            min_vram_gb = 2.0  # Base VRAM requirement
            
            if model_type in ["gpt2", "gpt", "causal_lm", "llama", "mistral", "falcon", "bloom", "opt"]:
                min_ram_gb = 8.0
                min_vram_gb = 4.0
            elif model_type in ["bert", "roberta", "distilbert"]:
                min_ram_gb = 6.0
                min_vram_gb = 2.0
            elif "translation" in model_type or "marian" in model_type:
                min_ram_gb = 6.0
                min_vram_gb = 3.0
            elif model_type in ["t5", "bart"]:
                min_ram_gb = 8.0
                min_vram_gb = 4.0
            elif "vision" in model_type or "image" in model_type:
                min_ram_gb = 12.0
                min_vram_gb = 6.0
            
            # Adjust based on estimated model size
            if model_size_gb:
                min_ram_gb = max(min_ram_gb, model_size_gb * 2)  # Need 2x model size in RAM
                min_vram_gb = max(min_vram_gb, model_size_gb * 1.5)  # Need 1.5x model size in VRAM
            
            # Check RAM
            ram_sufficient = available_ram_gb >= min_ram_gb
            ram_message = f"RAM: {available_ram_gb:.1f}GB available, {min_ram_gb:.1f}GB required"
            
            # Check VRAM
            vram_sufficient = True
            vram_message = ""
            if vram_available:
                vram_sufficient = vram_gb >= min_vram_gb
                vram_message = f"VRAM: {vram_gb:.1f}GB available, {min_vram_gb:.1f}GB required"
            else:
                vram_message = "VRAM: CUDA not available, will use CPU"
            
            # Overall assessment
            overall_sufficient = ram_sufficient and (not vram_available or vram_sufficient)
            message = f"{ram_message}\n{vram_message}"
            
            if not overall_sufficient:
                message += f"\n⚠️ WARNING: Insufficient memory for {model_type} model"
                if not ram_sufficient:
                    message += f"\n- Consider closing other applications to free RAM"
                if vram_available and not vram_sufficient:
                    message += f"\n- This model may run slowly on CPU only"
            
            return overall_sufficient, message
            
        except Exception as e:
            return False, f"Error checking memory: {str(e)}"
    
    def estimate_model_size(self, model_path: str) -> float:
        """Estimate model size in GB"""
        try:
            total_size = 0
            model_dir = Path(model_path)
            
            if model_dir.exists():
                for file_path in model_dir.rglob("*"):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
                
                return total_size / (1024**3)  # Convert to GB
            return 0.0
        except:
            return 0.0
    
    def run(self):
        try:
            import time
            start_time = time.time()
            
            self.progress_bar_signal.emit(5)
            self.loading_progress_signal.emit("Analyzing model configuration...")
            
            if self.should_abort:
                return
            
            # Check for common model files to determine type
            model_path = Path(self.model_path)
            
            if (model_path / "config.json").exists():
                import json
                with open(model_path / "config.json", 'r') as f:
                    config = json.load(f)
                
                model_type = config.get("model_type", "unknown")
                self.loading_progress_signal.emit(f"Model type detected: {model_type}")
                self.progress_bar_signal.emit(15)
                
                if self.should_abort:
                    return
                
                # Check timeout
                if time.time() - start_time > self.timeout_seconds:
                    self.error_signal.emit(f"Model loading timed out after {self.timeout_seconds} seconds")
                    return
                
                # Estimate model size
                model_size_gb = self.estimate_model_size(self.model_path)
                if model_size_gb > 0:
                    self.loading_progress_signal.emit(f"Model size: {model_size_gb:.2f}GB")
                
                # Check memory requirements
                self.progress_bar_signal.emit(25)
                self.loading_progress_signal.emit("Checking memory requirements...")
                memory_sufficient, memory_message = self.check_memory_requirements(model_type, model_size_gb)
                self.memory_check_signal.emit(memory_sufficient, memory_message)
                
                if self.should_abort:
                    return
                
                # Check timeout
                if time.time() - start_time > self.timeout_seconds:
                    self.error_signal.emit(f"Model loading timed out after {self.timeout_seconds} seconds")
                    return
                
                # Continue loading even if memory is insufficient (user can decide)
                self.progress_bar_signal.emit(35)
                self.loading_progress_signal.emit("Loading model into memory...")
                
                # Load appropriate pipeline based on model type
                if model_type in ["gpt2", "gpt", "causal_lm", "llama", "mistral", "falcon", "bloom", "opt"]:
                    self.loading_progress_signal.emit("Loading text generation pipeline...")
                    self.progress_bar_signal.emit(50)
                    
                    if self.should_abort:
                        return
                    
                    # Check timeout
                    if time.time() - start_time > self.timeout_seconds:
                        self.error_signal.emit(f"Model loading timed out after {self.timeout_seconds} seconds")
                        return
                    
                    from transformers import pipeline
                    pipeline_obj = pipeline("text-generation", model=str(model_path))
                    self.progress_bar_signal.emit(90)
                    self.model_loaded_signal.emit(pipeline_obj, model_type)
                    
                elif model_type in ["bert", "roberta", "distilbert"]:
                    self.loading_progress_signal.emit("Loading text classification pipeline...")
                    self.progress_bar_signal.emit(50)
                    
                    if self.should_abort:
                        return
                    
                    # Check timeout
                    if time.time() - start_time > self.timeout_seconds:
                        self.error_signal.emit(f"Model loading timed out after {self.timeout_seconds} seconds")
                        return
                    
                    from transformers import pipeline
                    pipeline_obj = pipeline("text-classification", model=str(model_path))
                    self.progress_bar_signal.emit(90)
                    self.model_loaded_signal.emit(pipeline_obj, model_type)
                    
                elif "translation" in model_type or "marian" in model_type:
                    self.loading_progress_signal.emit("Loading translation pipeline...")
                    self.progress_bar_signal.emit(50)
                    
                    if self.should_abort:
                        return
                    
                    # Check timeout
                    if time.time() - start_time > self.timeout_seconds:
                        self.error_signal.emit(f"Model loading timed out after {self.timeout_seconds} seconds")
                        return
                    
                    from transformers import pipeline
                    pipeline_obj = pipeline("translation", model=str(model_path))
                    self.progress_bar_signal.emit(90)
                    self.model_loaded_signal.emit(pipeline_obj, model_type)
                    
                elif model_type in ["t5", "bart"]:
                    self.loading_progress_signal.emit("Loading text-to-text generation pipeline...")
                    self.progress_bar_signal.emit(50)
                    
                    if self.should_abort:
                        return
                    
                    # Check timeout
                    if time.time() - start_time > self.timeout_seconds:
                        self.error_signal.emit(f"Model loading timed out after {self.timeout_seconds} seconds")
                        return
                    
                    from transformers import pipeline
                    pipeline_obj = pipeline("text2text-generation", model=str(model_path))
                    self.progress_bar_signal.emit(90)
                    self.model_loaded_signal.emit(pipeline_obj, model_type)
                    
                elif "vision" in model_type or "image" in model_type:
                    self.loading_progress_signal.emit("Loading vision pipeline...")
                    self.progress_bar_signal.emit(50)
                    
                    if self.should_abort:
                        return
                    
                    # Check timeout
                    if time.time() - start_time > self.timeout_seconds:
                        self.error_signal.emit(f"Model loading timed out after {self.timeout_seconds} seconds")
                        return
                    
                    from transformers import pipeline
                    pipeline_obj = pipeline("image-classification", model=str(model_path))
                    self.progress_bar_signal.emit(90)
                    self.model_loaded_signal.emit(pipeline_obj, model_type)
                    
                else:
                    self.error_signal.emit(f"Model type '{model_type}' not supported for testing")
                    
            else:
                self.error_signal.emit("Could not load model configuration")
                
        except Exception as e:
            self.error_signal.emit(f"Error loading model: {str(e)}")
        finally:
            self.progress_bar_signal.emit(100)


class ModelTestDialog(QDialog):
    """Dialog for testing/interacting with downloaded models"""
    
    def __init__(self, model_name: str, model_path: str, parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.model_path = model_path
        self.pipeline = None
        self.model_load_thread = None
        
        # Memory management
        self.model_loaded = False
        self.memory_warning_shown = False
        
        # Gradio interface manager
        self.gradio_manager = None
        if GRADIO_AVAILABLE:
            self.gradio_manager = GradioInterfaceManager()
        
        # Set up cache directories in the local user directory
        home_dir = Path.home()
        self.cache_dir = home_dir / ".huggingdrive_cache"
        self.chat_history_file = self.cache_dir / f"chat_history_{model_name.replace('/', '_')}.txt"
        self.model_cache_dir = self.cache_dir / "model_cache"
        self.datasets_cache_dir = self.cache_dir / "datasets"
        
        # Create cache directories
        self.cache_dir.mkdir(exist_ok=True)
        self.model_cache_dir.mkdir(exist_ok=True)
        self.datasets_cache_dir.mkdir(exist_ok=True)
        
        # Set environment variables for transformers cache
        import os
        os.environ['TRANSFORMERS_CACHE'] = str(self.model_cache_dir)
        os.environ['HF_HOME'] = str(self.cache_dir)
        os.environ['HF_DATASETS_CACHE'] = str(self.datasets_cache_dir)
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle(f"Test Model: {self.model_name}")
        self.setGeometry(200, 200, 800, 600)
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Model info
        info_label = QLabel(f"Model: {self.model_name}")
        info_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(info_label)
        
        path_label = QLabel(f"Path: {self.model_path}")
        path_label.setFont(QFont("Arial", 10))
        layout.addWidget(path_label)
        
        # Create a horizontal layout for tabs and web interface button
        tabs_and_button_layout = QHBoxLayout()
        
        # Tab widget for different test types
        self.tab_widget = QTabWidget()
        tabs_and_button_layout.addWidget(self.tab_widget)
        
        # Web interface button will be created in the web interface tab
        self.web_interface_btn = None
        
        layout.addLayout(tabs_and_button_layout)
        
        # Text Generation Tab
        self.create_text_generation_tab()
        
        # Text Classification Tab
        self.create_text_classification_tab()
        
        # Web Interface Tab (replacing Translation)
        self.create_web_interface_tab()
        
        # Chat Tab
        self.create_chat_tab()
        
        # Model Info Tab
        self.create_model_info_tab()
        
        # Load/Unload buttons
        load_buttons_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("Load Model")
        self.load_btn.clicked.connect(self.load_model)
        load_buttons_layout.addWidget(self.load_btn)
        
        self.abort_btn = QPushButton("Abort Loading")
        self.abort_btn.clicked.connect(self.abort_loading)
        self.abort_btn.setEnabled(False)
        load_buttons_layout.addWidget(self.abort_btn)
        
        self.unload_btn = QPushButton("Unload Model")
        self.unload_btn.clicked.connect(self.unload_model)
        self.unload_btn.setEnabled(False)
        load_buttons_layout.addWidget(self.unload_btn)
        
        layout.addLayout(load_buttons_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Memory info
        self.memory_info_label = QLabel("")
        self.memory_info_label.setStyleSheet("color: blue; font-size: 10px;")
        layout.addWidget(self.memory_info_label)
        
        # Status
        self.status_label = QLabel("Model not loaded")
        layout.addWidget(self.status_label)
        
        # Gradio status indicator
        self.gradio_status_label = QLabel("")
        self.gradio_status_label.setStyleSheet("color: blue; font-size: 10px;")
        layout.addWidget(self.gradio_status_label)
    
    def create_text_generation_tab(self):
        """Create text generation testing tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Input
        layout.addWidget(QLabel("Input Text:"))
        self.text_input = QTextEdit()
        self.text_input.setMaximumHeight(100)
        self.text_input.setPlaceholderText("Enter text to continue...")
        layout.addWidget(self.text_input)
        
        # Parameters
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("Max Length:"))
        self.max_length_spin = QComboBox()
        self.max_length_spin.addItems(["50", "100", "200", "500"])
        self.max_length_spin.setCurrentText("100")
        params_layout.addWidget(self.max_length_spin)
        
        params_layout.addWidget(QLabel("Temperature:"))
        self.temperature_spin = QComboBox()
        self.temperature_spin.addItems(["0.1", "0.5", "0.7", "1.0", "1.2"])
        self.temperature_spin.setCurrentText("0.7")
        params_layout.addWidget(self.temperature_spin)
        
        layout.addLayout(params_layout)
        
        # Generate button
        self.generate_btn = QPushButton("Generate Text")
        self.generate_btn.clicked.connect(self.generate_text)
        self.generate_btn.setEnabled(False)
        layout.addWidget(self.generate_btn)
        
        # Output
        layout.addWidget(QLabel("Generated Text:"))
        self.text_output = QTextBrowser()
        layout.addWidget(self.text_output)
        
        self.tab_widget.addTab(tab, "Text Generation")
    
    def create_text_classification_tab(self):
        """Create text classification testing tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Input
        layout.addWidget(QLabel("Text to Classify:"))
        self.classify_input = QTextEdit()
        self.classify_input.setMaximumHeight(100)
        self.classify_input.setPlaceholderText("Enter text to classify...")
        layout.addWidget(self.classify_input)
        
        # Classify button
        self.classify_btn = QPushButton("Classify Text")
        self.classify_btn.clicked.connect(self.classify_text)
        self.classify_btn.setEnabled(False)
        layout.addWidget(self.classify_btn)
        
        # Output
        layout.addWidget(QLabel("Classification Results:"))
        self.classify_output = QTextBrowser()
        layout.addWidget(self.classify_output)
        
        self.tab_widget.addTab(tab, "Text Classification")
    
    def create_web_interface_tab(self):
        """Create web interface tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Title
        title_label = QLabel("🌐 Gradio Web Interface")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #FF5722; margin: 20px;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel("Launch a web interface to interact with your model through a browser.")
        desc_label.setStyleSheet("font-size: 14px; color: #666; margin: 10px;")
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc_label)
        
        # Launch button
        self.web_interface_btn = QPushButton("🌐 Launch Web Interface")
        self.web_interface_btn.clicked.connect(self.launch_web_interface)
        self.web_interface_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF5722;
                color: white;
                font-weight: bold;
                font-size: 16px;
                padding: 20px;
                border: 3px solid #E64A19;
                border-radius: 10px;
                min-width: 300px;
                min-height: 60px;
                margin: 20px;
            }
            QPushButton:hover {
                background-color: #E64A19;
            }
        """)
        layout.addWidget(self.web_interface_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Status
        self.web_status_label = QLabel("Ready to launch web interface")
        self.web_status_label.setStyleSheet("color: green; font-size: 12px; margin: 10px;")
        self.web_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.web_status_label)
        
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "🌐 Web Interface")
    
    def create_chat_tab(self):
        """Create chat interface tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Chat history display
        layout.addWidget(QLabel("Chat History:"))
        self.chat_history = QTextBrowser()
        self.chat_history.setMinimumHeight(300)
        layout.addWidget(self.chat_history)
        
        # Model capabilities info
        self.chat_capabilities_label = QLabel("Model capabilities will be shown here when a model is loaded")
        self.chat_capabilities_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.chat_capabilities_label)
        
        # Input area
        input_layout = QHBoxLayout()
        
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Type your message here...")
        self.chat_input.returnPressed.connect(self.send_chat_message)
        input_layout.addWidget(self.chat_input)
        
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_chat_message)
        self.send_btn.setEnabled(False)
        input_layout.addWidget(self.send_btn)
        
        layout.addLayout(input_layout)
        
        # Chat controls
        controls_layout = QHBoxLayout()
        
        self.clear_chat_btn = QPushButton("Clear Chat")
        self.clear_chat_btn.clicked.connect(self.clear_chat_history)
        controls_layout.addWidget(self.clear_chat_btn)
        
        self.save_chat_btn = QPushButton("Save Chat")
        self.save_chat_btn.clicked.connect(self.save_chat_history)
        controls_layout.addWidget(self.save_chat_btn)
        
        self.load_chat_btn = QPushButton("Load Chat")
        self.load_chat_btn.clicked.connect(self.load_chat_history)
        controls_layout.addWidget(self.load_chat_btn)
        
        # Chat parameters
        controls_layout.addWidget(QLabel("Max Length:"))
        self.chat_max_length = QComboBox()
        self.chat_max_length.addItems(["100", "200", "500", "1000"])
        self.chat_max_length.setCurrentText("200")
        controls_layout.addWidget(self.chat_max_length)
        
        controls_layout.addWidget(QLabel("Temperature:"))
        self.chat_temperature = QComboBox()
        self.chat_temperature.addItems(["0.1", "0.3", "0.5", "0.7", "0.9", "1.0"])
        self.chat_temperature.setCurrentText("0.7")
        controls_layout.addWidget(self.chat_temperature)
        
        layout.addLayout(controls_layout)
        
        self.tab_widget.addTab(tab, "Chat")
    
    def create_model_info_tab(self):
        """Create model information tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        tab.setLayout(layout)
        
        # Add refresh button for memory info
        refresh_layout = QHBoxLayout()
        refresh_btn = QPushButton("🔄 Refresh Memory Info")
        refresh_btn.clicked.connect(self.refresh_memory_info)
        refresh_layout.addWidget(refresh_btn)
        refresh_layout.addStretch()
        layout.addLayout(refresh_layout)
        
        self.info_browser = QTextBrowser()
        layout.addWidget(self.info_browser)
        
        # Initialize with system info
        self.refresh_memory_info()
        
        self.tab_widget.addTab(tab, "Model Info")
    
    def load_model(self):
        """Load the model for testing using a background thread"""
        try:
            # Check memory before loading
            if not self.check_memory_usage():
                self.status_label.setText("⚠️ Low memory - model loading may fail")
            
            self.status_label.setText("🔄 Loading model in background...")
            self.load_btn.setEnabled(False)
            self.abort_btn.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Unload previous model if any
            if self.pipeline:
                self.unload_model()
            
            # Start model loading in background thread
            self.model_load_thread = ModelLoadThread(self.model_path, self.model_name)
            self.model_load_thread.model_loaded_signal.connect(self.on_model_loaded)
            self.model_load_thread.loading_progress_signal.connect(self.on_loading_progress)
            self.model_load_thread.progress_bar_signal.connect(self.on_progress_update)
            self.model_load_thread.memory_check_signal.connect(self.on_memory_check)
            self.model_load_thread.error_signal.connect(self.on_model_load_error)
            self.model_load_thread.start()
            
        except Exception as e:
            self.status_label.setText(f"❌ Error starting model load: {str(e)}")
            self.load_btn.setEnabled(True)
            self.abort_btn.setEnabled(False)
            self.progress_bar.setVisible(False)
    
    def on_model_loaded(self, pipeline_obj, model_type):
        """Handle successful model loading"""
        try:
            self.pipeline = pipeline_obj
            
            # Update model info
            model_path = Path(self.model_path)
            if (model_path / "config.json").exists():
                import json
                with open(model_path / "config.json", 'r') as f:
                    config = json.load(f)
                
                self.info_browser.append(f"Model Type: {model_type}")
                self.info_browser.append(f"Architecture: {config.get('architectures', ['Unknown'])[0]}")
            
            # Show model type warnings
            self.show_model_type_warning(model_type)
            
            # Refresh memory info to show current usage
            self.refresh_memory_info()
            
            # Enable appropriate buttons based on model type
            if model_type in ["gpt2", "gpt", "causal_lm", "llama", "mistral", "falcon", "bloom", "opt"]:
                self.generate_btn.setEnabled(True)
                self.status_label.setText("✅ Text generation model loaded")
            elif model_type in ["bert", "roberta", "distilbert"]:
                self.classify_btn.setEnabled(True)
                self.status_label.setText("✅ Text classification model loaded")
            elif "translation" in model_type or "marian" in model_type:
                self.status_label.setText("✅ Translation model loaded (use web interface)")
            elif model_type in ["t5", "bart"]:
                self.generate_btn.setEnabled(True)
                self.status_label.setText("✅ Text-to-text generation model loaded")
            
            # Enable chat for text generation models
            supports_chat = model_type in ["gpt2", "gpt", "causal_lm", "llama", "mistral", "falcon", "bloom", "opt", "t5", "bart"]
            if supports_chat:
                self.send_btn.setEnabled(True)
                
                # Try to load chat history from cache
                if not self.load_chat_history_from_cache():
                    # If no cached history, start fresh
                    self.chat_history.append("🤖 <b>AI Assistant:</b> Hello! I'm ready to chat with you. How can I help you today?")
                
                # Show model capabilities and info
                capabilities = []
                if model_type in ["gpt2", "gpt", "causal_lm", "llama", "mistral", "falcon", "bloom", "opt"]:
                    capabilities.append("💬 Natural conversation")
                    capabilities.append("✍️ Text generation")
                    capabilities.append("🧠 Context understanding")
                elif model_type in ["t5", "bart"]:
                    capabilities.append("💬 Question answering")
                    capabilities.append("📝 Text summarization")
                    capabilities.append("🔄 Text transformation")
                
                capabilities_text = " | ".join(capabilities)
                model_info = f"🤖 <b>Model:</b> {self.model_name} ({model_type}) | <b>Capabilities:</b> {capabilities_text}"
                self.chat_capabilities_label.setText(model_info)
                self.chat_capabilities_label.setStyleSheet("color: green; font-weight: bold;")
                
                # Clear chat history for new model
                self.chat_history.clear()
                self.chat_history.append(f"🤖 <b>AI Assistant:</b> Hello! I'm {self.model_name}, a {model_type} model. How can I help you today?")
                
                # Show model loading success
                self.status_label.setText(f"✅ {self.model_name} loaded successfully - Chat enabled!")
            
            self.load_btn.setEnabled(True)
            self.abort_btn.setEnabled(False)
            self.progress_bar.setVisible(False)
            
        except Exception as e:
            self.status_label.setText(f"❌ Error setting up model: {str(e)}")
            self.load_btn.setEnabled(True)
            self.abort_btn.setEnabled(False)
            self.progress_bar.setVisible(False)
            self.pipeline = None
    
    def on_loading_progress(self, message):
        """Handle loading progress updates"""
        self.status_label.setText(f"🔄 {message}")
    
    def on_progress_update(self, value):
        """Handle progress bar updates"""
        self.progress_bar.setValue(value)
    
    def on_memory_check(self, is_sufficient: bool, message: str):
        """Handle memory check results"""
        self.memory_info_label.setText(message)
        if not is_sufficient:
            self.memory_info_label.setStyleSheet("color: orange; font-size: 10px; font-weight: bold;")
        else:
            self.memory_info_label.setStyleSheet("color: green; font-size: 10px;")
    
    def abort_loading(self):
        """Abort the current model loading process"""
        if self.model_load_thread and self.model_load_thread.isRunning():
            self.model_load_thread.abort_loading()
            self.model_load_thread.wait()  # Wait for thread to finish
            self.status_label.setText("⏹️ Model loading aborted")
            self.load_btn.setEnabled(True)
            self.abort_btn.setEnabled(False)
            self.progress_bar.setVisible(False)
            self.memory_info_label.setText("")
    
    def on_model_load_error(self, error_message):
        """Handle model loading errors"""
        self.status_label.setText(f"❌ {error_message}")
        self.load_btn.setEnabled(True)
        self.abort_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.memory_info_label.setText("")
        self.pipeline = None
        # Disable chat if model failed to load
        self.send_btn.setEnabled(False)
        self.chat_capabilities_label.setText("❌ Model failed to load - chat disabled")
        self.chat_capabilities_label.setStyleSheet("color: red; font-weight: bold;")
    
    def unload_model(self):
        """Unload the model and free memory"""
        try:
            if self.pipeline:
                # Clear pipeline
                self.pipeline = None
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Clear CUDA cache if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass
                
                # Disable all buttons
                self.generate_btn.setEnabled(False)
                self.classify_btn.setEnabled(False)
                self.send_btn.setEnabled(False)
                
                # Update status
                self.status_label.setText("✅ Model unloaded and memory freed")
                self.model_loaded = False
                
                # Clear chat capabilities
                self.chat_capabilities_label.setText("❌ No model loaded - chat disabled")
                self.chat_capabilities_label.setStyleSheet("color: red; font-weight: bold;")
                
                # Clear memory info
                self.memory_info_label.setText("")
                
                # Also unload from Gradio interface if it's loaded there
                if self.gradio_manager and self.gradio_manager.is_model_loaded(self.model_name):
                    try:
                        self.gradio_manager.stop_interface(self.model_name)
                        self.status_label.setText("✅ Model unloaded (including Gradio interface)")
                    except Exception as e:
                        print(f"Warning: Could not unload model from Gradio: {e}")
                
                # Update Gradio status
                self.update_gradio_status()
                
        except Exception as e:
            self.status_label.setText(f"❌ Error unloading model: {str(e)}")
    
    def update_gradio_status(self):
        """Update the Gradio status indicator"""
        if self.gradio_manager:
            active_interfaces = self.gradio_manager.get_active_interfaces()
            loaded_models = self.gradio_manager.get_loaded_models()
            
            if active_interfaces:
                status_text = f"🌐 Active Gradio interfaces: {', '.join(active_interfaces)}"
                if loaded_models:
                    status_text += f" | Loaded models: {', '.join(loaded_models)}"
                self.gradio_status_label.setText(status_text)
                self.gradio_status_label.setStyleSheet("color: green; font-size: 10px; font-weight: bold;")
            else:
                self.gradio_status_label.setText("🌐 No active Gradio interfaces")
                self.gradio_status_label.setStyleSheet("color: gray; font-size: 10px;")
        else:
            self.gradio_status_label.setText("🌐 Gradio not available")
            self.gradio_status_label.setStyleSheet("color: red; font-size: 10px;")
    
    def check_memory_usage(self):
        """Check if system has enough memory for large models"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            total_gb = memory.total / (1024**3)
            
            # Check VRAM if CUDA is available
            vram_info = ""
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                vram_info = f" | VRAM: {vram_gb:.1f}GB"
            
            # Update memory info label
            self.memory_info_label.setText(f"RAM: {available_gb:.1f}GB/{total_gb:.1f}GB available{vram_info}")
            
            if available_gb < 4:
                if not self.memory_warning_shown:
                    QMessageBox.warning(
                        self, 
                        "Low Memory Warning", 
                        f"Only {available_gb:.1f}GB RAM available. Large models may cause issues.\n\n"
                        "Consider:\n"
                        "- Closing other applications\n"
                        "- Using smaller models\n"
                        "- Adding more RAM"
                    )
                    self.memory_warning_shown = True
                return False
            return True
        except:
            return True  # Assume OK if we can't check
    
    def show_model_type_warning(self, model_type: str):
        """Show warnings for specific model types"""
        if "vision" in model_type or "image" in model_type:
            if not torch.cuda.is_available():
                QMessageBox.warning(
                    self,
                    "Vision Model Warning",
                    f"This is a vision model ({model_type}) but CUDA is not available.\n\n"
                    "Vision models typically require significant VRAM and may run very slowly on CPU.\n\n"
                    "Consider:\n"
                    "- Using a GPU-enabled system\n"
                    "- Using smaller vision models\n"
                    "- Using text-based models instead"
                )
            else:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if vram_gb < 4:
                    QMessageBox.warning(
                        self,
                        "Low VRAM Warning",
                        f"This is a vision model ({model_type}) but only {vram_gb:.1f}GB VRAM is available.\n\n"
                        "Vision models typically require 4GB+ VRAM for optimal performance.\n\n"
                        "The model may run slowly or fail to load."
                    )
    
    def refresh_memory_info(self):
        """Refresh and display current memory usage information"""
        try:
            import psutil
            
            # Get system memory info
            memory = psutil.virtual_memory()
            total_ram_gb = memory.total / (1024**3)
            available_ram_gb = memory.available / (1024**3)
            used_ram_gb = memory.used / (1024**3)
            ram_percent = memory.percent
            
            # Get VRAM info if available
            vram_info = ""
            if torch.cuda.is_available():
                vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                vram_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                vram_cached = torch.cuda.memory_reserved(0) / (1024**3)
                vram_free = vram_total - vram_allocated
                vram_info = f"""
🎮 GPU Memory (VRAM):
   Total: {vram_total:.1f}GB
   Allocated: {vram_allocated:.1f}GB
   Cached: {vram_cached:.1f}GB
   Free: {vram_free:.1f}GB
   Device: {torch.cuda.get_device_name(0)}
"""
            else:
                vram_info = "🎮 GPU: CUDA not available (CPU only mode)"
            
            # Get disk space info
            disk = psutil.disk_usage('/')
            total_disk_gb = disk.total / (1024**3)
            free_disk_gb = disk.free / (1024**3)
            used_disk_gb = disk.used / (1024**3)
            disk_percent = (disk.used / disk.total) * 100
            
            # Display info
            info_text = f"""💻 System Information:

🧠 RAM Memory:
   Total: {total_ram_gb:.1f}GB
   Used: {used_ram_gb:.1f}GB ({ram_percent:.1f}%)
   Available: {available_ram_gb:.1f}GB

{vram_info}

💾 Disk Space:
   Total: {total_disk_gb:.1f}GB
   Used: {used_disk_gb:.1f}GB ({disk_percent:.1f}%)
   Free: {free_disk_gb:.1f}GB

📊 Model Loading Recommendations:
   • Small models (<1GB): {available_ram_gb:.1f}GB RAM sufficient
   • Medium models (1-4GB): {available_ram_gb:.1f}GB RAM should be OK
   • Large models (>4GB): {available_ram_gb:.1f}GB RAM may be insufficient
   • Vision models: {'GPU recommended' if torch.cuda.is_available() else 'Will run on CPU (slow)'}
"""
            
            self.info_browser.setText(info_text)
            
        except Exception as e:
            self.info_browser.setText(f"Error getting system information: {str(e)}")
    
    def generate_text(self):
        """Generate text using the loaded model"""
        try:
            input_text = self.text_input.toPlainText().strip()
            if not input_text:
                self.text_output.setText("Please enter some input text")
                return
            
            max_length = int(self.max_length_spin.currentText())
            temperature = float(self.temperature_spin.currentText())
            
            result = self.pipeline(input_text, max_length=max_length, temperature=temperature, do_sample=True)
            
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0]['generated_text']
                self.text_output.setText(generated_text)
            else:
                self.text_output.setText("No output generated")
                
        except Exception as e:
            self.text_output.setText(f"Error generating text: {str(e)}")
    
    def classify_text(self):
        """Classify text using the loaded model"""
        try:
            input_text = self.classify_input.toPlainText().strip()
            if not input_text:
                self.classify_output.setText("Please enter some text to classify")
                return
            
            result = self.pipeline(input_text)
            
            if isinstance(result, list) and len(result) > 0:
                output = "Classification Results:\n\n"
                for item in result:
                    output += f"Label: {item['label']}\n"
                    output += f"Score: {item['score']:.4f}\n\n"
                self.classify_output.setText(output)
            else:
                self.classify_output.setText("No classification results")
                
        except Exception as e:
            self.classify_output.setText(f"Error classifying text: {str(e)}")
    
    def translate_text(self):
        """Translation moved to web interface"""
        QMessageBox.information(self, "Translation", "Translation functionality is now available in the Web Interface tab.")
    
    def send_chat_message(self):
        """Send a chat message and get AI response (optimized for chat models)"""
        try:
            if not self.pipeline:
                self.chat_history.append("🤖 <b>AI Assistant:</b> Error: Model not loaded. Please load the model first.")
                return
            message = self.chat_input.text().strip()
            if not message:
                return
            # Add user message to chat history
            from datetime import datetime
            timestamp = datetime.now().strftime("%H:%M")
            self.chat_history.append(f"👤 <b>You ({timestamp}):</b> {message}")
            self.chat_input.clear()
            # Show typing indicator
            self.chat_history.append("🤖 <b>AI Assistant:</b> <i>Thinking...</i>")
            QApplication.processEvents()
            max_length = int(self.chat_max_length.currentText())
            temperature = float(self.chat_temperature.currentText())
            
            # Check if this is a GGUF model (llama-cpp-python)
            if hasattr(self.pipeline, 'create_chat_completion'):
                # Use proper chat completion for GGUF models
                messages = [{"role": "user", "content": message}]
                result = self.pipeline.create_chat_completion(
                    messages=messages,
                    max_tokens=max_length,
                    temperature=temperature,
                    top_p=0.9,
                    stop=["User:", "Human:", "\n\n"]
                )
                
                if result and 'choices' in result and len(result['choices']) > 0:
                    new_response = result['choices'][0]['message']['content'].strip()
                else:
                    new_response = "I'm here to help. What would you like to know?"
            else:
                # Use chat-style prompt for causal LMs
                context = self.get_conversation_context()
                if hasattr(self.pipeline, 'task') and self.pipeline.task == 'text2text-generation':
                    result = self.pipeline(message, max_length=max_length, temperature=temperature, do_sample=True)
                else:
                    prompt = context + f" User: {message}\nAssistant:"
                    result = self.pipeline(prompt, max_length=max_length, temperature=temperature, do_sample=True)
                
                if isinstance(result, list) and len(result) > 0:
                    if hasattr(self.pipeline, 'task') and self.pipeline.task == 'text2text-generation':
                        new_response = result[0]['generated_text']
                    else:
                        full_response = result[0]['generated_text']
                        prompt = context + f" User: {message}\nAssistant:"
                        if full_response.startswith(prompt):
                            new_response = full_response[len(prompt):].strip()
                        else:
                            new_response = full_response.strip()
                else:
                    new_response = "I'm not sure how to respond to that."
            
            # Remove the "thinking..." message
            cursor = self.chat_history.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.movePosition(cursor.MoveOperation.StartOfLine, cursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
            cursor.deletePreviousChar()
            
            # Clean up the response and format it nicely
            if new_response:
                response = self.clean_repetitive_response(new_response)
                if len(response) > 300:
                    sentences = response.split('.')
                    if len(sentences) > 1:
                        response = '. '.join(sentences[:-1]) + '.'
                    else:
                        response = response[:300] + "..."
                import re
                if not response or len(response.strip()) < 5:
                    response = "Hello! I'm here to help. How can I assist you today?"
                elif len(response) < 10 and response.lower() in ['hi', 'hello', 'hey']:
                    response = "Hello! Nice to meet you. How can I help you today?"
                elif len(response) < 15 or not re.search(r'[a-zA-Z]', response):
                    response = "I understand. Could you please rephrase your question or ask me something else?"
                elif response.count('!') > 3 or response.count('?') > 3:
                    response = re.sub(r'[!?]{2,}', '!', response)
                    if len(response.strip()) < 10:
                        response = "I'm here to help. What would you like to know?"
                self.chat_history.append(f"🤖 <b>AI Assistant ({timestamp}):</b> {response}")
            else:
                self.chat_history.append(f"🤖 <b>AI Assistant ({timestamp}):</b> I'm not sure how to respond to that.")
            self.save_chat_history_to_cache()
        except Exception as e:
            cursor = self.chat_history.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            cursor.movePosition(cursor.MoveOperation.StartOfLine, cursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
            cursor.deletePreviousChar()
            self.chat_history.append(f"🤖 <b>AI Assistant:</b> Error: {str(e)}")
    
    def clean_repetitive_response(self, response: str) -> str:
        """Clean repetitive content from model response with advanced handling"""
        if not response:
            return response
        import re
        response = re.sub(r'[!]{2,}', '!', response)
        response = re.sub(r'[?]{2,}', '?', response)
        response = re.sub(r'\d+[!?]*', '', response)
        response = re.sub(r'\b[a-zA-Z]\s+[a-zA-Z]\s+[a-zA-Z]\s+[a-zA-Z]\b', '', response)
        response = re.sub(r'\b[A-Za-z]\s+[A-Za-z]\s+[A-Za-z]\s+[A-Za-z]\s+[A-Za-z]\b', '', response)
        words = response.split()
        cleaned_words = []
        prev_word = None
        repeat_count = 0
        for word in words:
            if word == prev_word:
                repeat_count += 1
                if repeat_count > 1:
                    continue
            else:
                repeat_count = 0
            cleaned_words.append(word)
            prev_word = word
        response = ' '.join(cleaned_words)
        response = re.sub(r'\b(me|I|you|he|she|it|we|they)\s+\1\b', r'\1', response, flags=re.IGNORECASE)
        response = re.sub(r'\b(thanks|thank you)\s+\1\b', r'\1', response, flags=re.IGNORECASE)
        response = re.sub(r'\b(yeah|yes|no|ok|okay)\s+\1\b', r'\1', response, flags=re.IGNORECASE)
        response = re.sub(r'\b(haha|lol|XD)\s+\1\b', r'\1', response, flags=re.IGNORECASE)
        response = re.sub(r'\s+', ' ', response).strip()
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        sentences = [s for s in sentences if len(s) > 10 and re.search(r'[a-zA-Z]', s)]
        unique_sentences = []
        for sentence in sentences:
            is_duplicate = False
            sentence_words = set(sentence.lower().split())
            for existing in unique_sentences:
                existing_words = set(existing.lower().split())
                if len(sentence_words) > 0 and len(existing_words) > 0:
                    similarity = len(sentence_words.intersection(existing_words)) / len(sentence_words.union(existing_words))
                    if similarity > 0.6:
                        is_duplicate = True
                        break
            if not is_duplicate:
                unique_sentences.append(sentence)
        if len(unique_sentences) > 3:
            unique_sentences = unique_sentences[:3]
        result = '. '.join(unique_sentences)
        if result and not result.endswith('.'):
            result += '.'
        if len(result) < 5 or not re.search(r'[a-zA-Z]', result):
            sentences = re.split(r'[.!?]', response)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 5 and re.search(r'[a-zA-Z]', sentence):
                    result = sentence
                    break
        return result
    
    def get_conversation_context(self):
        """Get the conversation context for the model"""
        # Get the last few messages for context
        chat_text = self.chat_history.toPlainText()
        lines = chat_text.split('\n')
        
        # Extract user and AI messages for better context
        conversation = []
        for line in lines:
            if line.startswith('👤 You:'):
                message = line.replace('👤 You:', '').strip()
                conversation.append(f"User: {message}")
            elif line.startswith('🤖 AI Assistant:') and not line.startswith('🤖 AI Assistant: <i>Thinking...</i>'):
                message = line.replace('🤖 AI Assistant:', '').strip()
                conversation.append(f"Assistant: {message}")
        
        # Use the last few exchanges for context
        recent_conversation = conversation[-6:]  # Last 6 messages (3 exchanges)
        context = " ".join(recent_conversation)
        
        return context
    
    def clear_chat_history(self):
        """Clear the chat history"""
        self.chat_history.clear()
        if self.pipeline:
            self.chat_history.append("🤖 <b>AI Assistant:</b> Hello! I'm ready to chat with you. How can I help you today?")
        # Save empty history to cache
        self.save_chat_history_to_cache()
    
    def save_chat_history(self):
        """Save chat history to a file"""
        try:
            from PyQt6.QtWidgets import QFileDialog
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Chat History", "", "Text Files (*.txt);;All Files (*)"
            )
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.chat_history.toPlainText())
                QMessageBox.information(self, "Success", "Chat history saved successfully!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save chat history: {str(e)}")
    
    def load_chat_history(self):
        """Load chat history from a file"""
        try:
            from PyQt6.QtWidgets import QFileDialog
            filename, _ = QFileDialog.getOpenFileName(
                self, "Load Chat History", "", "Text Files (*.txt);;All Files (*)"
            )
            if filename:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.chat_history.setPlainText(content)
                QMessageBox.information(self, "Success", "Chat history loaded successfully!")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load chat history: {str(e)}")
    
    def save_chat_history_to_cache(self):
        """Save chat history to local cache"""
        try:
            with open(self.chat_history_file, 'w', encoding='utf-8') as f:
                f.write(self.chat_history.toPlainText())
        except Exception as e:
            print(f"Warning: Could not save chat history to cache: {e}")
    
    def load_chat_history_from_cache(self):
        """Load chat history from local cache"""
        try:
            if self.chat_history_file.exists():
                with open(self.chat_history_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.chat_history.setPlainText(content)
                return True
        except Exception as e:
            print(f"Warning: Could not load chat history from cache: {e}")
        return False
    
    def launch_web_interface(self):
        """Launch Gradio web interface for the model"""
        if not GRADIO_AVAILABLE:
            QMessageBox.warning(self, "Gradio Not Available", 
                              "Gradio is not installed. Please install it with: pip install gradio")
            return
        
        if not self.gradio_manager:
            QMessageBox.warning(self, "Error", "Gradio manager not initialized")
            return
        
        try:
            # Launch the web interface
            url = self.gradio_manager.launch_interface(self.model_path, self.model_name)
            
            # Show success message
            QMessageBox.information(self, "Web Interface Launched", 
                                  f"Web interface launched successfully!\n\n"
                                  f"URL: {url}\n\n"
                                  f"The interface should open in your default browser.\n"
                                  f"You can share this URL with others to let them use the model.")
            
            # Update button text
            if self.web_interface_btn:
                self.web_interface_btn.setText("🌐 Web Interface Active")
                self.web_interface_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
            
            # Update Gradio status
            self.update_gradio_status()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch web interface: {str(e)}")
    
    def closeEvent(self, event):
        """Handle dialog close event - clean up Gradio interfaces"""
        if self.gradio_manager and self.model_name in self.gradio_manager.get_active_interfaces():
            self.gradio_manager.stop_interface(self.model_name)
        event.accept()


class ModelDownloader(QThread):
    """Thread for downloading models without blocking the GUI"""
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    progress_bar_signal = pyqtSignal(int)  # NEW: signal for progress bar
    download_stats_signal = pyqtSignal(str)  # NEW: signal for download statistics
    
    def __init__(self, model_name: str, target_path: str, model_type: str = "transformers"):
        super().__init__()
        self.model_name = model_name
        self.target_path = target_path
        self.model_type = model_type
        
        # Set up cache directories in the local user directory instead of external drive
        home_dir = Path.home()
        self.cache_dir = home_dir / ".huggingdrive_cache"
        self.model_cache_dir = self.cache_dir / "model_cache"
        self.datasets_cache_dir = self.cache_dir / "datasets"
        
        # Create cache directories in local folder
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.model_cache_dir.mkdir(parents=True, exist_ok=True)
            self.datasets_cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create cache directories: {e}")
            # Continue without cache directories
        
        # Set environment variables for transformers cache
        import os
        os.environ['TRANSFORMERS_CACHE'] = str(self.model_cache_dir)
        os.environ['HF_HOME'] = str(self.cache_dir)
        os.environ['HF_DATASETS_CACHE'] = str(self.datasets_cache_dir)
        
    def run(self):
        try:
            self.progress_signal.emit(f"Starting download of {self.model_name}...")
            self.progress_signal.emit(f"Target directory: {self.target_path}")
            
            # Validate target path
            if not self.target_path:
                raise ValueError("Target path is empty")
            
            # Create target directory
            try:
                os.makedirs(self.target_path, exist_ok=True)
                self.progress_signal.emit("Created target directory")
            except PermissionError as e:
                error_msg = f"❌ Permission denied: Cannot write to drive. The drive may be read-only or you don't have write permissions."
                self.progress_signal.emit(error_msg)
                self.progress_signal.emit("Please check drive permissions or select a different drive.")
                self.finished_signal.emit(False, error_msg)
                return
            except OSError as e:
                if e.errno == 30:  # Read-only file system
                    error_msg = f"❌ Drive is read-only: Cannot write to {self.target_path}"
                    self.progress_signal.emit(error_msg)
                    self.progress_signal.emit("Please select a different drive or check drive permissions.")
                    self.finished_signal.emit(False, error_msg)
                    return
                else:
                    raise Exception(f"Failed to create target directory: {str(e)}")
            except Exception as e:
                raise Exception(f"Failed to create target directory: {str(e)}")
            
            # Check for specialized models that might cause issues
            if "unsloth" in self.model_name.lower():
                self.progress_signal.emit("⚠️ Warning: Unsloth model detected!")
                self.progress_signal.emit("This model requires special setup and may not work with standard transformers.")
                self.progress_signal.emit("Consider using a standard model for better compatibility.")
            
            # Check if model exists and get info
            try:
                from huggingface_hub import model_info, HfApi
                api = HfApi()
                
                # Check if model exists
                try:
                    info = api.model_info(self.model_name)
                    self.progress_signal.emit(f"✅ Model found: {self.model_name}")
                    self.progress_signal.emit(f"Model info retrieved - preparing to download files...")
                except Exception as e:
                    error_msg = f"❌ Model '{self.model_name}' not found on Hugging Face Hub"
                    self.progress_signal.emit(error_msg)
                    self.progress_signal.emit(f"Error: {str(e)}")
                    self.progress_signal.emit("Please check the model name and try again.")
                    self.finished_signal.emit(False, error_msg)
                    return
                
                # Show what files will be downloaded
                total_files = 0
                if hasattr(info, 'safetensors') and info.safetensors:
                    total_files = len(info.safetensors)
                    self.progress_signal.emit(f"Found {total_files} safetensors files to download")
                elif hasattr(info, 'pytorch_model') and info.pytorch_model:
                    total_files = len(info.pytorch_model)
                    self.progress_signal.emit(f"Found {total_files} pytorch model files to download")
                
            except Exception as e:
                self.progress_signal.emit(f"Could not get model info: {str(e)}")
                total_files = 1  # Assume at least one file
            
            # Download with progress tracking
            self.progress_signal.emit("Starting file download...")
            self.progress_bar_signal.emit(10)  # 10% - started
            
            # Show download start time
            import time
            start_time = time.time()
            self.download_stats_signal.emit(f"Download started at {time.strftime('%H:%M:%S')}")
            
            # Use snapshot_download with periodic progress updates
            from huggingface_hub import snapshot_download
            import threading
            import time
            
            # Start a background thread to update progress periodically
            def update_progress():
                progress = 10
                while progress < 90:
                    time.sleep(1)  # Update every second
                    progress += 5  # Increment by 5% each second
                    if progress < 90:
                        self.progress_bar_signal.emit(progress)
                        self.progress_signal.emit(f"Downloading... {progress}% complete")
            
            # Start progress update thread
            progress_thread = threading.Thread(target=update_progress, daemon=True)
            progress_thread.start()
            
            # Download the model
            snapshot_download(
                repo_id=self.model_name,
                local_dir=self.target_path,
                local_dir_use_symlinks=False
            )
            
            # Show download completion time and duration
            end_time = time.time()
            duration = end_time - start_time
            self.download_stats_signal.emit(f"Download completed in {duration:.1f} seconds")
            
            self.progress_bar_signal.emit(90)  # 90% - download completed
            
            # Show completion details
            self.progress_signal.emit("Download completed!")
            self.progress_signal.emit(f"Verifying downloaded files...")
            
            # Verify the download
            downloaded_files = list(Path(self.target_path).rglob("*"))
            file_count = len([f for f in downloaded_files if f.is_file()])
            self.progress_signal.emit(f"Downloaded {file_count} files successfully")
            
            # Show some key files that were downloaded
            key_files = ["config.json", "pytorch_model.bin", "tokenizer.json", "model.safetensors"]
            found_files = []
            for key_file in key_files:
                if (Path(self.target_path) / key_file).exists():
                    found_files.append(key_file)
            
            if found_files:
                self.progress_signal.emit(f"Key files downloaded: {', '.join(found_files)}")
            
            self.progress_bar_signal.emit(100)
            self.progress_signal.emit(f"✅ Successfully downloaded {self.model_name}")
            self.finished_signal.emit(True, f"Model {self.model_name} downloaded successfully!")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            error_msg = f"❌ Error downloading {self.model_name}: {str(e)}"
            self.progress_signal.emit(error_msg)
            self.progress_signal.emit(f"Error details: {error_details}")
            self.finished_signal.emit(False, error_msg)


class DriveManager:
    """Manages external drives and model installations"""
    
    def __init__(self):
        self.config_file = Path.home() / ".huggingdrive" / "config.json"
        self.config_file.parent.mkdir(exist_ok=True)
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {
                "external_drives": {},
                "installed_models": {},
                "default_drive": None,
                "custom_folders": []  # Store custom folder paths
            }
            self.save_config()
    
    def save_config(self):
        """Save configuration to file"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def add_custom_folder(self, folder_path: str):
        """Add a custom folder to the configuration"""
        if "custom_folders" not in self.config:
            self.config["custom_folders"] = []
        
        if folder_path not in self.config["custom_folders"]:
            self.config["custom_folders"].append(folder_path)
            self.save_config()
    
    def get_custom_folders(self) -> List[str]:
        """Get list of custom folders from configuration"""
        return self.config.get("custom_folders", [])
    
    def is_drive_writable(self, drive_path: str) -> bool:
        """Check if a drive is writable by attempting to create a test file"""
        try:
            test_file = Path(drive_path) / ".huggingdrive_test_write"
            test_file.touch()
            test_file.unlink()  # Remove the test file
            return True
        except (OSError, PermissionError):
            return False
    
    def get_external_drives(self) -> List[Dict]:
        """Get list of available external drives (cross-platform)"""
        drives = []
        system = platform.system()
        for partition in psutil.disk_partitions():
            try:
                # macOS: external drives are usually under /Volumes/
                if system == "Darwin":
                    if partition.mountpoint.startswith("/Volumes/"):
                        usage = psutil.disk_usage(partition.mountpoint)
                        
                        # Check if drive is writable
                        is_writable = self.is_drive_writable(partition.mountpoint)
                        
                        drives.append({
                            'device': partition.device,
                            'mountpoint': partition.mountpoint,
                            'filesystem': partition.fstype,
                            'total_gb': usage.total / (1024**3),
                            'free_gb': usage.free / (1024**3),
                            'used_gb': usage.used / (1024**3),
                            'writable': is_writable
                        })
                # Windows: look for removable drives
                elif system == "Windows":
                    if 'removable' in partition.opts:
                        usage = psutil.disk_usage(partition.mountpoint)
                        drives.append({
                            'device': partition.device,
                            'mountpoint': partition.mountpoint,
                            'filesystem': partition.fstype,
                            'total_gb': usage.total / (1024**3),
                            'free_gb': usage.free / (1024**3),
                            'used_gb': usage.used / (1024**3)
                        })
                # Linux: look for /media or /run/media
                else:
                    if partition.mountpoint.startswith("/media/") or partition.mountpoint.startswith("/run/media/"):
                        usage = psutil.disk_usage(partition.mountpoint)
                        drives.append({
                            'device': partition.device,
                            'mountpoint': partition.mountpoint,
                            'filesystem': partition.fstype,
                            'total_gb': usage.total / (1024**3),
                            'free_gb': usage.free / (1024**3),
                            'used_gb': usage.used / (1024**3)
                        })
            except PermissionError:
                continue
        return drives
    
    def scan_for_models(self, drive_path: str) -> List[Dict]:
        """Scan a drive for existing Hugging Face models"""
        models = []
        drive_path = Path(drive_path)
        
        # Look for models in the default location
        default_models_dir = drive_path / "huggingface_models"
        
        if default_models_dir.exists():
            for model_dir in default_models_dir.iterdir():
                if model_dir.is_dir():
                    # Check if it contains model files
                    if self.is_valid_model_directory(model_dir):
                        model_name = model_dir.name.replace("_", "/")  # Convert back from filesystem-safe name
                        models.append({
                            "name": model_name,
                            "path": str(model_dir),
                            "drive": str(drive_path),
                            "detected_at": str(Path().cwd()),
                            "source": "auto-detected"
                        })
        
        return models
    
    def scan_custom_folder_for_models(self, custom_folder_path: str) -> List[Dict]:
        """Scan a custom folder for existing Hugging Face models"""
        models = []
        custom_folder = Path(custom_folder_path)
        
        if custom_folder.exists():
            for model_dir in custom_folder.iterdir():
                if model_dir.is_dir():
                    # Check if it contains model files
                    if self.is_valid_model_directory(model_dir):
                        model_name = model_dir.name.replace("_", "/")  # Convert back from filesystem-safe name
                        models.append({
                            "name": model_name,
                            "path": str(model_dir),
                            "drive": str(custom_folder.parent),  # Use parent as drive
                            "detected_at": str(Path().cwd()),
                            "source": "auto-detected"
                        })
        
        return models
    
    def is_valid_model_directory(self, model_dir: Path) -> bool:
        """Check if a directory contains a valid Hugging Face model"""
        # Look for common model files
        model_files = [
            "config.json",
            "pytorch_model.bin",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "model.safetensors"
        ]
        
        # Check if at least one model file exists
        return any((model_dir / file).exists() for file in model_files)
    
    def update_installed_models_from_drives(self):
        """Scan all drives and update the installed models list"""
        drives = self.get_external_drives()
        all_models = {}
        
        # Scan default locations on drives
        for drive in drives:
            drive_models = self.scan_for_models(drive['mountpoint'])
            for model in drive_models:
                all_models[model['name']] = model
        
        # Scan custom folders
        custom_folders = self.get_custom_folders()
        for custom_folder in custom_folders:
            custom_models = self.scan_custom_folder_for_models(custom_folder)
            for model in custom_models:
                all_models[model['name']] = model
        
        # Update config with detected models
        self.config["installed_models"] = all_models
        self.save_config()
        return all_models
    
    def install_model(self, model_name: str, drive_path: str) -> str:
        """Install a model to the specified drive"""
        model_dir = Path(drive_path) / "huggingface_models" / model_name.replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Download the model
        snapshot_download(
            repo_id=model_name,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False
        )
        
        # Update config
        self.config["installed_models"][model_name] = {
            "path": str(model_dir),
            "drive": drive_path,
            "installed_at": str(Path().cwd()),
            "source": "downloaded"
        }
        self.save_config()
        
        return str(model_dir)
    
    def get_installed_models(self) -> Dict:
        """Get list of installed models"""
        return self.config["installed_models"]
    
    def remove_model(self, model_name: str) -> bool:
        """Remove an installed model"""
        if model_name in self.config["installed_models"]:
            model_path = self.config["installed_models"][model_name]["path"]
            
            # Check if the model directory exists
            if not Path(model_path).exists():
                # Model directory doesn't exist, just remove from config
                del self.config["installed_models"][model_name]
                self.save_config()
                return True
            
            try:
                # Handle macOS hidden files and external drive issues
                import os
                import stat
                import errno
                
                def handle_remove_readonly(func, path, exc):
                    """Handle read-only files on external drives"""
                    excvalue = exc[1]
                    if func in (os.rmdir, os.remove, os.unlink) and excvalue.errno == errno.EACCES:
                        # Make file writable and retry
                        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 0777
                        func(path)
                    else:
                        raise
                
                # Remove the directory with error handling
                shutil.rmtree(model_path, onerror=handle_remove_readonly)
                
                # Remove from config
                del self.config["installed_models"][model_name]
                self.save_config()
                return True
                
            except Exception as e:
                print(f"Error removing model: {e}")
                # Try alternative removal method
                try:
                    import subprocess
                    # Use system command to force remove
                    if sys.platform == "darwin":  # macOS
                        subprocess.run(["rm", "-rf", model_path], check=True)
                    else:
                        subprocess.run(["rmdir", "/s", "/q", model_path], check=True, shell=True)
                    
                    # Remove from config
                    del self.config["installed_models"][model_name]
                    self.save_config()
                    return True
                    
                except Exception as e2:
                    print(f"Alternative removal also failed: {e2}")
                    # Even if we can't delete the folder, remove from config and inform user
                    del self.config["installed_models"][model_name]
                    self.save_config()
                    return True  # Return True but user needs to manually delete folder
        return False


# Add category-specific popular models with approximate sizes
POPULAR_MODELS = {
    "all": [
        {"name": "gpt2", "size": "500MB"},
        {"name": "gpt2-medium", "size": "1.5GB"},
        {"name": "gpt2-large", "size": "3GB"},
        {"name": "bert-base-uncased", "size": "420MB"},
        {"name": "bert-large-uncased", "size": "1.3GB"},
        {"name": "distilbert-base-uncased", "size": "250MB"},
        {"name": "roberta-base", "size": "500MB"},
        {"name": "microsoft/DialoGPT-medium", "size": "1.5GB"},
        {"name": "EleutherAI/gpt-neo-125M", "size": "500MB"},
        {"name": "EleutherAI/gpt-neo-1.3B", "size": "2.5GB"},
        {"name": "Helsinki-NLP/opus-mt-en-fr", "size": "300MB"},
        {"name": "google/vit-base-patch16-224", "size": "330MB"},
        {"name": "microsoft/resnet-50", "size": "98MB"},
        {"name": "openai/whisper-base", "size": "500MB"},
        {"name": "openai/whisper-small", "size": "500MB"},
        {"name": "openai/whisper-medium", "size": "1.5GB"},
        {"name": "runwayml/stable-diffusion-v1-5", "size": "4GB"},
        {"name": "CompVis/stable-diffusion-v1-4", "size": "4GB"},
        {"name": "t5-base", "size": "1GB"},
        {"name": "t5-large", "size": "3GB"},
        {"name": "facebook/bart-base", "size": "500MB"},
        {"name": "facebook/bart-large", "size": "1.6GB"},
        {"name": "microsoft/DialoGPT-small", "size": "500MB"},
        {"name": "microsoft/DialoGPT-large", "size": "3GB"},
    ],
    "text": [
        {"name": "gpt2", "size": "500MB"},
        {"name": "gpt2-medium", "size": "1.5GB"},
        {"name": "gpt2-large", "size": "3GB"},
        {"name": "bert-base-uncased", "size": "420MB"},
        {"name": "bert-large-uncased", "size": "1.3GB"},
        {"name": "distilbert-base-uncased", "size": "250MB"},
        {"name": "roberta-base", "size": "500MB"},
        {"name": "microsoft/DialoGPT-medium", "size": "1.5GB"},
        {"name": "microsoft/DialoGPT-small", "size": "500MB"},
        {"name": "microsoft/DialoGPT-large", "size": "3GB"},
        {"name": "EleutherAI/gpt-neo-125M", "size": "500MB"},
        {"name": "EleutherAI/gpt-neo-1.3B", "size": "2.5GB"},
        {"name": "t5-base", "size": "1GB"},
        {"name": "t5-large", "size": "3GB"},
        {"name": "facebook/bart-base", "size": "500MB"},
        {"name": "facebook/bart-large", "size": "1.6GB"},
    ],
    "vision": [
        {"name": "google/vit-base-patch16-224", "size": "330MB"},
        {"name": "google/vit-large-patch16-224", "size": "1.1GB"},
        {"name": "microsoft/resnet-50", "size": "98MB"},
        {"name": "microsoft/resnet-101", "size": "170MB"},
        {"name": "microsoft/resnet-152", "size": "230MB"},
        {"name": "facebook/detr-resnet-50", "size": "150MB"},
        {"name": "facebook/detr-resnet-101", "size": "250MB"},
        {"name": "microsoft/swin-base-patch4-window7-224", "size": "200MB"},
        {"name": "microsoft/swin-large-patch4-window7-224", "size": "650MB"},
    ],
    "audio": [
        {"name": "openai/whisper-base", "size": "500MB"},
        {"name": "openai/whisper-small", "size": "500MB"},
        {"name": "openai/whisper-medium", "size": "1.5GB"},
        {"name": "openai/whisper-large", "size": "3GB"},
        {"name": "facebook/wav2vec2-base", "size": "95MB"},
        {"name": "facebook/wav2vec2-large", "size": "1.2GB"},
        {"name": "facebook/wav2vec2-large-xlsr-53", "size": "1.2GB"},
        {"name": "microsoft/speecht5_asr", "size": "1GB"},
        {"name": "microsoft/speecht5_tts", "size": "1GB"},
    ],
    "multimodal": [
        {"name": "runwayml/stable-diffusion-v1-5", "size": "4GB"},
        {"name": "CompVis/stable-diffusion-v1-4", "size": "4GB"},
        {"name": "stabilityai/stable-diffusion-2-1", "size": "4GB"},
        {"name": "microsoft/git-base", "size": "500MB"},
        {"name": "microsoft/git-large", "size": "1.5GB"},
        {"name": "Salesforce/blip-image-captioning-base", "size": "500MB"},
        {"name": "Salesforce/blip-image-captioning-large", "size": "1.5GB"},
        {"name": "microsoft/DialoGPT-medium", "size": "1.5GB"},
        {"name": "Helsinki-NLP/opus-mt-en-fr", "size": "300MB"},
    ]
}

# Model categories for filtering
MODEL_CATEGORIES = {
    "text": ["text-generation", "text-classification", "translation", "summarization", "question-answering"],
    "vision": ["image-classification", "object-detection", "image-segmentation", "image-to-text"],
    "audio": ["audio-classification", "automatic-speech-recognition", "text-to-speech"],
    "multimodal": ["image-to-text", "text-to-image", "visual-question-answering"]
}

class ModelSearchThread(QThread):
    """Thread for searching models on Hugging Face Hub"""
    search_results_signal = pyqtSignal(list)
    error_signal = pyqtSignal(str)
    
    def __init__(self, query: str, category: str = None, max_size_gb: float = None, sort_by: str = "downloads", show_popular: bool = False):
        super().__init__()
        self.query = query
        self.category = category
        self.max_size_gb = max_size_gb
        self.sort_by = sort_by
        self.show_popular = show_popular
        
    def run(self):
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            
            # Build search parameters
            search_params = {
                "limit": 25,  # Reduced limit for faster loading
            }
            
            if self.show_popular:
                # For popular models, use predefined list instead of API search
                # This is much faster than searching the API
                popular_results = []
                
                # Show only GGUF models in popular list
                gguf_popular_models = [
                    {"name": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", "size": "638MB"},
                    {"name": "TheBloke/Llama-2-7B-Chat-GGUF", "size": "4GB"},
                    {"name": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF", "size": "4GB"},
                    {"name": "TheBloke/CodeLlama-7B-Python-GGUF", "size": "4GB"},
                    {"name": "TheBloke/Llama-2-13B-Chat-GGUF", "size": "8GB"},
                    {"name": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF", "size": "26GB"},
                    {"name": "TheBloke/Llama-2-70B-Chat-GGUF", "size": "39GB"},
                    {"name": "TheBloke/CodeLlama-34B-Python-GGUF", "size": "19GB"},
                    {"name": "TheBloke/Qwen2.5-7B-Instruct-GGUF", "size": "4GB"},
                    {"name": "TheBloke/Qwen2.5-14B-Instruct-GGUF", "size": "8GB"},
                ]
                
                for popular_model in gguf_popular_models:
                    popular_results.append({
                        "name": popular_model["name"],
                        "size": popular_model["size"],
                        "downloads": 1000000,  # High download count for popular models
                        "likes": 5000,  # High like count for popular models
                        "tags": ["popular", "gguf"]  # Default tags
                    })
                
                self.search_results_signal.emit(popular_results)
                return  # Exit early for popular models
            else:
                # Always include GGUF in search for better results
                search_params["search"] = f"{self.query} GGUF"
            
            # Add category filter if specified
            if self.category and self.category in MODEL_CATEGORIES:
                # For category filtering, we'll use a more targeted approach
                if self.category == "text":
                    search_params["filter"] = "task:text-generation"
                elif self.category == "vision":
                    search_params["filter"] = "task:image-classification"
                elif self.category == "audio":
                    search_params["filter"] = "task:automatic-speech-recognition"
                elif self.category == "multimodal":
                    search_params["filter"] = "task:image-to-text"
                
                # Category filter applied
                pass
            else:
                # No category filter
                pass
            
            # Get models
            models = list(api.list_models(**search_params))
            
            # Filter for GGUF models only
            gguf_models = []
            for model in models:
                model_id_lower = getattr(model, 'modelId', '').lower()
                # Check if model name contains GGUF indicators
                if any(indicator in model_id_lower for indicator in [
                    '-gguf', 'gguf', 'ggml', 'llama.cpp', 'thebloke'
                ]):
                    gguf_models.append(model)
            
            # Use filtered GGUF models
            models = gguf_models
            
            # Sort models by downloads first, then by size
            if self.sort_by == "downloads":
                models.sort(key=lambda x: getattr(x, 'downloads', 0) or 0, reverse=True)
            elif self.sort_by == "likes":
                models.sort(key=lambda x: getattr(x, 'likes', 0) or 0, reverse=True)
            
            # Format results with size information (optimized for speed)
            results = []
            for model in models:
                # Use estimated size based on model type instead of fetching actual size
                size_info = self.estimate_model_size(getattr(model, 'modelId', ''), getattr(model, 'tags', []))
                
                results.append({
                    "name": getattr(model, 'modelId', ''),
                    "size": size_info,
                    "downloads": getattr(model, 'downloads', 0) or 0,
                    "likes": getattr(model, 'likes', 0) or 0,
                    "tags": getattr(model, 'tags', [])
                })
            
            # Note: Size filtering removed for better performance
            # Models will be checked for available space during download instead
            
            # Filter by popular models if specified
            if self.show_popular:
                popular_names = [m['name'] for m in POPULAR_MODELS]
                popular_results = [r for r in results if r['name'] in popular_names]
                results = popular_results[:10]  # Limit to top 10 popular models
            
            self.search_results_signal.emit(results)
            
        except Exception as e:
            self.error_signal.emit(f"Search error: {str(e)}")
    
    def estimate_model_size(self, model_id: str, tags: list) -> str:
        """Estimate model size based on model name and tags (much faster than API calls)"""
        model_id_lower = model_id.lower()
        tags_lower = [tag.lower() for tag in tags]
        
        # Check for common model patterns and estimate size
        if any(name in model_id_lower for name in ['gpt-4', 'llama-2-70b', 'llama-2-13b', 'llama-3-70b', 'llama-3-8b']):
            return "13.5GB"  # Large language models
        elif any(name in model_id_lower for name in ['gpt-2', 'gpt-3', 'llama-2-7b', 'llama-3-1b']):
            return "1.5GB"   # Medium language models
        elif any(name in model_id_lower for name in ['bert-large', 'roberta-large', 'distilbert-large']):
            return "500MB"   # Large BERT models
        elif any(name in model_id_lower for name in ['bert-base', 'roberta-base', 'distilbert-base']):
            return "420MB"   # Base BERT models
        elif any(name in model_id_lower for name in ['resnet-50', 'resnet-101', 'resnet-152']):
            return "100MB"   # ResNet models
        elif any(name in model_id_lower for name in ['vit-base', 'vit-large']):
            return "330MB"   # Vision Transformer models
        elif any(name in model_id_lower for name in ['whisper-large', 'whisper-medium']):
            return "1.5GB"   # Whisper models
        elif any(name in model_id_lower for name in ['whisper-base', 'whisper-small']):
            return "500MB"   # Smaller Whisper models
        elif any(name in model_id_lower for name in ['stable-diffusion', 'diffusers']):
            return "4GB"     # Diffusion models
        elif any(name in model_id_lower for name in ['t5-large', 't5-base']):
            return "1GB"     # T5 models
        elif any(name in model_id_lower for name in ['gpt-neo-125m', 'gpt-neo-350m']):
            return "500MB"   # GPT-Neo small models
        elif any(name in model_id_lower for name in ['gpt-neo-1.3b', 'gpt-neo-2.7b']):
            return "2.5GB"   # GPT-Neo large models
        
        # Estimate based on tags
        if 'text-generation' in tags_lower:
            return "1GB"     # Default for text generation
        elif 'image-classification' in tags_lower:
            return "200MB"   # Default for image classification
        elif 'automatic-speech-recognition' in tags_lower:
            return "500MB"   # Default for speech recognition
        elif 'text-to-image' in tags_lower or 'image-to-text' in tags_lower:
            return "2GB"     # Default for multimodal
        
        # Default estimate
        return "500MB"
    
    def get_model_size(self, model_id: str) -> str:
        """Get the actual size of a model from Hugging Face Hub (kept for compatibility)"""
        try:
            from huggingface_hub import model_info
            info = model_info(model_id)
            
            # Get size from model info
            if hasattr(info, 'safetensors') and info.safetensors:
                total_size = sum(file.get('size', 0) for file in info.safetensors.values())
                return self.format_size(total_size)
            elif hasattr(info, 'pytorch_model') and info.pytorch_model:
                total_size = sum(file.get('size', 0) for file in info.pytorch_model.values())
                return self.format_size(total_size)
            else:
                return "Unknown"
        except Exception:
            return "Unknown"
    
    def format_size(self, size_bytes: int) -> str:
        """Format size in bytes to human readable format"""
        if size_bytes == 0:
            return "0B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f}{size_names[i]}"
    
    def parse_size_to_gb(self, size_str: str) -> float:
        """Parse size string to GB for filtering"""
        if size_str == "Unknown":
            return None
        
        try:
            size_str = size_str.upper()
            if "GB" in size_str:
                return float(size_str.replace("GB", ""))
            elif "MB" in size_str:
                return float(size_str.replace("MB", "")) / 1024
            elif "KB" in size_str:
                return float(size_str.replace("KB", "")) / (1024 * 1024)
            elif "B" in size_str:
                return float(size_str.replace("B", "")) / (1024 * 1024 * 1024)
            else:
                return None
        except:
            return None
    
class HuggingDriveGUI(QMainWindow):
    """Main GUI window for HuggingDrive"""
    
    def __init__(self):
        super().__init__()
        self.drive_manager = DriveManager()
        self.downloader = None
        self.selected_download_folder = None  # Store the user-selected folder
        self.search_thread = None  # For live search
        self.current_cache_dir = None  # Current cache directory
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("HuggingDrive - External Drive Model Manager")
        self.setGeometry(100, 100, 1200, 800)  # Made window slightly larger
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Top status bar with RAM usage
        status_bar_layout = QHBoxLayout()
        
        # RAM usage indicator
        self.ram_usage_label = QLabel("🧠 RAM: Loading...")
        self.ram_usage_label.setStyleSheet("background-color: #f0f0f0; padding: 5px; border-radius: 3px; font-weight: bold;")
        status_bar_layout.addWidget(self.ram_usage_label)
        
        # App memory usage
        self.app_memory_label = QLabel("📱 App: Loading...")
        self.app_memory_label.setStyleSheet("background-color: #e8f4fd; padding: 5px; border-radius: 3px; font-weight: bold;")
        status_bar_layout.addWidget(self.app_memory_label)
        
        status_bar_layout.addStretch()  # Push labels to the left
        
        # Refresh button for memory info
        refresh_memory_btn = QPushButton("🔄 Refresh")
        refresh_memory_btn.clicked.connect(self.update_memory_display)
        refresh_memory_btn.setStyleSheet("padding: 5px;")
        status_bar_layout.addWidget(refresh_memory_btn)
        
        main_layout.addLayout(status_bar_layout)
        
        # Content layout
        content_layout = QHBoxLayout()
        
        # Left panel - Drive and Model Management
        left_panel = self.create_left_panel()
        content_layout.addWidget(left_panel, 1)
        
        # Right panel - Download and Status
        right_panel = self.create_right_panel()
        content_layout.addWidget(right_panel, 1)
        
        main_layout.addLayout(content_layout)
        
        # Update drive list
        self.update_drive_list()
        
        # Scan for existing models and update the list
        self.scan_for_existing_models()
        
        # Load popular models by default
        self.load_popular_models()
        
        # Initialize memory display
        self.update_memory_display()
        
        # Set up timer to update memory usage every 5 seconds
        from PyQt6.QtCore import QTimer
        self.memory_timer = QTimer()
        self.memory_timer.timeout.connect(self.update_memory_display)
        self.memory_timer.start(5000)  # Update every 5 seconds
        
    def create_left_panel(self) -> QWidget:
        """Create the left panel with drive and model management"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # External Drives Group
        drive_group = QGroupBox("External Drives")
        drive_layout = QVBoxLayout()
        drive_group.setLayout(drive_layout)
        
        self.drive_list = QListWidget()
        self.drive_list.currentItemChanged.connect(self.on_drive_list_selection_changed)
        self.refresh_drives_btn = QPushButton("Refresh Drives")
        self.refresh_drives_btn.clicked.connect(self.update_drive_list)
        
        drive_layout.addWidget(self.drive_list)
        drive_layout.addWidget(self.refresh_drives_btn)
        layout.addWidget(drive_group)
        
        # Installed Models Group
        models_group = QGroupBox("Installed Models")
        models_layout = QVBoxLayout()
        models_group.setLayout(models_layout)
        
        self.model_list = QListWidget()
        self.model_list.currentItemChanged.connect(self.on_model_selection_changed)
        self.remove_model_btn = QPushButton("Remove Selected Model")
        self.remove_model_btn.clicked.connect(self.remove_selected_model)
        
        # Add model interaction buttons
        model_actions_layout = QHBoxLayout()
        
        self.test_model_btn = QPushButton("Test Model")
        self.test_model_btn.clicked.connect(self.test_selected_model)
        self.test_model_btn.setEnabled(False)  # Disabled until model is selected
        
        self.open_model_folder_btn = QPushButton("Open Model Folder")
        self.open_model_folder_btn.clicked.connect(self.open_selected_model_folder)
        self.open_model_folder_btn.setEnabled(False)  # Disabled until model is selected
        
        model_actions_layout.addWidget(self.test_model_btn)
        model_actions_layout.addWidget(self.open_model_folder_btn)
        
        models_layout.addWidget(self.model_list)
        models_layout.addWidget(self.remove_model_btn)
        models_layout.addLayout(model_actions_layout)
        layout.addWidget(models_group)
        
        return panel
    
    def create_right_panel(self) -> QWidget:
        """Create the right panel with download functionality"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Model Download Group
        download_group = QGroupBox("Download Model")
        download_layout = QVBoxLayout()
        download_group.setLayout(download_layout)
        
        # Model name input with live search
        download_layout.addWidget(QLabel("Model Name (search GGUF models on Hugging Face Hub):"))
        self.model_name_input = QLineEdit()
        self.model_name_input.setPlaceholderText("Start typing to search GGUF models...")
        self.model_name_input.textChanged.connect(self.on_model_search_text_changed)
        download_layout.addWidget(self.model_name_input)
        
        # Search filters
        filters_layout = QHBoxLayout()
        
        # Category filter
        filters_layout.addWidget(QLabel("Category:"))
        self.category_combo = QComboBox()
        self.category_combo.addItem("All Categories", "")
        self.category_combo.addItem("Text", "text")
        self.category_combo.addItem("Vision", "vision")
        self.category_combo.addItem("Audio", "audio")
        self.category_combo.addItem("Multimodal", "multimodal")
        self.category_combo.currentTextChanged.connect(self.on_filter_changed)
        filters_layout.addWidget(self.category_combo)
        
        # Size filter removed for better performance
        # Models will be checked for available space during download instead
        
        # Sort by
        filters_layout.addWidget(QLabel("Sort by:"))
        self.sort_combo = QComboBox()
        self.sort_combo.addItem("Downloads", "downloads")
        self.sort_combo.addItem("Likes", "likes")
        self.sort_combo.currentTextChanged.connect(self.on_filter_changed)
        filters_layout.addWidget(self.sort_combo)
        
        download_layout.addLayout(filters_layout)
        
        # Search results list
        download_layout.addWidget(QLabel("Search Results:"))
        self.search_results_list = QListWidget()
        self.search_results_list.setMaximumHeight(150)
        self.search_results_list.itemClicked.connect(self.on_search_result_selected)
        download_layout.addWidget(self.search_results_list)
        
        # Drive selection
        download_layout.addWidget(QLabel("Target Drive:"))
        self.drive_combo = QComboBox()
        self.drive_combo.currentTextChanged.connect(self.on_drive_selection_changed)
        download_layout.addWidget(self.drive_combo)
        
        # Eject drive button
        self.eject_drive_btn = QPushButton("🗂️ Eject Drive")
        self.eject_drive_btn.clicked.connect(self.eject_selected_drive)
        self.eject_drive_btn.setEnabled(False)
        self.eject_drive_btn.setToolTip("Safely eject the selected external drive")
        download_layout.addWidget(self.eject_drive_btn)
        
        # Drive status indicator
        self.drive_status_label = QLabel("No drive selected")
        self.drive_status_label.setStyleSheet("color: red; font-weight: bold;")
        download_layout.addWidget(self.drive_status_label)
        
        # Choose Download Folder button
        self.choose_folder_btn = QPushButton("Choose Download Folder")
        self.choose_folder_btn.clicked.connect(self.choose_download_folder)
        download_layout.addWidget(self.choose_folder_btn)
        
        # Download button
        self.download_btn = QPushButton("Download Model")
        self.download_btn.clicked.connect(self.start_download)
        download_layout.addWidget(self.download_btn)
        
        layout.addWidget(download_group)
        
        # Progress and Status Group
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        status_group.setLayout(status_layout)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFormat("Download progress: %p%")
        self.progress_bar.setTextVisible(True)
        status_layout.addWidget(self.progress_bar)
        
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(200)
        self.status_text.setReadOnly(True)
        status_layout.addWidget(self.status_text)
        
        layout.addWidget(status_group)
        
        # Quick Actions Group
        actions_group = QGroupBox("Quick Actions")
        actions_layout = QVBoxLayout()
        actions_group.setLayout(actions_layout)
        
        self.open_models_dir_btn = QPushButton("Open Models Directory")
        self.open_models_dir_btn.clicked.connect(self.open_models_directory)
        actions_layout.addWidget(self.open_models_dir_btn)
        
        self.load_popular_btn = QPushButton("Load Popular Models")
        self.load_popular_btn.clicked.connect(self.load_popular_models)
        actions_layout.addWidget(self.load_popular_btn)
        
        self.browse_all_models_btn = QPushButton("Browse All Models on Hugging Face")
        self.browse_all_models_btn.clicked.connect(self.browse_all_models)
        actions_layout.addWidget(self.browse_all_models_btn)
        
        self.export_config_btn = QPushButton("Export Configuration")
        self.export_config_btn.clicked.connect(self.export_configuration)
        actions_layout.addWidget(self.export_config_btn)
        
        self.cleanup_cache_btn = QPushButton("Clean Up Old Cache")
        self.cleanup_cache_btn.clicked.connect(self.cleanup_old_cache)
        actions_layout.addWidget(self.cleanup_cache_btn)
        
        self.test_download_btn = QPushButton("Test Download (GPT-2)")
        self.test_download_btn.clicked.connect(self.test_download)
        actions_layout.addWidget(self.test_download_btn)
        
        self.check_system_btn = QPushButton("Check System")
        self.check_system_btn.clicked.connect(self.check_system_requirements)
        actions_layout.addWidget(self.check_system_btn)
        
        layout.addWidget(actions_group)
        
        return panel
    
    def on_drive_list_selection_changed(self, current, previous):
        """When a drive is selected in the list, update the combo box to match."""
        if current:
            drive = current.data(Qt.ItemDataRole.UserRole)
            mountpoint = drive['mountpoint']
            # Find the index in the combo box that matches this mountpoint
            for i in range(self.drive_combo.count()):
                if self.drive_combo.itemData(i) == mountpoint:
                    self.drive_combo.setCurrentIndex(i)
                    break
            
            # Set cache directories for this drive
            self.set_cache_directories(mountpoint)
    
    def set_cache_directories(self, drive_path: str):
        """Set cache directories in local user directory instead of external drive"""
        try:
            # Create cache directories in the local user directory
            home_dir = Path.home()
            cache_dir = home_dir / ".huggingdrive_cache"
            model_cache_dir = cache_dir / "model_cache"
            datasets_cache_dir = cache_dir / "datasets"
            search_cache_dir = cache_dir / "search_cache"
            
            # Create directories
            cache_dir.mkdir(exist_ok=True)
            model_cache_dir.mkdir(exist_ok=True)
            datasets_cache_dir.mkdir(exist_ok=True)
            search_cache_dir.mkdir(exist_ok=True)
            
            # Set environment variables
            import os
            os.environ['TRANSFORMERS_CACHE'] = str(model_cache_dir)
            os.environ['HF_HOME'] = str(cache_dir)
            os.environ['HF_DATASETS_CACHE'] = str(datasets_cache_dir)
            
            self.current_cache_dir = cache_dir
            self.status_text.append(f"Cache set to local directory: {cache_dir}")
            
        except Exception as e:
            self.status_text.append(f"Warning: Could not set cache directories: {str(e)}")
    
    def cleanup_old_cache(self):
        """Clean up old cache files to save space"""
        try:
            if self.current_cache_dir and self.current_cache_dir.exists():
                # Remove cache files older than 30 days
                import time
                current_time = time.time()
                cutoff_time = current_time - (30 * 24 * 60 * 60)  # 30 days
                
                cache_files_removed = 0
                for cache_file in self.current_cache_dir.rglob("*"):
                    if cache_file.is_file() and cache_file.stat().st_mtime < cutoff_time:
                        cache_file.unlink()
                        cache_files_removed += 1
                
                if cache_files_removed > 0:
                    self.status_text.append(f"Cleaned up {cache_files_removed} old cache files")
                    
        except Exception as e:
            self.status_text.append(f"Warning: Could not clean up cache: {str(e)}")
    
    def test_download(self):
        """Test download with a simple model"""
        self.status_text.append("🧪 Testing download with gpt2...")
        self.status_text.append("This will test if the basic download functionality works...")
        
        # Check if drive is selected
        drive_path = self.drive_combo.currentData()
        if not drive_path and not self.selected_download_folder:
            self.status_text.append("❌ No drive selected - please select a drive first")
            QMessageBox.warning(self, "Test Failed", "Please select a drive before testing download")
            return
        
        self.model_name_input.setText("gpt2")
        self.start_download()
    
    def on_drive_selection_changed(self):
        """Handle drive selection change in combo box"""
        drive_path = self.drive_combo.currentData()
        if drive_path:
            self.drive_status_label.setText(f"✅ Drive selected: {Path(drive_path).name}")
            self.drive_status_label.setStyleSheet("color: green; font-weight: bold;")
            self.status_text.append(f"Drive selected: {drive_path}")
            # Enable eject button when drive is selected
            self.eject_drive_btn.setEnabled(True)
        else:
            self.drive_status_label.setText("❌ No drive selected")
            self.drive_status_label.setStyleSheet("color: red; font-weight: bold;")
            # Disable eject button when no drive is selected
            self.eject_drive_btn.setEnabled(False)
    
    def eject_selected_drive(self):
        """Safely eject the selected external drive"""
        drive_path = self.drive_combo.currentData()
        if not drive_path:
            QMessageBox.warning(self, "No Drive Selected", "Please select a drive to eject")
            return
        
        drive_name = Path(drive_path).name
        
        # Confirm ejection
        reply = QMessageBox.question(
            self, 
            "Eject Drive", 
            f"Are you sure you want to eject '{drive_name}'?\n\n"
            "This will safely unmount the drive. Make sure no files are being accessed.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.status_text.append(f"🔄 Ejecting drive: {drive_name}")
                
                # Check if drive is still mounted
                if not Path(drive_path).exists():
                    QMessageBox.information(self, "Drive Not Found", f"Drive '{drive_name}' is no longer mounted")
                    self.status_text.append(f"❌ Drive '{drive_name}' is not mounted")
                    self.update_drive_list()  # Refresh drive list
                    return
                
                # Use system command to eject drive
                import subprocess
                import platform
                
                system = platform.system()
                if system == "Darwin":  # macOS
                    # Use diskutil to eject
                    result = subprocess.run(
                        ["diskutil", "eject", drive_path], 
                        capture_output=True, 
                        text=True
                    )
                    
                    if result.returncode == 0:
                        self.status_text.append(f"✅ Successfully ejected '{drive_name}'")
                        QMessageBox.information(self, "Drive Ejected", f"Drive '{drive_name}' has been safely ejected")
                        
                        # Update drive list to reflect the change
                        self.update_drive_list()
                    else:
                        error_msg = f"Failed to eject drive: {result.stderr}"
                        self.status_text.append(f"❌ {error_msg}")
                        QMessageBox.critical(self, "Eject Failed", error_msg)
                
                elif system == "Windows":
                    # Use PowerShell to eject drive
                    result = subprocess.run(
                        ["powershell", "-Command", f"Eject-Volume -DriveLetter {drive_path[0]} -Confirm:$false"], 
                        capture_output=True, 
                        text=True
                    )
                    
                    if result.returncode == 0:
                        self.status_text.append(f"✅ Successfully ejected '{drive_name}'")
                        QMessageBox.information(self, "Drive Ejected", f"Drive '{drive_name}' has been safely ejected")
                        self.update_drive_list()
                    else:
                        error_msg = f"Failed to eject drive: {result.stderr}"
                        self.status_text.append(f"❌ {error_msg}")
                        QMessageBox.critical(self, "Eject Failed", error_msg)
                
                else:  # Linux
                    # Use umount to unmount drive
                    result = subprocess.run(
                        ["sudo", "umount", drive_path], 
                        capture_output=True, 
                        text=True
                    )
                    
                    if result.returncode == 0:
                        self.status_text.append(f"✅ Successfully unmounted '{drive_name}'")
                        QMessageBox.information(self, "Drive Unmounted", f"Drive '{drive_name}' has been safely unmounted")
                        self.update_drive_list()
                    else:
                        error_msg = f"Failed to unmount drive: {result.stderr}"
                        self.status_text.append(f"❌ {error_msg}")
                        QMessageBox.critical(self, "Unmount Failed", error_msg)
                
            except Exception as e:
                error_msg = f"Error ejecting drive: {str(e)}"
                self.status_text.append(f"❌ {error_msg}")
                QMessageBox.critical(self, "Eject Error", error_msg)
    
    def check_system_requirements(self):
        """Check system requirements for downloading models"""
        self.status_text.append("🔍 Checking system requirements...")
        
        # Check Python version
        import sys
        python_version = sys.version_info
        self.status_text.append(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check required packages
        required_packages = ['huggingface_hub', 'transformers', 'torch']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                self.status_text.append(f"✅ {package} - OK")
            except ImportError:
                missing_packages.append(package)
                self.status_text.append(f"❌ {package} - Missing")
        
        if missing_packages:
            self.status_text.append(f"❌ Missing packages: {', '.join(missing_packages)}")
            QMessageBox.warning(self, "Missing Packages", f"Please install: {', '.join(missing_packages)}")
        else:
            self.status_text.append("✅ All required packages are installed")
        
        # Check available memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            self.status_text.append(f"System memory: {memory_gb:.1f}GB total, {memory.available / (1024**3):.1f}GB available")
            
            if memory_gb < 8:
                self.status_text.append("⚠️ Warning: Less than 8GB RAM may cause issues with large models")
        except:
            self.status_text.append("Could not check system memory")
        
        # Check internet connection
        try:
            import urllib.request
            urllib.request.urlopen('http://www.google.com', timeout=5)
            self.status_text.append("✅ Internet connection - OK")
        except:
            self.status_text.append("❌ Internet connection - Failed")
            QMessageBox.warning(self, "Network Error", "No internet connection detected")
        
        self.status_text.append("🔍 System check completed")
    
    def on_model_search_text_changed(self):
        """Handle text changes in the model search field"""
        query = self.model_name_input.text().strip()
        if len(query) >= 2:  # Only search if query is at least 2 characters
            self.perform_model_search(query)
        else:
            self.search_results_list.clear()
    
    def on_filter_changed(self):
        """Handle filter changes"""
        query = self.model_name_input.text().strip()
        category = self.category_combo.currentData()
        
        # If there's a query, search with that query and category
        if len(query) >= 2:
            self.perform_model_search(query)
        # If there's a category selected but no query, show popular models in that category
        elif category:
            self.perform_model_search("", show_popular=True)
        # If no category and no query, show all popular models
        else:
            self.load_popular_models()
    
    def perform_model_search(self, query: str, show_popular: bool = False):
        """Perform a model search with current filters"""
        # Cancel any existing search
        if self.search_thread and self.search_thread.isRunning():
            self.search_thread.terminate()
            self.search_thread.wait()
        
        # Get current filter values
        category = self.category_combo.currentData()
        sort_by = self.sort_combo.currentData()
        
        # Search parameters logged for debugging
        pass
        
        # Show loading message
        self.search_results_list.clear()
        self.search_results_list.addItem("Searching for models...")
        
        # Start new search (no size filtering for better performance)
        self.search_thread = ModelSearchThread(query, category, None, sort_by, show_popular)
        self.search_thread.search_results_signal.connect(self.on_search_results)
        self.search_thread.error_signal.connect(self.on_search_error)
        self.search_thread.start()
    
    def on_search_results(self, results: list):
        """Handle search results"""
        self.search_results_list.clear()
        
        if not results:
            self.search_results_list.addItem("No models found")
            return
        
        for result in results:
            # Format the display text
            downloads_str = f"{result['downloads']:,}" if result['downloads'] > 0 else "Unknown"
            size_str = result['size'] if result['size'] != "Unknown" else "Unknown"
            
            # Add compatibility warning for large models
            compatibility = "✅"
            if size_str != "Unknown":
                try:
                    size_gb = self.parse_size_to_gb(size_str)
                    if size_gb and size_gb > 10:
                        compatibility = "⚠️ Large"
                    if size_gb and size_gb > 20:
                        compatibility = "❌ Very Large"
                except:
                    pass
            
            # Check for problematic model names
            model_name = result['name'].lower()
            if any(keyword in model_name for keyword in ['unsloth', 'bnb', '4bit', '8bit', 'quantized']):
                compatibility = "❌ Special"
            
            # Check for authentication requirements
            if result.get('requires_auth', False):
                compatibility = "🔒 Auth"
            
            display_text = f"{compatibility} {result['name']} ({size_str}) - {downloads_str} downloads"
            
            item = QListWidgetItem(display_text)
            item.setData(Qt.ItemDataRole.UserRole, result)
            self.search_results_list.addItem(item)
    
    def on_search_error(self, error: str):
        """Handle search errors"""
        self.search_results_list.clear()
        self.search_results_list.addItem(f"Search error: {error}")
        self.status_text.append(f"Search error: {error}")
    
    def on_search_result_selected(self, item: QListWidgetItem):
        """Handle selection of a search result"""
        result_data = item.data(Qt.ItemDataRole.UserRole)
        if result_data and isinstance(result_data, dict):
            model_name = result_data['name']
            self.model_name_input.setText(model_name)
            self.status_text.append(f"Selected model: {model_name}")
    
    def choose_download_folder(self):
        # Get the currently selected drive path
        drive_path = self.drive_combo.currentData()
        if not drive_path:
            QMessageBox.warning(self, "Error", "Please select a drive first")
            return
        
        # Open folder dialog starting from the selected drive
        folder = QFileDialog.getExistingDirectory(
            self, 
            "Select Download Folder", 
            drive_path  # Start from the selected drive
        )
        if folder:
            self.selected_download_folder = folder
            self.status_text.append(f"Selected download folder: {folder}")
    
    def update_drive_list(self):
        """Update the list of available external drives"""
        self.drive_list.clear()
        drives = self.drive_manager.get_external_drives()
        
        for drive in drives:
            item_text = f"{drive['mountpoint']} ({drive['free_gb']:.1f}GB free / {drive['total_gb']:.1f}GB total)"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, drive)
            self.drive_list.addItem(item)
        
        # Update combo box
        self.drive_combo.clear()
        for drive in drives:
            self.drive_combo.addItem(f"{drive['mountpoint']} ({drive['free_gb']:.1f}GB free)", drive['mountpoint'])
        
        if drives:
            self.status_text.append(f"Found {len(drives)} external drive(s)")
            # Auto-scan for existing models
            self.scan_for_existing_models()
        else:
            self.status_text.append("No external drives found")
    
    def scan_for_existing_models(self):
        """Scan drives for existing models and update the display"""
        self.status_text.append("Scanning for existing models...")
        
        # Get available drives
        drives = self.drive_manager.get_external_drives()
        self.status_text.append(f"Found {len(drives)} external drive(s)")
        
        # Scan for models on drives
        models = self.drive_manager.update_installed_models_from_drives()
        
        # Also scan custom folder if one is selected
        if self.selected_download_folder:
            self.status_text.append(f"Scanning custom folder: {self.selected_download_folder}")
            custom_models = self.drive_manager.scan_custom_folder_for_models(self.selected_download_folder)
            for model in custom_models:
                models[model['name']] = model
        
        # Also scan any custom folders from config
        custom_folders = self.drive_manager.get_custom_folders()
        if custom_folders:
            self.status_text.append(f"Scanning {len(custom_folders)} custom folder(s) from config")
            for custom_folder in custom_folders:
                custom_models = self.drive_manager.scan_custom_folder_for_models(custom_folder)
                for model in custom_models:
                    models[model['name']] = model
        
        self.update_model_list()
        self.status_text.append(f"Found {len(models)} existing model(s)")
        
        # Show details of found models
        if models:
            for model_name, model_info in models.items():
                self.status_text.append(f"  - {model_name} at {model_info['path']}")
        else:
            self.status_text.append("No models found. Make sure you have models in:")
            self.status_text.append("  - /Volumes/[DriveName]/huggingface_models/ (on macOS)")
            self.status_text.append("  - [DriveLetter]:\\huggingface_models\\ (on Windows)")
            self.status_text.append("  - /media/[username]/[DriveName]/huggingface_models/ (on Linux)")
            self.status_text.append("  - Or use 'Choose Download Folder' to select a custom location")
    
    def update_model_list(self):
        """Update the list of installed models"""
        self.model_list.clear()
        models = self.drive_manager.get_installed_models()
        
        # Also include models from custom folder
        if self.selected_download_folder:
            custom_models = self.drive_manager.scan_custom_folder_for_models(self.selected_download_folder)
            for model in custom_models:
                models[model['name']] = model
        
        for model_name, model_info in models.items():
            # Calculate model size
            model_path = Path(model_info['path'])
            size_gb = 0.0
            if model_path.exists():
                try:
                    total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
                    size_gb = total_size / (1024**3)  # Convert to GB
                except:
                    size_gb = 0.0
            
            # Format size nicely
            if size_gb >= 1.0:
                size_str = f"{size_gb:.2f}GB"
            else:
                size_mb = size_gb * 1024
                size_str = f"{size_mb:.1f}MB"
            
            # Get model type from config if available
            model_type = "Unknown"
            config_path = model_path / "config.json"
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    model_type = config.get("model_type", "Unknown")
                except:
                    pass
            
            # Get download date if available
            download_date = model_info.get('download_date', 'Unknown')
            if download_date != 'Unknown':
                try:
                    # Format date nicely
                    from datetime import datetime
                    date_obj = datetime.fromisoformat(download_date.replace('Z', '+00:00'))
                    download_date = date_obj.strftime("%Y-%m-%d %H:%M")
                except:
                    pass
            
            # Create detailed item text
            item_text = f"📦 {model_name}\n"
            item_text += f"   💾 Size: {size_str}\n"
            item_text += f"   🏷️ Type: {model_type}\n"
            item_text += f"   📅 Downloaded: {download_date}\n"
            item_text += f"   📁 Path: {model_info['path']}"
            
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, model_name)
            self.model_list.addItem(item)
        
        self.status_text.append(f"Displaying {len(models)} model(s) with size information")
    
    def start_download(self):
        """Start downloading a model"""
        model_name = self.model_name_input.text().strip()
        # Remove size info if present (e.g., 'gpt2 (500MB)' -> 'gpt2')
        if ' (' in model_name:
            model_name = model_name.split(' (')[0].strip()
        if not model_name:
            QMessageBox.warning(self, "Error", "Please enter a model name")
            return

        # Check if a drive or folder is selected
        drive_path = self.drive_combo.currentData()
        if not drive_path and not self.selected_download_folder:
            QMessageBox.warning(self, "Error", "Please select an external drive or choose a download folder first")
            self.status_text.append("❌ No drive or folder selected for download")
            return
        
        # Validate the selected drive/folder
        if drive_path:
            if not Path(drive_path).exists():
                QMessageBox.warning(self, "Error", f"Selected drive '{drive_path}' no longer exists")
                self.status_text.append(f"❌ Drive {drive_path} not found")
                return
            self.status_text.append(f"✅ Using drive: {drive_path}")
        
        if self.selected_download_folder:
            if not Path(self.selected_download_folder).exists():
                QMessageBox.warning(self, "Error", f"Selected folder '{self.selected_download_folder}' no longer exists")
                self.status_text.append(f"❌ Folder {self.selected_download_folder} not found")
                return
            self.status_text.append(f"✅ Using custom folder: {self.selected_download_folder}")

        # Check available space before starting download
        available_space_gb = self.get_available_space_gb()
        if available_space_gb is not None:
            # Try to get model size from search results
            model_size_str = None
            for i in range(self.search_results_list.count()):
                item = self.search_results_list.item(i)
                if item.text().startswith(model_name):
                    # Extract size from item text (e.g., "gpt2 (500MB)")
                    if ' (' in item.text() and ')' in item.text():
                        size_part = item.text().split(' (')[1].split(')')[0]
                        model_size_str = size_part
                        break
            
            if model_size_str and model_size_str != "Unknown":
                model_size_gb = self.parse_size_to_gb(model_size_str)
                if model_size_gb and model_size_gb > available_space_gb:
                    QMessageBox.warning(self, "Insufficient Space", 
                                      f"Model size ({model_size_str}) exceeds available space ({available_space_gb:.1f}GB).\n"
                                      f"Please free up space or choose a different location.")
                    return

        # Determine the target path
        if self.selected_download_folder:
            target_path = str(Path(self.selected_download_folder) / model_name.replace("/", "_"))
            # Set cache directories for the custom folder
            self.set_cache_directories(str(Path(self.selected_download_folder).parent))
        else:
            drive_path = self.drive_combo.currentData()
            target_path = str(Path(drive_path) / "huggingface_models" / model_name.replace("/", "_"))
            # Set cache directories for the drive
            self.set_cache_directories(drive_path)

        # Disable download button
        self.download_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        try:
            # Start download in separate thread
            self.downloader = ModelDownloader(model_name, target_path)
            self.downloader.progress_signal.connect(self.update_status)
            self.downloader.finished_signal.connect(self.download_finished)
            self.downloader.progress_bar_signal.connect(self.update_progress_bar)
            self.downloader.download_stats_signal.connect(self.update_download_stats)
            self.downloader.start()
            
            self.status_text.append(f"🚀 Started download thread for {model_name}")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            error_msg = f"❌ Failed to start download: {str(e)}"
            self.status_text.append(error_msg)
            self.status_text.append(f"Error details: {error_details}")
            
            # Re-enable download button
            self.download_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            
            QMessageBox.critical(self, "Download Error", error_msg)
    
    def update_status(self, message: str):
        """Update status text"""
        self.status_text.append(message)
        self.status_text.ensureCursorVisible()
    
    def update_progress_bar(self, value):
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(value)
    
    def update_download_stats(self, stats: str):
        """Update download statistics in the status bar"""
        self.status_text.append(f"📊 {stats}")
    
    def on_model_selection_changed(self, current, previous):
        """Handle model selection in the list"""
        if current:
            # Enable the test and open folder buttons
            self.test_model_btn.setEnabled(True)
            self.open_model_folder_btn.setEnabled(True)
        else:
            # Disable the buttons when no model is selected
            self.test_model_btn.setEnabled(False)
            self.open_model_folder_btn.setEnabled(False)
    
    def test_selected_model(self):
        """Open the model testing dialog for the selected model"""
        current_item = self.model_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Error", "Please select a model to test")
            return
        
        model_name = current_item.data(Qt.ItemDataRole.UserRole)
        models = self.drive_manager.get_installed_models()
        
        if model_name in models:
            model_info = models[model_name]
            model_path = model_info['path']
            
            # Open the model testing dialog
            dialog = ModelTestDialog(model_name, model_path, self)
            dialog.exec()
        else:
            QMessageBox.warning(self, "Error", f"Model {model_name} not found in installed models")
    
    def open_selected_model_folder(self):
        """Open the folder containing the selected model"""
        current_item = self.model_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Error", "Please select a model to open")
            return
        
        model_name = current_item.data(Qt.ItemDataRole.UserRole)
        models = self.drive_manager.get_installed_models()
        
        if model_name in models:
            model_info = models[model_name]
            model_path = model_info['path']
            
            try:
                # Open the model directory in file explorer
                if sys.platform == "darwin":  # macOS
                    os.system(f"open '{model_path}'")
                elif sys.platform == "win32":  # Windows
                    os.system(f"explorer '{model_path}'")
                else:  # Linux
                    os.system(f"xdg-open '{model_path}'")
                
                self.status_text.append(f"Opened model folder: {model_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to open model folder: {str(e)}")
                self.status_text.append(f"Error opening model folder: {str(e)}")
        else:
            QMessageBox.warning(self, "Error", f"Model {model_name} not found in installed models")
    
    def download_finished(self, success: bool, message: str):
        """Handle download completion"""
        self.download_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if success:
            # Get the model name from the input field
            model_name = self.model_name_input.text().strip()
            if ' (' in model_name:
                model_name = model_name.split(' (')[0].strip()
            
            # Save download date to the model info
            if model_name:
                models = self.drive_manager.get_installed_models()
                if model_name in models:
                    # Add download date
                    from datetime import datetime
                    models[model_name]['download_date'] = datetime.now().isoformat()
                    self.drive_manager.save_config()
            
            # Refresh the model list to include the newly downloaded model
            self.update_model_list()
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Error", message)
    
    def remove_selected_model(self):
        """Remove the selected model"""
        current_item = self.model_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Error", "Please select a model to remove")
            return
        
        model_name = current_item.data(Qt.ItemDataRole.UserRole)
        models = self.drive_manager.get_installed_models()
        
        if model_name not in models:
            QMessageBox.warning(self, "Error", f"Model {model_name} not found in installed models")
            return
        
        model_path = models[model_name]['path']
        
        # Check if model directory exists
        if not Path(model_path).exists():
            reply = QMessageBox.question(self, "Model Not Found", 
                                       f"Model folder '{model_path}' doesn't exist.\n\n"
                                       f"Remove '{model_name}' from the installed models list?",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            
            if reply == QMessageBox.StandardButton.Yes:
                if self.drive_manager.remove_model(model_name):
                    self.update_model_list()
                    self.status_text.append(f"Removed {model_name} from installed models list")
                    QMessageBox.information(self, "Success", f"Removed {model_name} from installed models list")
                else:
                    QMessageBox.critical(self, "Error", f"Failed to remove {model_name} from list")
            return
        
        # Model directory exists, ask for confirmation
        reply = QMessageBox.question(self, "Confirm Removal", 
                                   f"Are you sure you want to remove '{model_name}'?\n\n"
                                   f"This will:\n"
                                   f"• Delete the model folder: {model_path}\n"
                                   f"• Remove it from the installed models list\n\n"
                                   f"This action cannot be undone!",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                if self.drive_manager.remove_model(model_name):
                    self.update_model_list()
                    self.status_text.append(f"✅ Successfully removed model: {model_name}")
                    QMessageBox.information(self, "Success", f"Successfully removed model: {model_name}")
                else:
                    # This shouldn't happen with our improved remove_model method
                    QMessageBox.critical(self, "Error", f"Failed to remove model: {model_name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error removing model: {str(e)}")
                self.status_text.append(f"❌ Error removing model: {str(e)}")
    
    def open_models_directory(self):
        """Open the models directory in file explorer"""
        try:
            # If a custom folder is selected, open that
            if self.selected_download_folder:
                models_dir = Path(self.selected_download_folder)
                if models_dir.exists():
                    # Open directory in file explorer
                    if sys.platform == "darwin":  # macOS
                        os.system(f"open '{str(models_dir)}'")
                    elif sys.platform == "win32":  # Windows
                        os.system(f"explorer '{str(models_dir)}'")
                    else:  # Linux
                        os.system(f"xdg-open '{str(models_dir)}'")
                    self.status_text.append(f"Opened custom folder: {models_dir}")
                else:
                    QMessageBox.warning(self, "Error", f"Folder does not exist: {models_dir}")
            # Otherwise, open the default models directory on the selected drive
            elif self.drive_combo.count() > 0:
                drive_path = self.drive_combo.currentData()
                if drive_path:
                    models_dir = Path(drive_path) / "huggingface_models"
                    models_dir.mkdir(exist_ok=True)
                    
                    # Open directory in file explorer
                    if sys.platform == "darwin":  # macOS
                        os.system(f"open '{str(models_dir)}'")
                    elif sys.platform == "win32":  # Windows
                        os.system(f"explorer '{str(models_dir)}'")
                    else:  # Linux
                        os.system(f"xdg-open '{str(models_dir)}'")
                    self.status_text.append(f"Opened models directory: {models_dir}")
                else:
                    QMessageBox.warning(self, "Error", "No drive path available")
            else:
                QMessageBox.warning(self, "Error", "No drive selected and no custom folder set")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open directory: {str(e)}")
            self.status_text.append(f"Error opening directory: {str(e)}")
    
    def export_configuration(self):
        """Export current configuration to a file"""
        # Prepare configuration data
        config_data = {
            "installed_models": self.drive_manager.get_installed_models(),
            "custom_folders": self.drive_manager.get_custom_folders(),
            "selected_download_folder": self.selected_download_folder,
            "drives": self.drive_manager.get_external_drives(),
            "export_date": str(Path().cwd()),
            "app_version": "1.0.0"
        }
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Configuration", "huggingdrive_config.json", "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(config_data, f, indent=2)
                QMessageBox.information(self, "Success", f"Configuration exported to {file_path}")
                self.status_text.append(f"Configuration exported to: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export configuration: {str(e)}")
                self.status_text.append(f"Export failed: {str(e)}")

    def load_popular_models(self):
        """Load popular models by default"""
        self.status_text.append("Loading popular models...")
        self.perform_model_search("", show_popular=True)
    
    def browse_all_models(self):
        """Open Hugging Face Hub models page in default browser"""
        try:
            import webbrowser
            url = "https://huggingface.co/models"
            webbrowser.open(url)
            self.status_text.append("Opened Hugging Face Hub models page in browser")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open browser: {str(e)}")
            self.status_text.append(f"Error opening browser: {str(e)}")
    
    def get_available_space_gb(self) -> float:
        """Get available space on the selected drive in GB"""
        if self.selected_download_folder:
            # Use custom folder's drive
            drive_path = Path(self.selected_download_folder).parent
        elif self.drive_combo.count() > 0:
            drive_path = self.drive_combo.currentData()
        else:
            return None
        
        try:
            usage = psutil.disk_usage(drive_path)
            return usage.free / (1024**3)  # Convert to GB
        except:
            return None
    
    def update_memory_display(self):
        """Update the memory usage display in the top status bar"""
        try:
            import psutil
            import os
            
            # Get system memory info
            memory = psutil.virtual_memory()
            total_ram_gb = memory.total / (1024**3)
            used_ram_gb = memory.used / (1024**3)
            available_ram_gb = memory.available / (1024**3)
            ram_percent = memory.percent
            
            # Get current process memory usage
            process = psutil.Process(os.getpid())
            app_memory_mb = process.memory_info().rss / (1024 * 1024)  # Convert to MB
            
            # Update RAM usage label
            ram_color = "green" if ram_percent < 70 else "orange" if ram_percent < 90 else "red"
            self.ram_usage_label.setText(f"🧠 RAM: {used_ram_gb:.1f}GB/{total_ram_gb:.1f}GB ({ram_percent:.0f}%)")
            self.ram_usage_label.setStyleSheet(f"background-color: {ram_color}; color: white; padding: 5px; border-radius: 3px; font-weight: bold;")
            
            # Update app memory label
            app_color = "green" if app_memory_mb < 500 else "orange" if app_memory_mb < 1000 else "red"
            self.app_memory_label.setText(f"📱 App: {app_memory_mb:.0f}MB")
            self.app_memory_label.setStyleSheet(f"background-color: {app_color}; color: white; padding: 5px; border-radius: 3px; font-weight: bold;")
            
        except Exception as e:
            self.ram_usage_label.setText("🧠 RAM: Error")
            self.app_memory_label.setText("📱 App: Error")
            self.ram_usage_label.setStyleSheet("background-color: red; color: white; padding: 5px; border-radius: 3px; font-weight: bold;")
            self.app_memory_label.setStyleSheet("background-color: red; color: white; padding: 5px; border-radius: 3px; font-weight: bold;")
    
    def parse_size_to_gb(self, size_str: str) -> float:
        """Parse size string to GB for space checking"""
        if size_str == "Unknown":
            return None
        
        try:
            size_str = size_str.upper()
            if "GB" in size_str:
                return float(size_str.replace("GB", ""))
            elif "MB" in size_str:
                return float(size_str.replace("MB", "")) / 1024
            elif "KB" in size_str:
                return float(size_str.replace("KB", "")) / (1024 * 1024)
            elif "B" in size_str:
                return float(size_str.replace("B", "")) / (1024 * 1024 * 1024)
            else:
                return None
        except:
            return None


def main():
    """Main function to run the application"""
    app = QApplication(sys.argv)
    app.setApplicationName("HuggingDrive")
    app.setApplicationVersion("1.0.0")
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window using the modular structure
    from huggingdrive.gui import HuggingDriveGUI
    window = HuggingDriveGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 