"""
GUI Main Window - HuggingDriveGUI class and related methods
"""

import sys
import os
import json
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional

import psutil
import PyQt6
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, QTextEdit, QComboBox, QProgressBar, QFileDialog, QMessageBox, QListWidget, QListWidgetItem, QGroupBox, QCompleter, QTextBrowser, QTabWidget, QSplitter, QDialog, QGridLayout, QTableWidget, QTableWidgetItem)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QFontMetrics, QColor
from .drive_manager import DriveManager
from .downloader import ModelDownloader
from .search import ModelSearchThread
from .loader import ModelLoadThread
from .test_dialog import ModelTestDialog
from .auth_manager import HuggingFaceAuthManager, LoginDialog
from .gguf_converter import GGUFConverter
from .quantization_dialog import QuantizationDialog
import requests
import random
import socket
import time
from urllib.parse import urlparse, unquote


# Simplified lock mechanism - locks stay visible even when logged in


class HuggingDriveGUI(QMainWindow):
    """Main GUI window for HuggingDrive"""
    
    # Define signals at class level
    progress_signal = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.drive_manager = DriveManager()
        self.auth_manager = HuggingFaceAuthManager()  # Add authentication manager
        self.downloader = None
        self.selected_download_folder = None  # Store the user-selected folder
        self.search_thread = None  # For live search
        self.current_cache_dir = None  # Current cache directory
        self.api_server_process = None
        self.api_server_port = None
        self.currently_downloading = None  # Track currently downloading model
        self.user_token = None
        
        # Initialize UI elements that are referenced across methods
        self.gguf_file_combo = QComboBox()
        self.gguf_file_combo_label = QLabel("Available GGUF Files:")
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_progress_animation)
        self.animation_offset = 0
        
        # Register cleanup handlers
        import atexit
        import signal
        from .downloader import ModelDownloader
        
        def cleanup_handler(*args):
            """Clean up resources on exit"""
            try:
                # Stop any active downloads
                if self.downloader:
                    self.downloader.stop()
                    self.downloader = None
                
                # Clean up all downloaders
                ModelDownloader.cleanup_resources()
                
                # Stop any running API server
                if self.api_server_process:
                    self.api_server_process.terminate()
                    self.api_server_process = None
                
                # Stop animation timer
                if self.animation_timer.isActive():
                    self.animation_timer.stop()
                
                # Stop all Gradio interfaces
                try:
                    from .gradio_manager import GradioInterfaceManager
                    gradio_manager = GradioInterfaceManager()
                    gradio_manager.stop_all_interfaces()
                except Exception as e:
                    print(f"Warning: Could not stop Gradio interfaces: {e}")
                
            except:
                pass  # Ignore cleanup errors
        
        # Register cleanup handlers
        atexit.register(cleanup_handler)
        signal.signal(signal.SIGTERM, cleanup_handler)
        signal.signal(signal.SIGINT, cleanup_handler)
        
        self.init_ui()

    def closeEvent(self, event):
        """Handle application close event"""
        try:
            # Show a brief message that cleanup is happening
            print("Cleaning up resources before closing...")

            # Stop any active downloads with timeout
            if self.downloader:
                try:
                    self.downloader.stop()
                    # Wait for downloader to stop with timeout
                    if self.downloader.isRunning():
                        self.downloader.wait(2000)
                except Exception as e:
                    print(f"Warning: Error stopping downloader: {e}")
                finally:
                    self.downloader = None

            # Clean up all downloaders with timeout
            try:
                from .downloader import ModelDownloader
                ModelDownloader.cleanup_resources()
            except Exception as e:
                print(f"Warning: Error cleaning up downloaders: {e}")

            # Stop any running API server with timeout
            if self.api_server_process:
                try:
                    self.api_server_process.terminate()
                    # Wait for process to terminate with timeout
                    self.api_server_process.wait(3000)
                    if self.api_server_process.poll() is None:
                        # Force kill if still running
                        self.api_server_process.kill()
                except Exception as e:
                    print(f"Warning: Error stopping API server: {e}")
                finally:
                    self.api_server_process = None

            # Stop animation timer
            if self.animation_timer.isActive():
                self.animation_timer.stop()

            # Stop all Gradio interfaces with timeout
            try:
                from .gradio_manager import GradioInterfaceManager
                gradio_manager = GradioInterfaceManager()
                gradio_manager.stop_all_interfaces()
            except Exception as e:
                print(f"Warning: Could not stop Gradio interfaces: {e}")

            # Force garbage collection to free memory
            try:
                import gc
                gc.collect()
                # Clear CUDA cache if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass
            except Exception as e:
                print(f"Warning: Error during memory cleanup: {e}")

            print("Cleanup completed")

        except Exception as e:
            print(f"Error during closeEvent: {e}")

        # Always accept the close event, even if cleanup fails
        event.accept()

    def init_ui(self):
        self.setWindowTitle("HuggingDrive - External Drive Model Manager")
        self.setGeometry(100, 100, 1200, 800)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        status_bar_layout = QHBoxLayout()
        self.ram_usage_label = QLabel("🧠 RAM: Loading...")
        self.ram_usage_label.setStyleSheet("background-color: #f0f0f0; padding: 5px; border-radius: 3px; font-weight: bold;")
        status_bar_layout.addWidget(self.ram_usage_label)
        self.app_memory_label = QLabel("📱 App: Loading...")
        self.app_memory_label.setStyleSheet("background-color: #e8f4fd; padding: 5px; border-radius: 3px; font-weight: bold;")
        status_bar_layout.addWidget(self.app_memory_label)
        status_bar_layout.addStretch()
        refresh_memory_btn = QPushButton("🔄 Refresh")
        refresh_memory_btn.clicked.connect(self.update_memory_display)
        refresh_memory_btn.setStyleSheet("padding: 5px;")
        # Add authentication status and buttons
        self.auth_status_label = QLabel("🔓 Not logged in")
        self.auth_status_label.setStyleSheet("background-color: #fff3cd; padding: 5px; border-radius: 3px; font-weight: bold; color: #856404;")
        status_bar_layout.addWidget(self.auth_status_label)
        
        self.login_btn = QPushButton("🔑 Login to HF")
        self.login_btn.clicked.connect(self.show_login_dialog)
        self.login_btn.setStyleSheet("padding: 5px; background-color: #4CAF50; color: white; border: none; border-radius: 3px;")
        status_bar_layout.addWidget(self.login_btn)
        
        self.logout_btn = QPushButton("🚪 Logout")
        self.logout_btn.clicked.connect(self.logout_huggingface)
        self.logout_btn.setVisible(False)
        self.logout_btn.setStyleSheet("padding: 5px; background-color: #f44336; color: white; border: none; border-radius: 3px;")
        status_bar_layout.addWidget(self.logout_btn)
        
        # Add tip label in small font
        tip_label = QLabel("Tip: Login to access restricted models")
        tip_label.setStyleSheet("font-size: 10px; color: #888; margin-left: 10px;")
        status_bar_layout.addWidget(tip_label)
        status_bar_layout.addWidget(refresh_memory_btn)
        main_layout.addLayout(status_bar_layout)
        content_layout = QHBoxLayout()
        left_panel = self.create_left_panel()
        content_layout.addWidget(left_panel, 1)
        right_panel = self.create_right_panel()
        content_layout.addWidget(right_panel, 1)
        main_layout.addLayout(content_layout)
        self.update_drive_list()
        self.scan_for_existing_models()
        self.load_popular_models()
        self.update_memory_display()
        self.update_auth_status()  # Update authentication status
        
        # Set user_token if already authenticated
        if self.auth_manager.is_authenticated():
            self.user_token = self.auth_manager.get_token()
        from PyQt6.QtCore import QTimer
        self.memory_timer = QTimer()
        self.memory_timer.timeout.connect(self.update_memory_display)
        self.memory_timer.start(10000)  # Update every 10 seconds instead of 5 to reduce UI interference

    def create_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
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
        models_group = QGroupBox("Installed Models")
        models_layout = QVBoxLayout()
        models_group.setLayout(models_layout)
        self.model_list = QListWidget()
        self.model_list.currentItemChanged.connect(self.on_model_selection_changed)
        self.remove_model_btn = QPushButton("Remove Selected Model")
        self.remove_model_btn.clicked.connect(self.remove_selected_model)
        model_actions_layout = QHBoxLayout()
        self.test_model_btn = QPushButton("Test Model")
        self.test_model_btn.clicked.connect(self.test_selected_model)
        self.test_model_btn.setEnabled(False)
        self.open_model_folder_btn = QPushButton("Open Model Folder")
        self.open_model_folder_btn.clicked.connect(self.open_selected_model_folder)
        self.open_model_folder_btn.setEnabled(False)
        model_actions_layout.addWidget(self.test_model_btn)
        model_actions_layout.addWidget(self.open_model_folder_btn)
        
        # Add GGUF conversion button
        self.convert_to_gguf_btn = QPushButton("Convert to GGUF")
        self.convert_to_gguf_btn.clicked.connect(self.convert_selected_model_to_gguf)
        self.convert_to_gguf_btn.setEnabled(False)
        model_actions_layout.addWidget(self.convert_to_gguf_btn)
        
        # Add Start API button
        self.start_api_btn = QPushButton("Start API Server (Test mode)")
        self.start_api_btn.clicked.connect(self.toggle_api_server)
        self.start_api_btn.setEnabled(False)
        model_actions_layout.addWidget(self.start_api_btn)
        
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
        self.search_results_list.setMaximumHeight(300)
        self.search_results_list.itemClicked.connect(self.on_search_result_selected)
        download_layout.addWidget(self.search_results_list)
        
        # Model info label (replaces GGUF file picker)
        self.model_info_label = QLabel("Select a model from search results to download")
        self.model_info_label.setStyleSheet("color: #666; font-style: italic;")
        download_layout.addWidget(self.model_info_label)
        
        # Add Paste URL button
        self.paste_url_btn = QPushButton("Paste Model URL...")
        self.paste_url_btn.clicked.connect(self.show_paste_gguf_url_dialog)
        download_layout.addWidget(self.paste_url_btn)
        
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
        
        # Download buttons
        download_buttons_layout = QHBoxLayout()
        
        # Check Access button
        self.check_access_btn = QPushButton("🔍 Check Access")
        self.check_access_btn.clicked.connect(self.check_model_access)
        self.check_access_btn.setToolTip("Check if you have access to the selected model")
        download_buttons_layout.addWidget(self.check_access_btn)
        
        # Download button
        self.download_btn = QPushButton("Download Model")
        self.download_btn.clicked.connect(self.start_download)
        download_buttons_layout.addWidget(self.download_btn)
        
        download_layout.addLayout(download_buttons_layout)
        
        # Add Stop Download button
        self.stop_download_btn = QPushButton("Stop Download")
        self.stop_download_btn.setEnabled(False)
        self.stop_download_btn.clicked.connect(self.stop_download)
        download_layout.addWidget(self.stop_download_btn)
        
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
        
        # Status text area
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(200)
        status_layout.addWidget(self.status_text)
        
        layout.addWidget(status_group)
        
        return panel

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
        
        if len(query) >= 2:
            self.perform_model_search(query)
        else:
            self.load_popular_models()
    
    def perform_model_search(self, query: str, show_popular: bool = False):
        """Perform a model search with current filters"""
        # Cancel any existing search
        if self.search_thread and self.search_thread.isRunning():
            self.search_thread.terminate()
            self.search_thread.wait()
        
        # Get current filter values
        sort_by = self.sort_combo.currentData()
        
        # Search parameters logged for debugging
        pass
        
        # Reset model selection when starting new search
        self.reset_model_selection()
        
        # Show loading message
        self.search_results_list.clear()
        self.search_results_list.addItem("Searching for models...")
        
        # Start new search (no size filtering for better performance)
        self.search_thread = ModelSearchThread(query, limit=10, category='', sort_by=sort_by)
        self.search_thread.results_signal.connect(self.on_search_results)
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
            likes_str = f"{result['likes']:,}" if result['likes'] > 0 else "Unknown"
            
            # Add compatibility warning for large models (keep for now, but don't show size)
            compatibility = "✅"
            model_name = result['name'].lower()
            if any(keyword in model_name for keyword in ['unsloth', 'bnb', '4bit', '8bit', 'quantized']):
                compatibility = "❌ Special"
            if result.get('requires_auth', False):
                compatibility = "🔒 Auth"
            
            # Show name, downloads, and likes
            display_text = f"{compatibility} {result['name']} - {downloads_str} downloads • {likes_str} likes"
            
            item = QListWidgetItem(display_text)
            item.setData(Qt.ItemDataRole.UserRole, result)
            
            # Add tooltip with detailed information
            tooltip_text = f"<b>Model:</b> {result['name']}<br>"
            tooltip_text += f"<b>Downloads:</b> {downloads_str}<br>"
            tooltip_text += f"<b>Likes:</b> {likes_str}<br>"
            tooltip_text += f"<b>Size:</b> {result.get('size', 'Unknown')}<br>"
            
            # Add tags if available
            if result.get('tags'):
                tags_str = ', '.join(result['tags'][:5])  # Show first 5 tags
                tooltip_text += f"<b>Tags:</b> {tags_str}<br>"
            
            # Add timestamp if available
            if result.get('timestamp'):
                tooltip_text += f"<b>Last Updated:</b> {result['timestamp']}"
            
            item.setToolTip(tooltip_text)
            self.search_results_list.addItem(item)
    
    def on_search_error(self, error: str):
        """Handle search errors"""
        self.search_results_list.clear()
        self.search_results_list.addItem(f"Search error: {error}")
        self.status_text.append(f"Search error: {error}")
    
    def is_model_locked(self, model_name: str) -> bool:
        """Check if a model requires authentication based on its name"""
        model_lower = model_name.lower()
        
        # Models that typically require authentication
        auth_indicators = [
            'meta/llama', 'meta-llama', 'llama-2', 'llama-3',  # Meta models
            'microsoft/phi', 'microsoft-phi',  # Microsoft models
            'google/gemma', 'google-gemma',  # Google models
            'anthropic/claude', 'anthropic-claude',  # Anthropic models
            'openai/gpt', 'openai-gpt',  # OpenAI models
            'cohere/command', 'cohere-command',  # Cohere models
            'mistralai/mistral', 'mistralai-mistral',  # Mistral AI models
            'deepseek-ai', 'deepseekai',  # DeepSeek models
            'qwen', 'alibaba',  # Alibaba models
            'baichuan', 'baichuan-inc',  # Baichuan models
            'zhipu', 'glm',  # Zhipu models
        ]
        
        return any(indicator in model_lower for indicator in auth_indicators)

    def on_search_result_selected(self, item: QListWidgetItem):
        """Handle selection of a search result"""
        result_data = item.data(Qt.ItemDataRole.UserRole)
        if result_data and isinstance(result_data, dict):
            model_name = result_data['name']
            
            # Temporarily disconnect the text change handler to prevent search refresh
            self.model_name_input.textChanged.disconnect(self.on_model_search_text_changed)
            
            # Update the model selection
            self._update_model_selection(model_name)
            
            # Reconnect the text change handler after a short delay
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(100, lambda: self.model_name_input.textChanged.connect(self.on_model_search_text_changed))
    
    def _update_model_selection(self, model_name: str):
        """Update the model selection (called with a small delay to prevent UI interference)"""
        self.model_name_input.setText(model_name)
        self.status_text.append(f"Selected model: {model_name}")
        
        # Update the model info label
        self.model_info_label.setText(f"Selected: {model_name}")
        self.model_info_label.setStyleSheet("color: #2c3e50; font-weight: bold;")
        
        # Show helpful information about the model
        self.status_text.append("Model selected for download.")
        self.status_text.append("This will download the entire model (config, weights, tokenizer, etc.).")
        self.status_text.append("You can convert it to GGUF format later using the 'Convert to GGUF' button if needed.")
    
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
    
    def start_download(self):
        """Start downloading a model"""
        model_name = self.model_name_input.text().strip()
        
        # Check if this is a direct URL
        if model_name.startswith(('http://', 'https://')):
            # Direct URL download - support any file type
            filename = os.path.basename(model_name.split('?')[0])
            download_url = model_name
            model_size = "Unknown (direct URL)"
            
        else:
            # Repository name from search results
            if '/' not in model_name:
                QMessageBox.warning(self, "Error", "Please select a model from the search results or enter a valid model URL")
                return
            
            # Check if this is a GGUF repository with multiple quantization options
            if self.is_gguf_repository(model_name):
                self.status_text.append("🔍 GGUF repository detected! Checking quantization options...")
                dialog = GGUFQuantizationDialog(model_name, self)
                result = dialog.exec()
                if result == 1 and dialog.selected_file:  # 1 = QDialog.DialogCode.Accepted
                    # User selected a specific GGUF file
                    selected_file = dialog.selected_file
                    self.status_text.append(f"✅ Selected: {selected_file}")
                    
                    # Download the specific GGUF file
                    self.download_specific_gguf_file(model_name, selected_file)
                    return
                else:
                    self.status_text.append("❌ GGUF download cancelled by user")
                    return
            
            # Get actual model size before downloading
            self.status_text.append("📏 Fetching model size...")
            model_size = self.get_model_size(model_name)
            
            # Show size confirmation dialog
            if model_size != "Unknown":
                # Custom dialog with link to Hugging Face
                from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
                from PyQt6.QtCore import QUrl
                from PyQt6.QtGui import QDesktopServices
                
                class KnownSizeDialog(QDialog):
                    def __init__(self, parent, model_id, model_size):
                        super().__init__(parent)
                        self.setWindowTitle("Confirm Download")
                        self.setModal(True)
                        self.resize(450, 200)
                        layout = QVBoxLayout()
                        self.setLayout(layout)
                        
                        info = QLabel(f"<b>Model:</b> {model_id}<br><b>Size:</b> {model_size}")
                        layout.addWidget(info)
                        
                        # Add link to Hugging Face page
                        link_label = QLabel(f'<a href="https://huggingface.co/{model_id}">🔗 Open on Hugging Face Hub</a>')
                        link_label.setOpenExternalLinks(True)
                        link_label.setStyleSheet("color: #007bff; text-decoration: underline;")
                        layout.addWidget(link_label)
                        
                        layout.addWidget(QLabel("Do you want to download this model?"))
                        
                        btn_layout = QHBoxLayout()
                        self.yes_btn = QPushButton("Download")
                        self.yes_btn.clicked.connect(self.accept)
                        btn_layout.addWidget(self.yes_btn)
                        self.no_btn = QPushButton("Cancel")
                        self.no_btn.clicked.connect(self.reject)
                        btn_layout.addWidget(self.no_btn)
                        layout.addLayout(btn_layout)
                
                dialog = KnownSizeDialog(self, model_name, model_size)
                reply = dialog.exec()
                if reply != QDialog.DialogCode.Accepted:
                    self.status_text.append("❌ Download cancelled by user")
                    return
            else:
                # Custom dialog for unknown size
                from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QTextEdit, QHBoxLayout
                class UnknownSizeDialog(QDialog):
                    def __init__(self, parent, model_id, get_file_list_func):
                        super().__init__(parent)
                        self.setWindowTitle("Model Size Unknown")
                        self.setModal(True)
                        self.resize(500, 400)
                        layout = QVBoxLayout()
                        self.setLayout(layout)
                        info = QLabel("<b>Model:</b> {}<br><b>Size:</b> <span style='color:red;'>Unknown</span>".format(model_id))
                        layout.addWidget(info)
                        
                        # Add link to Hugging Face page
                        from PyQt6.QtCore import QUrl
                        from PyQt6.QtGui import QDesktopServices
                        link_label = QLabel(f'<a href="https://huggingface.co/{model_id}">🔗 Open on Hugging Face Hub</a>')
                        link_label.setOpenExternalLinks(True)
                        link_label.setStyleSheet("color: #007bff; text-decoration: underline;")
                        layout.addWidget(link_label)
                        
                        warn = QLabel("<span style='color:orange;'>The size of this model could not be determined automatically. This is common for private, gated, or complex models. You can still download, or click 'Check Size / Show Files' to see the file list.</span>")
                        warn.setWordWrap(True)
                        layout.addWidget(warn)
                        self.file_list = QTextEdit()
                        self.file_list.setReadOnly(True)
                        self.file_list.setVisible(False)
                        layout.addWidget(self.file_list)
                        btn_layout = QHBoxLayout()
                        self.check_btn = QPushButton("Check Size / Show Files")
                        self.check_btn.clicked.connect(lambda: self.show_files(model_id, get_file_list_func))
                        btn_layout.addWidget(self.check_btn)
                        self.yes_btn = QPushButton("Download Anyway")
                        self.yes_btn.clicked.connect(self.accept)
                        btn_layout.addWidget(self.yes_btn)
                        self.no_btn = QPushButton("Cancel")
                        self.no_btn.clicked.connect(self.reject)
                        btn_layout.addWidget(self.no_btn)
                        layout.addLayout(btn_layout)
                        
                        # Store parameters
                        self.model_id = model_id
                        self.get_file_list_func = get_file_list_func
                        
                    def show_files(self, model_id, get_file_list_func):
                        self.file_list.setVisible(True)
                        self.file_list.setPlainText("Fetching file list...")
                        self.check_btn.setEnabled(False)
                        self.check_btn.setText("Fetching...")
                        
                        # Use QTimer to run the fetch asynchronously
                        from PyQt6.QtCore import QTimer
                        def fetch_files():
                            try:
                                files = get_file_list_func(model_id)
                                if not files:
                                    self.file_list.setPlainText("Could not fetch file list or no files found.")
                                else:
                                    lines = []
                                    total_size = 0
                                    for f in files:
                                        lines.append(f"{f['name']}\t{f['size']}")
                                        # Try to calculate total size
                                        if f['size'] != 'Unknown':
                                            try:
                                                size_str = f['size'].upper()
                                                if 'GB' in size_str:
                                                    total_size += float(size_str.replace('GB', '')) * 1024 * 1024 * 1024
                                                elif 'MB' in size_str:
                                                    total_size += float(size_str.replace('MB', '')) * 1024 * 1024
                                                elif 'KB' in size_str:
                                                    total_size += float(size_str.replace('KB', '')) * 1024
                                            except:
                                                pass
                                    
                                    if total_size > 0:
                                        # Format size manually
                                        if total_size >= 1024**3:
                                            total_str = f"{total_size/(1024**3):.1f}GB"
                                        elif total_size >= 1024**2:
                                            total_str = f"{total_size/(1024**2):.1f}MB"
                                        elif total_size >= 1024:
                                            total_str = f"{total_size/1024:.1f}KB"
                                        else:
                                            total_str = f"{total_size:.1f}B"
                                        lines.insert(0, f"Total estimated size: {total_str}")
                                        lines.insert(1, "-" * 50)
                                    
                                    self.file_list.setPlainText("\n".join(lines))
                            except Exception as e:
                                self.file_list.setPlainText(f"Error fetching files: {str(e)}")
                            finally:
                                self.check_btn.setEnabled(True)
                                self.check_btn.setText("Check Size / Show Files")
                        
                        # Run after a short delay to allow UI to update
                        QTimer.singleShot(100, fetch_files)
                def get_file_list(model_id):
                    try:
                        from huggingface_hub import model_info
                        token = self.auth_manager.get_token() if self.auth_manager.is_authenticated() else None
                        info = model_info(model_id, token=token)
                        files = []
                        if hasattr(info, 'siblings') and info.siblings:
                            for sibling in info.siblings:
                                name = getattr(sibling, 'rfilename', getattr(sibling, 'name', 'unknown'))
                                size = getattr(sibling, 'size', None)
                                size_str = self.format_size(size) if size else 'Unknown'
                                files.append({'name': name, 'size': size_str})
                        return files
                    except Exception as e:
                        self.status_text.append(f"⚠️ Could not fetch file list: {str(e)}")
                        return []
                dialog = UnknownSizeDialog(self, model_name, get_file_list)
                result = dialog.exec()
                if result != QDialog.DialogCode.Accepted:
                    self.status_text.append("❌ Download cancelled by user")
                    return
            
            # Download the entire model repository (regular format)
            download_url = model_name  # Pass the repository name directly
            filename = model_name.split('/')[-1]  # Use the model name as filename
        
        # Determine the target path
        if self.selected_download_folder:
            target_path = str(Path(self.selected_download_folder))
        else:
            drive_path = self.drive_combo.currentData()
            if not drive_path:
                QMessageBox.warning(self, "Error", "Please select a drive or choose a download folder")
                return
            target_path = str(Path(drive_path) / "huggingface_models")
        
        # Create the target directory if it doesn't exist
        try:
            os.makedirs(target_path, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not create download directory: {str(e)}")
            return
        
        # Create model directory
        if filename.endswith('.gguf'):
            # For GGUF files, create a directory without the .gguf extension
            model_dir = os.path.join(target_path, filename.replace('.gguf', ''))
        else:
            # For regular models, use the model name as directory
            model_dir = os.path.join(target_path, filename)
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Store currently downloading model info
        self.currently_downloading = {
            'name': filename.replace('.gguf', '') if filename.endswith('.gguf') else filename,
            'path': model_dir,
            'size': model_size,
            'source': 'Direct Download' if model_name.startswith(('http://', 'https://')) else 'Repository Download'
        }
        
        # Update model list to show downloading model
        self.update_model_list()
        
        # Start the download
        try:
            self.downloader = ModelDownloader(download_url, model_dir)
            self.downloader.progress_signal.connect(self.update_status)
            self.downloader.finished_signal.connect(self.download_finished)
            self.downloader.progress_bar_signal.connect(self.update_progress_bar)
            self.downloader.download_stats_signal.connect(self.update_download_stats)
            self.downloader.start()
            
            self.download_btn.setEnabled(False)
            self.stop_download_btn.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            self.status_text.append(f"🚀 Starting download of {filename}")
            
            # Start progress animation with proper timing
            if not self.animation_timer.isActive():
                self.animation_timer.start(50)  # Update every 50ms for smooth animation
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start download: {str(e)}")
            self.status_text.append(f"Error starting download: {str(e)}")

    def update_status(self, message: str):
        """Update status text"""
        self.status_text.append(message)
        self.status_text.ensureCursorVisible()
    
    def update_progress_bar(self, value):
        """Update the progress bar with download progress"""
        # Ensure progress bar is visible and properly configured
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(value)
        
        # Update the progress text
        self.progress_bar.setFormat(f"Download progress: {value}%")
        
        # Ensure animation is running (don't restart if already active)
        if not self.animation_timer.isActive():
            self.animation_timer.start(50)  # Update every 50ms for smooth animation
        
        # Force update the UI
        self.progress_bar.repaint()
    
    def update_progress_animation(self):
        """Update the progress bar animation with moving white overlay"""
        # Check if progress bar is visible and animation should continue
        if not self.progress_bar.isVisible():
            self.animation_timer.stop()
            return
            
        # Update animation offset for smooth movement
        self.animation_offset = (self.animation_offset + 3) % 100  # Slower movement for better visibility
        
        # Create base animated gradient
        animated_style = f"""
            QProgressBar {{
                border: 2px solid #e1e5e9;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                color: #2c3e50;
                background-color: #f8f9fa;
                height: 25px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #007bff, stop:0.5 #0056b3, stop:1 #004085);
                border-radius: 6px;
                margin: 2px;
            }}
        """
        
        # Add animated white overlay for shimmer effect
        overlay_style = f"""
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #007bff, 
                    stop:{max(0, (self.animation_offset-15)/100)} #007bff,
                    stop:{max(0, (self.animation_offset-8)/100)} rgba(255,255,255,0.4),
                    stop:{min(1, self.animation_offset/100)} rgba(255,255,255,0.8),
                    stop:{min(1, (self.animation_offset+8)/100)} rgba(255,255,255,0.4),
                    stop:{min(1, (self.animation_offset+15)/100)} #007bff,
                    stop:1 #007bff);
                border-radius: 6px;
                margin: 2px;
            }}
        """
        
        # Apply the combined style
        self.progress_bar.setStyleSheet(animated_style + overlay_style)
        
        # Force a repaint to ensure the animation is visible
        self.progress_bar.repaint()
    
    def update_download_stats(self, stats: str):
        """Update download statistics in the status bar"""
        # Update the status text with download stats
        self.status_text.append(f"📊 {stats}")
        
        # Also update the progress bar tooltip with detailed stats
        self.progress_bar.setToolTip(f"Download Statistics:\n{stats}")
        
        # Force update the UI
        self.status_text.ensureCursorVisible()
    
    def download_finished(self, success: bool, message: str):
        """Handle download completion"""
        self.download_btn.setEnabled(True)
        self.stop_download_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        # Stop the animation timer
        self.animation_timer.stop()
        
        # Reset progress bar style to default
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #e1e5e9;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                color: #2c3e50;
                background-color: #f8f9fa;
                height: 25px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #007bff, stop:0.5 #0056b3, stop:1 #004085);
                border-radius: 6px;
                margin: 2px;
            }
        """)
        
        if success:
            # Store the model name before clearing currently_downloading
            downloaded_model_name = None
            if self.currently_downloading:
                downloaded_model_name = self.currently_downloading['name']
            
            # Clear currently downloading model
            self.currently_downloading = None
            
            # Refresh the model list to include the newly downloaded model
            # This will scan drives and find the actual downloaded model
            self.scan_for_existing_models()
            
            if downloaded_model_name:
                self.status_text.append(f"✅ Download completed: {downloaded_model_name}")
            
            QMessageBox.information(self, "Success", message)
        else:
            # Clear currently downloading model on failure
            failed_model_name = None
            failed_model_dir = None
            if self.currently_downloading:
                failed_model_name = self.currently_downloading.get('name')
                failed_model_dir = self.currently_downloading.get('dir')
            self.currently_downloading = None
            
            # Remove failed model from installed models and disk
            if failed_model_name:
                try:
                    self.drive_manager.remove_model(failed_model_name)
                    self.status_text.append(f"❌ Removed failed model: {failed_model_name}")
                except Exception as e:
                    self.status_text.append(f"⚠️ Could not remove failed model: {failed_model_name} ({e})")
                # Also try to remove the directory if it still exists
                if failed_model_dir and os.path.exists(failed_model_dir):
                    import shutil
                    try:
                        shutil.rmtree(failed_model_dir, ignore_errors=True)
                        self.status_text.append(f"🗑️ Deleted failed model directory: {failed_model_dir}")
                    except Exception as e:
                        self.status_text.append(f"⚠️ Could not delete failed model directory: {failed_model_dir} ({e})")
            
            # Refresh model list to remove any partial entries
            self.update_model_list()
            
            QMessageBox.critical(self, "Error", message)
    
    def on_conversion_finished(self, success: bool, message: str):
        """Handle conversion completion"""
        self.convert_to_gguf_btn.setEnabled(True)
        if success:
            self.status_text.append(f"✅ {message}")
            QMessageBox.information(self, "Conversion Complete", message)
        else:
            self.status_text.append(f"❌ {message}")
            QMessageBox.warning(self, "Conversion Failed", message)
    
    def update_conversion_stats(self, stats: str):
        """Update conversion statistics display"""
        self.status_text.append(f"📊 Conversion Stats: {stats}")
    
    def get_model_size(self, model_id: str) -> str:
        """Get the actual size of a model from Hugging Face Hub"""
        try:
            from huggingface_hub import model_info
            import signal
            
            # Check cache first
            cache_dir = Path.home() / ".huggingdrive_cache"
            cache_dir.mkdir(exist_ok=True)
            size_cache_file = cache_dir / "model_sizes.json"
            
            # Load cache
            size_cache = {}
            if size_cache_file.exists():
                try:
                    with open(size_cache_file, 'r') as f:
                        size_cache = json.load(f)
                except Exception:
                    pass
            
            # Check if size is cached
            if model_id in size_cache:
                return size_cache[model_id]
            
            # Use token if available
            token = self.auth_manager.get_token() if self.auth_manager.is_authenticated() else None
            
            # Set a timeout for the API call
            def timeout_handler(signum, frame):
                raise TimeoutError("Size fetch timed out")
            
            # Set 5 second timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(5)
            
            try:
                # Fetch from API
                info = model_info(model_id, token=token)
                signal.alarm(0)  # Cancel timeout
                
                # Calculate total size from siblings (all files in the repo)
                total_size = 0
                
                if hasattr(info, 'siblings') and info.siblings:
                    for sibling in info.siblings:
                        if hasattr(sibling, 'size') and sibling.size:
                            total_size += sibling.size
                
                # Format size
                if total_size > 0:
                    size_str = self.format_size(total_size)
                    # Cache the result
                    size_cache[model_id] = size_str
                    try:
                        with open(size_cache_file, 'w') as f:
                            json.dump(size_cache, f, indent=2)
                    except Exception:
                        pass
                    return size_str
                else:
                    return "Unknown"
                    
            except TimeoutError:
                signal.alarm(0)  # Cancel timeout
                self.status_text.append("⚠️ Size fetch timed out, showing Unknown")
                return "Unknown"
            except Exception as e:
                signal.alarm(0)  # Cancel timeout
                self.status_text.append(f"⚠️ Could not fetch model size: {str(e)}")
                return "Unknown"
                
        except Exception as e:
            self.status_text.append(f"⚠️ Could not fetch model size: {str(e)}")
            return "Unknown"
    
    def format_size(self, size_bytes: float) -> str:
        """Format size in bytes to human readable format"""
        if size_bytes == 0:
            return "0B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f}{size_names[i]}"
    
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
    
    def get_available_space_gb(self) -> Optional[float]:
        """Get available space on the selected drive in GB"""
        if self.selected_download_folder:
            # Use custom folder's drive
            drive_path = str(Path(self.selected_download_folder).parent)
        elif self.drive_combo.count() > 0:
            drive_path = self.drive_combo.currentData()
        else:
            return None
        
        try:
            usage = psutil.disk_usage(drive_path)
            return usage.free / (1024**3)  # Convert to GB
        except:
            return None
    
    def parse_size_to_gb(self, size_str: str) -> Optional[float]:
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

    def on_drive_selection_changed(self):
        """Handle drive selection change"""
        drive_path = self.drive_combo.currentData()
        if drive_path:
            self.eject_drive_btn.setEnabled(True)
            self.drive_status_label.setText(f"Drive selected: {drive_path}")
            self.drive_status_label.setStyleSheet("color: green; font-weight: bold;")
            self.set_cache_directories(drive_path)
        else:
            self.eject_drive_btn.setEnabled(False)
            self.drive_status_label.setText("No drive selected")
            self.drive_status_label.setStyleSheet("color: red; font-weight: bold;")

    def eject_selected_drive(self):
        """Safely eject the selected external drive"""
        drive_path = self.drive_combo.currentData()
        if not drive_path:
            QMessageBox.warning(self, "Error", "No drive selected")
            return
        
        reply = QMessageBox.question(self, "Confirm Eject", 
                                   f"Are you sure you want to eject '{drive_path}'?",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                if sys.platform == "darwin":  # macOS
                    os.system(f"diskutil eject '{drive_path}'")
                elif sys.platform == "win32":  # Windows
                    os.system(f"powershell -Command \"(New-Object -ComObject Shell.Application).NameSpace(17).ParseName('{drive_path}').InvokeVerb('Eject')\"")
                else:  # Linux
                    os.system(f"eject '{drive_path}'")
                
                self.status_text.append(f"Ejected drive: {drive_path}")
                self.update_drive_list()  # Refresh the drive list
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to eject drive: {str(e)}")
                self.status_text.append(f"Error ejecting drive: {str(e)}")

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

    def check_system_requirements(self):
        """Check system requirements and display information"""
        try:
            import platform
            import psutil
            
            # Get system info
            system_info = f"System: {platform.system()} {platform.release()}"
            cpu_info = f"CPU: {platform.processor()}"
            memory = psutil.virtual_memory()
            ram_info = f"RAM: {memory.total / (1024**3):.1f}GB total, {memory.available / (1024**3):.1f}GB available"
            
            # Check Python version
            python_info = f"Python: {sys.version}"
            
            # Check PyTorch
            try:
                import torch
                torch_info = f"PyTorch: {torch.__version__}"
                cuda_available = f"CUDA: {'Available' if torch.cuda.is_available() else 'Not available'}"
            except ImportError:
                torch_info = "PyTorch: Not installed"
                cuda_available = "CUDA: Not available"
            
            # Check transformers
            try:
                import transformers
                transformers_info = f"Transformers: {transformers.__version__}"
            except ImportError:
                transformers_info = "Transformers: Not installed"
            
            # Display information
            info_text = f"""
System Requirements Check:
{system_info}
{cpu_info}
{ram_info}
{python_info}
{torch_info}
{cuda_available}
{transformers_info}

Recommendations:
- At least 8GB RAM for most models
- 16GB+ RAM for large language models
- CUDA GPU recommended for faster inference
- At least 10GB free disk space for model storage
"""
            
            QMessageBox.information(self, "System Requirements", info_text)
            self.status_text.append("System requirements check completed")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to check system requirements: {str(e)}")
            self.status_text.append(f"Error checking system requirements: {str(e)}")

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

    def load_popular_models(self):
        """Load popular models by default"""
        self.status_text.append("Loading popular models...")
        self.perform_model_search("", show_popular=True)

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
        
        # Add currently downloading model if any (but don't duplicate if it already exists)
        if self.currently_downloading:
            downloading_name = self.currently_downloading['name']
            # Only add if not already in the models list
            if downloading_name not in models:
                models[downloading_name] = self.currently_downloading
        
        # Also include models from custom folder
        if self.selected_download_folder:
            custom_models = self.drive_manager.scan_custom_folder_for_models(self.selected_download_folder)
            for model in custom_models:
                models[model['name']] = model
        
        # Display models in the list
        for model_name, model_info in models.items():
            model_path = model_info.get('path', 'Unknown')
            source = model_info.get('source', 'Unknown')
            
            # Calculate size if not already available
            size_str = model_info.get('size', 'Unknown')
            if size_str == 'Unknown' and model_path != 'Unknown':
                size_str = self.drive_manager.calculate_model_size(model_path)
            
            use = "Unknown"
            
            # Try to determine model type from config.json
            try:
                config_path = Path(model_path) / "config.json"
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    model_type = config.get('model_type', '').lower()
                    archs = [a.lower() for a in config.get('architectures', [])]
                    if any(x in model_type for x in ["gpt", "llama", "mistral", "qwen", "falcon", "bloom", "opt"]):
                        use = "Chat/Text Generation"
                    elif any(x in model_type for x in ["bert", "roberta", "distilbert"]):
                        use = "Classification"
                    elif any(x in model_type for x in ["t5", "bart", "marian"]):
                        use = "Seq2Seq"
                    elif any(x in model_type for x in ["vit", "resnet", "vision"]):
                        use = "Vision"
                    elif any(x in model_type for x in ["wav2vec", "whisper", "audio"]):
                        use = "Audio"
                    elif any(x in archs for x in ["blenderbot"]):
                        use = "Seq2Seq"
            except Exception:
                pass
            
            # Add download indicator for currently downloading models
            status_indicator = ""
            if self.currently_downloading and model_name == self.currently_downloading.get('name'):
                status_indicator = "⏳ "
            
            item_text = f"{status_indicator}{model_name}\n  Path: {model_path}\n  Source: {source}\n  Size: {size_str}\n  Use: {use}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, model_name)
            self.model_list.addItem(item)
        
        self.status_text.append(f"Displaying {len(models)} model(s)")

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
            
            # Only update if values have changed significantly to reduce UI flicker
            new_ram_text = f"🧠 RAM: {used_ram_gb:.1f}GB/{total_ram_gb:.1f}GB ({ram_percent:.0f}%)"
            new_app_text = f"📱 App: {app_memory_mb:.0f}MB"
            
            # Update RAM usage label only if text changed
            if self.ram_usage_label.text() != new_ram_text:
                ram_color = "green" if ram_percent < 70 else "orange" if ram_percent < 90 else "red"
                self.ram_usage_label.setText(new_ram_text)
                self.ram_usage_label.setStyleSheet(f"background-color: {ram_color}; color: white; padding: 5px; border-radius: 3px; font-weight: bold;")
            
            # Update app memory label only if text changed
            if self.app_memory_label.text() != new_app_text:
                app_color = "green" if app_memory_mb < 500 else "orange" if app_memory_mb < 1000 else "red"
                self.app_memory_label.setText(new_app_text)
                self.app_memory_label.setStyleSheet(f"background-color: {app_color}; color: white; padding: 5px; border-radius: 3px; font-weight: bold;")
            
        except Exception as e:
            self.ram_usage_label.setText("🧠 RAM: Error")
            self.app_memory_label.setText("📱 App: Error")
            self.ram_usage_label.setStyleSheet("background-color: red; color: white; padding: 5px; border-radius: 3px; font-weight: bold;")
            self.app_memory_label.setStyleSheet("background-color: red; color: white; padding: 5px; border-radius: 3px; font-weight: bold;")

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
            
            # Refresh the model list to show models from the selected drive with updated sizes
            self.scan_for_existing_models()

    def on_model_selection_changed(self, current, previous):
        """Handle model selection in the list"""
        if current:
            # Enable the test, open folder, and GGUF conversion buttons
            self.test_model_btn.setEnabled(True)
            self.open_model_folder_btn.setEnabled(True)
            self.convert_to_gguf_btn.setEnabled(True)
            self.start_api_btn.setEnabled(True)
        else:
            # Disable the buttons when no model is selected
            self.test_model_btn.setEnabled(False)
            self.open_model_folder_btn.setEnabled(False)
            self.convert_to_gguf_btn.setEnabled(False)
            self.start_api_btn.setEnabled(False)

    def remove_selected_model(self):
        """Remove the selected installed model and delete its files"""
        current_item = self.model_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Error", "Please select a model to remove")
            return
        model_name = current_item.data(Qt.ItemDataRole.UserRole)
        models = self.drive_manager.get_installed_models()
        if model_name in models:
            model_info = models[model_name]
            model_path = model_info['path']
            
            # Verify the path exists before asking for confirmation
            if not os.path.exists(model_path):
                QMessageBox.warning(self, "Error", f"Model path does not exist: {model_path}\nThe model may have been moved or deleted manually.")
                # Remove from config anyway
                self.drive_manager.remove_model(model_name)
                self.update_model_list()
                return
            
            reply = QMessageBox.question(self, "Confirm Delete", 
                                       f"Are you sure you want to delete the model and all its files?\n\nModel: {model_name}\nPath: {model_path}", 
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                try:
                    self.status_text.append(f"🗑️ Deleting model: {model_name}")
                    self.status_text.append(f"   Path: {model_path}")
                    
                    def robust_remove_directory(path):
                        """Robustly remove directory and all contents, handling macOS hidden files"""
                        import stat
                        import subprocess
                        
                        # First, try to remove macOS hidden files using system commands
                        if sys.platform == "darwin":
                            try:
                                # Remove all ._* files first
                                subprocess.run(['find', path, '-name', '._*', '-delete'], 
                                             capture_output=True, check=False)
                                # Remove all .DS_Store files
                                subprocess.run(['find', path, '-name', '.DS_Store', '-delete'], 
                                             capture_output=True, check=False)
                            except Exception as e:
                                self.status_text.append(f"   Warning: Could not remove hidden files: {e}")
                        
                        # Walk through directory tree and remove files
                        for root, dirs, files in os.walk(path, topdown=False):
                            # Remove files first
                            for file in files:
                                file_path = os.path.join(root, file)
                                try:
                                    # Make file writable and remove
                                    os.chmod(file_path, stat.S_IWRITE | stat.S_IREAD)
                                    os.remove(file_path)
                                except (OSError, PermissionError) as e:
                                    # Try using system rm command as fallback
                                    try:
                                        if sys.platform == "darwin":
                                            subprocess.run(['rm', '-f', file_path], 
                                                         capture_output=True, check=False)
                                        else:
                                            subprocess.run(['rm', '-f', file_path], 
                                                         capture_output=True, check=False)
                                    except Exception:
                                        self.status_text.append(f"   Warning: Could not remove file {file_path}: {e}")
                            
                            # Remove directories
                            for dir in dirs:
                                dir_path = os.path.join(root, dir)
                                try:
                                    os.chmod(dir_path, stat.S_IWRITE | stat.S_IREAD)
                                    os.rmdir(dir_path)
                                except (OSError, PermissionError) as e:
                                    # Try using system rmdir command as fallback
                                    try:
                                        subprocess.run(['rmdir', dir_path], 
                                                     capture_output=True, check=False)
                                    except Exception:
                                        self.status_text.append(f"   Warning: Could not remove directory {dir_path}: {e}")
                        
                        # Finally remove the root directory
                        try:
                            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
                            os.rmdir(path)
                        except (OSError, PermissionError) as e:
                            # Try using system rmdir command as fallback
                            try:
                                subprocess.run(['rmdir', path], 
                                             capture_output=True, check=False)
                            except Exception:
                                self.status_text.append(f"   Warning: Could not remove root directory {path}: {e}")
                    
                    # Use robust removal method
                    robust_remove_directory(model_path)
                    
                    # Remove from drive manager config
                    success = self.drive_manager.remove_model(model_name)
                    if not success:
                        self.status_text.append("⚠️ Warning: Could not update model configuration")
                    
                    # Update the UI
                    self.update_model_list()
                    
                    # Final verification
                    if os.path.exists(model_path):
                        # Try one more time with system commands
                        try:
                            if sys.platform == "darwin":
                                subprocess.run(['rm', '-rf', model_path], 
                                             capture_output=True, check=False)
                            else:
                                subprocess.run(['rm', '-rf', model_path], 
                                             capture_output=True, check=False)
                        except Exception:
                            pass
                        
                        if os.path.exists(model_path):
                            QMessageBox.warning(self, "Deletion Incomplete", 
                                              f"Some files could not be deleted from:\n{model_path}\n\nYou may need to delete them manually.")
                            self.status_text.append(f"❌ Model deletion incomplete - some files remain at {model_path}")
                        else:
                            QMessageBox.information(self, "Model Removed", f"Model '{model_name}' was successfully deleted.")
                            self.status_text.append(f"✅ Model '{model_name}' deleted successfully")
                    else:
                        QMessageBox.information(self, "Model Removed", f"Model '{model_name}' was successfully deleted.")
                        self.status_text.append(f"✅ Model '{model_name}' deleted successfully")
                        
                except Exception as e:
                    error_msg = f"Failed to delete model: {str(e)}"
                    QMessageBox.critical(self, "Error", error_msg)
                    self.status_text.append(f"❌ {error_msg}")
        else:
            QMessageBox.warning(self, "Error", f"Model {model_name} not found in installed models")

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

    def convert_selected_model_to_gguf(self):
        """Convert the selected model to GGUF format"""
        current_item = self.model_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Error", "Please select a model to convert to GGUF")
            return
        
        model_name = current_item.data(Qt.ItemDataRole.UserRole)
        models = self.drive_manager.get_installed_models()
        
        if model_name in models:
            model_info = models[model_name]
            model_path = model_info['path']
            
            # Show quantization settings dialog
            dialog = QuantizationDialog(model_path, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                settings = dialog.get_conversion_settings()
                
                # Create output path
                output_path = str(Path(model_path).parent / f"{model_name}_gguf")
                
                # Start conversion thread with selected settings
                self.conversion_thread = GGUFConverter(
                    model_path, 
                    output_path,
                    quantization=settings['quantization'],
                    context_length=settings['context_length']
                )
                self.conversion_thread.progress_signal.connect(self.update_status)
                self.conversion_thread.finished_signal.connect(self.on_conversion_finished)
                self.conversion_thread.progress_bar_signal.connect(self.update_progress_bar)
                self.conversion_thread.conversion_stats_signal.connect(self.update_conversion_stats)
                self.conversion_thread.start()
                
                self.convert_to_gguf_btn.setEnabled(False)
                self.status_text.append(f"🚀 Starting GGUF conversion of {model_name}...")
        else:
            QMessageBox.warning(self, "Error", f"Model {model_name} not found in installed models")

    def check_model_access(self):
        """Check if user has access to the selected model"""
        model_name = self.model_name_input.text().strip()
        if not model_name:
            QMessageBox.warning(self, "Error", "Please enter a model name first")
            return
        
        # Remove size info if present
        if ' (' in model_name:
            model_name = model_name.split(' (')[0].strip()
        
        # Check if user is authenticated
        if not self.auth_manager.is_authenticated():
            QMessageBox.warning(self, "Not Authenticated", 
                              "You need to log in first to check model access.\n\n"
                              "Click the '🔑 Login to HF' button to log in.")
            return
        
        self.status_text.append(f"🔍 Checking access to {model_name}...")
        
        try:
            # Create a temporary downloader to use its check_model_access method
            from .downloader import ModelDownloader
            temp_downloader = ModelDownloader(model_name, "/tmp")
            has_access, message = temp_downloader.check_model_access(model_name)
            
            if has_access:
                QMessageBox.information(self, "Access Granted", 
                                      f"✅ {message}\n\n"
                                      f"You can now download this model!")
                self.status_text.append(f"✅ {message}")
            else:
                # Show detailed information about access issues
                if "request access" in message.lower():
                    QMessageBox.information(self, "Access Required", 
                                          f"🔒 {message}\n\n"
                                          f"To get access:\n"
                                          f"1. Visit: https://huggingface.co/{model_name}\n"
                                          f"2. Click 'Request access'\n"
                                          f"3. Wait for approval\n"
                                          f"4. Try downloading again")
                else:
                    QMessageBox.warning(self, "Access Denied", message)
                self.status_text.append(f"❌ {message}")
                
        except Exception as e:
            error_msg = f"Error checking access: {str(e)}"
            QMessageBox.critical(self, "Error", error_msg)
            self.status_text.append(f"❌ {error_msg}")
    
    def stop_download(self):
        """Stop the current download thread if running"""
        if hasattr(self, 'downloader') and self.downloader and self.downloader.isRunning():
            self.progress_signal.emit("Stopping download...")
            self.downloader.stop()  # Signal the downloader to stop
            self.downloader.wait()  # Wait for it to finish
            self.update_status("❌ Download stopped by user.")
            self.download_btn.setEnabled(True)
            self.stop_download_btn.setEnabled(False)
            self.progress_bar.setVisible(False)
            QMessageBox.information(self, "Download Stopped", "The download was stopped.")

    def reset_model_selection(self):
        """Reset the model selection display"""
        self.model_info_label.setText("Select a model from search results to download")
        self.model_info_label.setStyleSheet("color: #666; font-style: italic;")

    def toggle_api_server(self):
        if self.api_server_process and self.api_server_process.poll() is None:
            # Server is running, stop it
            self.api_server_process.terminate()
            self.api_server_process = None
            self.api_server_port = None
            self.start_api_btn.setText("Start API Server (Test mode)")
        else:
            # Start the server
            self.start_api_server()
            self.start_api_btn.setText("Stop API Server (Test mode)")

    def start_api_server(self):
        """Start a FastAPI server for the selected model in a background process."""
        # Get the selected model's path
        current_item = self.model_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Error", "Please select a model to serve via API.")
            return
        model_name = current_item.data(Qt.ItemDataRole.UserRole)
        models = self.drive_manager.get_installed_models()
        if model_name in models:
            model_info = models[model_name]
            model_path = model_info['path']
        else:
            QMessageBox.warning(self, "Error", f"Model {model_name} not found in installed models")
            return
        # Pick a random available port between 8000 and 9000
        for _ in range(20):
            port = random.randint(8000, 9000)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    self.api_server_port = port
                    break
        else:
            QMessageBox.warning(self, "Error", "Could not find a free port for the API server.")
            return
        # ... existing code to build server_code ...
        server_code = '''
import sys
from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from transformers import pipeline
import uvicorn

model_path = r"{}"
pipe = pipeline("text-generation", model=model_path)
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h2>Local Model API</h2>
    <form action='/generate' method='post'>
      <input name='prompt' placeholder='Type your prompt here' style='width:300px;'>
      <button type='submit'>Generate</button>
    </form>
    <p>Or POST JSON to <code>/generate</code> with {{"prompt": "your text"}}</p>
    """

@app.get("/generate")
def get_generate():
    return JSONResponse({{"error": "Use POST with a JSON body: {{'prompt': 'your text'}}"}})

@app.post("/generate")
async def generate(request: Request, prompt: str = Form(None)):
    try:
        # Support both form and JSON
        if prompt is None:
            try:
                data = await request.json()
                prompt = data.get("prompt", "")
            except Exception:
                prompt = ""
        if not prompt:
            return {{"error": "No prompt provided."}}
        result = pipe(prompt, max_new_tokens=50)
        return {{"result": result[0]["generated_text"]}}
    except Exception as e:
        return {{"error": str(e)}}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port={})
'''.format(model_path, self.api_server_port)
        import subprocess
        import tempfile
        import os
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(server_code)
            server_path = f.name
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_CONSOLE
        else:
            creationflags = 0
        self.api_server_process = subprocess.Popen([sys.executable, server_path], creationflags=creationflags)
        # Show a dialog with server info
        dlg = QMessageBox(self)
        dlg.setWindowTitle("API Server Running")
        dlg.setText(f"API server running for model: {self.model_name_input.text()}\n\nPOST to: http://127.0.0.1:{self.api_server_port}/generate\n\nExample JSON: {{\"prompt\": \"Hello world\"}}\n\nTo stop the server, click 'Stop API Server'.")
        dlg.setStandardButtons(QMessageBox.StandardButton.Ok)
        dlg.exec()

    def show_paste_gguf_url_dialog(self):
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLineEdit, QLabel, QPushButton
        from urllib.parse import urlparse, unquote
        
        class PasteUrlDialog(QDialog):
            def __init__(self, parent=None):
                super().__init__(parent)
                self.setWindowTitle("Paste Model URL")
                self.setModal(True)
                layout = QVBoxLayout()
                self.setLayout(layout)
                
                # Instructions
                layout.addWidget(QLabel("Paste the model URL below:"))
                layout.addWidget(QLabel("Example: https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/..."))
                
                # URL input
                self.url_input = QLineEdit()
                self.url_input.setPlaceholderText("https://huggingface.co/...")
                layout.addWidget(self.url_input)
                
                # Buttons
                btn_layout = QVBoxLayout()
                self.ok_btn = QPushButton("OK")
                self.ok_btn.clicked.connect(self.accept)
                btn_layout.addWidget(self.ok_btn)
                
                self.cancel_btn = QPushButton("Cancel")
                self.cancel_btn.clicked.connect(self.reject)
                btn_layout.addWidget(self.cancel_btn)
                
                layout.addLayout(btn_layout)
            
            def get_url(self):
                return self.url_input.text().strip()
        
        dialog = PasteUrlDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            url = dialog.get_url()
            if not url:
                return
            
            try:
                # Parse the URL to get the model name
                parsed = urlparse(url)
                path_parts = parsed.path.split('/')
                
                # Check if this is a direct file URL
                if 'resolve' in path_parts and url.endswith('.gguf'):
                    # This is a direct GGUF file download
                    # Extract the filename for display
                    filename = os.path.basename(parsed.path.split('?')[0])
                    
                    # Get file size before downloading
                    try:
                        response = requests.head(url, timeout=30)
                        total_size = int(response.headers.get('content-length', 0))
                        size_gb = total_size / (1024**3)
                        size_info = f" ({size_gb:.2f}GB)"
                    except:
                        size_info = ""
                    
                    # Set the URL directly as the model name for the downloader
                    self.model_name_input.setText(url)
                    
                    # Show helpful messages
                    self.update_status(f"✅ Direct GGUF file detected: {filename}{size_info}")
                    self.update_status("You can now click Download to start downloading the file")
                    
                    # Clear any previous search results
                    self.search_results_list.clear()
                    return
                
                # Check if this is a GGUF file in a repository
                if 'huggingface.co' in parsed.netloc and len(path_parts) >= 3:
                    repo_name = f"{path_parts[1]}/{path_parts[2]}"
                    
                    # If it's a specific file path
                    if len(path_parts) > 4 and path_parts[-1].endswith('.gguf'):
                        # This is a specific GGUF file in a repository
                        # Extract the file path without duplicate resolve/main
                        file_path = path_parts[-1]  # Just get the filename
                        direct_url = f"https://huggingface.co/{repo_name}/resolve/main/{file_path}"
                        
                        # Get file size with better error handling and retries
                        size_info = ""
                        try:
                            # Try HEAD request first
                            response = requests.head(direct_url, timeout=30, allow_redirects=True)
                            if response.status_code == 200:
                                total_size = int(response.headers.get('content-length', 0))
                                if total_size > 0:
                                    size_gb = total_size / (1024**3)
                                    size_info = f" ({size_gb:.2f}GB)"
                                else:
                                    # If HEAD doesn't return size, try GET with stream
                                    response = requests.get(direct_url, stream=True, timeout=30)
                                    response.raise_for_status()
                                    total_size = int(response.headers.get('content-length', 0))
                                    if total_size > 0:
                                        size_gb = total_size / (1024**3)
                                        size_info = f" ({size_gb:.2f}GB)"
                                    else:
                                        size_info = " (size unknown)"
                            else:
                                size_info = " (size unknown)"
                        except Exception as e:
                            self.status_text.append(f"Note: Could not fetch file size: {str(e)}")
                            size_info = " (size unknown)"
                        
                        # Store just the model name and filename for better organization
                        self.model_name_input.setText(f"{repo_name}/{file_path}")
                        
                        # Show helpful messages
                        self.update_status(f"✅ GGUF file found in repository: {file_path}{size_info}")
                        self.update_status("You can now click Download to start downloading the file")
                        
                        # Clear any previous search results
                        self.search_results_list.clear()
                        return
                    
                    # If it's just a repository URL, search for GGUF files
                    model_name = unquote(repo_name)  # Handle URL encoding
                    
                    # Set the model name in the search bar
                    self.model_name_input.setText(model_name)
                    
                    # Trigger a search for this model
                    self.perform_model_search(model_name)
                    
                    # Show a helpful message
                    self.update_status(f"✅ Found model repository: {model_name}")
                    self.update_status("Please select a specific GGUF file to download")
                    return
                
                QMessageBox.warning(self, "Error", "Please use a valid Hugging Face model URL (https://huggingface.co/...)")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to process URL: {str(e)}")
    
    def update_auth_status(self):
        """Update authentication status display"""
        if self.auth_manager.is_authenticated():
            username = self.auth_manager.get_username()
            self.auth_status_label.setText(f"✅ Logged in as {username}")
            self.auth_status_label.setStyleSheet("background-color: #d4edda; padding: 5px; border-radius: 3px; font-weight: bold; color: #155724;")
            # Show green login button when logged in
            self.login_btn.setText("✅ Logged In")
            self.login_btn.setStyleSheet("padding: 5px; background-color: #28a745; color: white; border: none; border-radius: 3px; font-weight: bold;")
            self.login_btn.setEnabled(False)  # Disable when logged in
            self.logout_btn.setVisible(True)
        else:
            self.auth_status_label.setText("🔓 Not logged in")
            self.auth_status_label.setStyleSheet("background-color: #fff3cd; padding: 5px; border-radius: 3px; font-weight: bold; color: #856404;")
            # Show normal login button when not logged in
            self.login_btn.setText("🔑 Login to HF")
            self.login_btn.setStyleSheet("padding: 5px; background-color: #4CAF50; color: white; border: none; border-radius: 3px;")
            self.login_btn.setEnabled(True)  # Enable when not logged in
            self.logout_btn.setVisible(False)
    
    def show_login_dialog(self):
        """Show the login dialog"""
        dialog = LoginDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            token = dialog.get_token()
            if token:
                success, message = self.auth_manager.login(token)
                if success:
                    QMessageBox.information(self, "Login Successful", message)
                    self.update_auth_status()
                    self.user_token = self.auth_manager.get_token()  # Get token from auth manager
                else:
                    QMessageBox.warning(self, "Login Failed", message)
    
    def logout_huggingface(self):
        """Logout from Hugging Face"""
        reply = QMessageBox.question(self, "Logout", 
            "Are you sure you want to logout? You'll need to login again to access restricted models.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            self.auth_manager.logout()
            self.update_auth_status()
            QMessageBox.information(self, "Logged Out", "Successfully logged out from Hugging Face Hub")

    # Simplified lock mechanism - locks stay visible even when logged in

    def is_gguf_repository(self, repo_id: str) -> bool:
        """Check if a repository contains multiple GGUF files"""
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            files = api.list_repo_files(repo_id=repo_id)
            gguf_files = [f for f in files if f.endswith('.gguf')]
            return len(gguf_files) > 1  # Multiple GGUF files indicate quantization options
        except Exception as e:
            self.status_text.append(f"⚠️ Error checking GGUF files: {str(e)}")
            return False
    
    def download_specific_gguf_file(self, repo_id: str, filename: str):
        """Download a specific GGUF file from a repository"""
        try:
            # Determine the target path
            if self.selected_download_folder:
                target_path = str(Path(self.selected_download_folder))
            else:
                drive_path = self.drive_combo.currentData()
                if not drive_path:
                    QMessageBox.warning(self, "Error", "Please select a drive or choose a download folder")
                    return
                target_path = str(Path(drive_path) / "huggingface_models")
            
            # Create the target directory if it doesn't exist
            try:
                os.makedirs(target_path, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not create download directory: {str(e)}")
                return
            
            # Create model directory (remove .gguf extension)
            model_name = filename.replace('.gguf', '')
            model_dir = os.path.join(target_path, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Store currently downloading model info
            self.currently_downloading = {
                'name': model_name,
                'path': model_dir,
                'size': 'Unknown (GGUF file)',
                'source': f'GGUF Download: {filename}'
            }
            
            # Update model list to show downloading model
            self.update_model_list()
            
            # Start the download using the specific file URL
            from huggingface_hub import hf_hub_url
            file_url = hf_hub_url(repo_id=repo_id, filename=filename)
            
            self.downloader = ModelDownloader(file_url, model_dir)
            self.downloader.progress_signal.connect(self.update_status)
            self.downloader.finished_signal.connect(self.download_finished)
            self.downloader.progress_bar_signal.connect(self.update_progress_bar)
            self.downloader.download_stats_signal.connect(self.update_download_stats)
            self.downloader.start()
            
            self.download_btn.setEnabled(False)
            self.stop_download_btn.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            self.status_text.append(f"🚀 Starting download of {filename}")
            
            # Start progress animation
            if not self.animation_timer.isActive():
                self.animation_timer.start(50)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start GGUF download: {str(e)}")
            self.status_text.append(f"Error starting GGUF download: {str(e)}")

    def get_system_recommendation(self):
        """Get smart recommendation based on system specs"""
        try:
            import psutil
            
            # Get system specs
            total_ram = psutil.virtual_memory().total / (1024**3)  # GB
            cpu_count = psutil.cpu_count()
            
            # Determine recommendation based on system capabilities
            if total_ram >= 32:
                return "Q4_K_M or Q5_K_M - Your system has plenty of RAM (32GB+) for high-quality models"
            elif total_ram >= 16:
                return "Q4_K_M - Optimal balance for your 16GB system. Q5_K_M works but uses more memory"
            elif total_ram >= 8:
                return "Q4_K_S or Q4_0 - Best for your 8GB system. Avoid Q8_0 or larger quantizations"
            else:
                return "Q4_K_S or Q3_K_M - Limited RAM detected. Choose smaller quantizations for stability"
                
        except Exception:
            return "Q4_K_M - Recommended default for most systems"
    
    def get_quality_level(self, quant_type):
        """Get quality level description for quantization type"""
        if not quant_type:
            return "Unknown"
        
        quant_lower = quant_type.lower()
        if 'q2' in quant_lower:
            return "Low"
        elif 'q3' in quant_lower:
            return "Low-Medium"
        elif 'q4' in quant_lower:
            return "Medium"
        elif 'q5' in quant_lower:
            return "Medium-High"
        elif 'q6' in quant_lower:
            return "High"
        elif 'q8' in quant_lower:
            return "Very High"
        elif 'f16' in quant_lower or 'bf16' in quant_lower:
            return "Original"
        else:
            return "Unknown"


class GGUFConversionThread(QThread):
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self, model_path, output_path):
        super().__init__()
        self.model_path = model_path
        self.output_path = output_path
        
    def run(self):
        try:
            self.progress_signal.emit("Checking model format...")
            model_path = Path(self.model_path)
            if not (model_path / "config.json").exists():
                self.finished_signal.emit(False, "Model does not appear to be in Hugging Face format")
                return
            with open(model_path / "config.json", 'r') as f:
                config = json.load(f)
            model_type = config.get('model_type', 'unknown').lower()
            architectures = [a.lower() for a in config.get('architectures', [])]
            self.progress_signal.emit(f"Model type: {model_type}, Architectures: {architectures}")
            # Determine conversion script
            if any('llama' in arch for arch in architectures) or 'llama' in model_type:
                conversion_script = "convert-llama-gguf.py"
            elif any('qwen' in arch for arch in architectures) or 'qwen' in model_type:
                conversion_script = "convert-qwen-gguf.py"
            elif any('mistral' in arch for arch in architectures) or 'mistral' in model_type:
                conversion_script = "convert-mistral-gguf.py"
            elif any('gpt2' in arch for arch in architectures) or 'gpt2' in model_type:
                conversion_script = "convert-gpt2-gguf.py"
            elif any('gpt' in arch for arch in architectures) or 'gpt' in model_type:
                conversion_script = "convert-gpt-gguf.py"
            else:
                conversion_script = "convert-gguf.py"
            self.progress_signal.emit(f"Using conversion script: {conversion_script}")
            possible_paths = [
                f"llama.cpp/{conversion_script}",
                conversion_script,
                "llama.cpp/convert.py",
                "convert.py"
            ]
            conversion_script_path = None
            for path in possible_paths:
                if Path(path).exists():
                    conversion_script_path = path
                    break
            if not conversion_script_path:
                self.finished_signal.emit(False, "llama.cpp conversion script not found. Please download llama.cpp and place the conversion scripts in the app directory. See: https://github.com/ggerganov/llama.cpp#converting-models")
                return
            output_dir = Path(self.output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            self.progress_signal.emit("Starting conversion...")
            cmd = [
                sys.executable,
                conversion_script_path,
                str(model_path),
                "--outfile", str(output_dir / f"{model_path.name}.gguf"),
                "--outtype", "q4_0"
            ]
            self.progress_signal.emit(f"Running: {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            if process.stdout:
                for line in process.stdout:
                    if line.strip():
                        self.progress_signal.emit(line.strip())
            process.wait()
            if process.returncode == 0:
                self.finished_signal.emit(True, f"Conversion successful! GGUF model saved to: {output_dir}")
            else:
                self.finished_signal.emit(False, f"Conversion failed with return code: {process.returncode}")
        except Exception as e:
            self.finished_signal.emit(False, f"Conversion error: {str(e)}")

class GGUFQuantizationDialog(QDialog):
    def __init__(self, repo_id, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select GGUF Quantization")
        self.setModal(True)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowStaysOnTopHint)
        self.selected_file = None
        self.repo_id = repo_id
        self.resize(600, 400)
        
        # Set dialog to be modal and block parent
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, False)
        
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Title
        title_label = QLabel(f"Available quantizations for: <b>{repo_id}</b>")
        title_label.setStyleSheet("font-size: 14px; font-weight: bold; margin-bottom: 10px; color: white;")
        layout.addWidget(title_label)
        
        # Smart recommendation based on system specs
        recommendation = self.get_system_recommendation()
        if recommendation:
            rec_label = QLabel(f"💡 <b>Smart Recommendation:</b> {recommendation}")
            rec_label.setStyleSheet("color: #4CAF50; font-size: 12px; margin-bottom: 10px; padding: 8px; background-color: rgba(76, 175, 80, 0.1); border-radius: 5px; border-left: 3px solid #4CAF50;")
            rec_label.setWordWrap(True)
            layout.addWidget(rec_label)
        
        # Description
        desc_label = QLabel("Choose the quantization level that best fits your needs:")
        desc_label.setStyleSheet("color: #666; margin-bottom: 10px;")
        layout.addWidget(desc_label)
        
        # Quantization options table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Quantization", "Size", "Quality", "Description"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)  # Better visual separation
        layout.addWidget(self.table)
        
        # Info panel
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("background-color: rgba(248, 249, 250, 0.1); padding: 10px; border-radius: 5px; margin: 10px 0; border: 1px solid #ddd;")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.ok_btn = QPushButton("Download Selected")
        self.ok_btn.clicked.connect(self.accept)
        self.ok_btn.setEnabled(False)
        self.ok_btn.setDefault(True)  # Make this the default button
        btn_layout.addWidget(self.ok_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(btn_layout)
        
        # Set focus to the table
        self.table.setFocus()
        
        # Populate data
        self.populate_quantizations(repo_id)
        self.table.itemSelectionChanged.connect(self.update_info)
        
        # Ensure dialog is properly modal
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
    
    def get_system_recommendation(self):
        """Get smart recommendation based on system specs"""
        try:
            import psutil
            
            # Get system specs
            total_ram = psutil.virtual_memory().total / (1024**3)  # GB
            cpu_count = psutil.cpu_count()
            
            # Determine recommendation based on system capabilities
            if total_ram >= 32:
                return "Q4_K_M or Q5_K_M - Your system has plenty of RAM (32GB+) for high-quality models"
            elif total_ram >= 16:
                return "Q4_K_M - Optimal balance for your 16GB system. Q5_K_M works but uses more memory"
            elif total_ram >= 8:
                return "Q4_K_S or Q4_0 - Best for your 8GB system. Avoid Q8_0 or larger quantizations"
            else:
                return "Q4_K_S or Q3_K_M - Limited RAM detected. Choose smaller quantizations for stability"
                
        except Exception:
            return "Q4_K_M - Recommended default for most systems"
    
    def populate_quantizations(self, repo_id):
        """Populate the quantization options table"""
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            files = api.list_repo_files(repo_id=repo_id)
            gguf_files = [f for f in files if f.endswith('.gguf')]
            
            if not gguf_files:
                self.table.setRowCount(1)
                self.table.setItem(0, 0, QTableWidgetItem("No GGUF files found"))
                self.ok_btn.setEnabled(False)
                return
            
            # Get file sizes using model info
            try:
                info = api.model_info(repo_id)
                siblings = getattr(info, 'siblings', []) or []
                file_sizes = {}
                for sibling in siblings:
                    filename = getattr(sibling, 'rfilename', None)
                    size = getattr(sibling, 'size', None)
                    if filename and size is not None:
                        file_sizes[filename] = size
                            
            except Exception as e:
                print(f"Warning: Could not fetch file sizes: {e}")
                file_sizes = {}
            
            # Quantization descriptions
            quant_descriptions = {
                'Q2_K': '2-bit quantization, smallest size, lowest quality',
                'Q3_K_S': '3-bit quantization, small size, low quality',
                'Q3_K_M': '3-bit quantization, medium size, moderate quality',
                'Q3_K_L': '3-bit quantization, large size, good quality',
                'Q4_0': '4-bit quantization, fast inference, moderate quality',
                'Q4_1': '4-bit quantization, balanced speed and quality',
                'Q4_K_S': '4-bit quantization, smaller size, lower quality',
                'Q4_K_M': '4-bit quantization, good quality, recommended',
                'Q5_0': '5-bit quantization, better quality than q4',
                'Q5_1': '5-bit quantization, balanced 5-bit option',
                'Q5_K_S': '5-bit quantization, smaller size',
                'Q5_K_M': '5-bit quantization, good quality',
                'Q6_K': '6-bit quantization, high quality, larger size',
                'Q8_0': '8-bit quantization, highest quality, largest size',
                'F16': '16-bit float, original quality, very large',
                'BF16': '16-bit bfloat, original quality, very large'
            }
            
            # Sort files by preference
            def sort_key(filename):
                filename_lower = filename.lower()
                # Priority order: Q4_K_M, Q4_0, Q5_K_M, Q8_0, others
                if 'q4_k_m' in filename_lower:
                    return 0
                elif 'q4_0' in filename_lower:
                    return 1
                elif 'q5_k_m' in filename_lower:
                    return 2
                elif 'q8_0' in filename_lower:
                    return 3
                elif 'q4_k_s' in filename_lower:
                    return 4
                elif 'q5_k_s' in filename_lower:
                    return 5
                elif 'q3_k_m' in filename_lower:
                    return 6
                elif 'q2_k' in filename_lower:
                    return 7
                else:
                    return 8
            
            gguf_files.sort(key=sort_key)
            
            self.table.setRowCount(len(gguf_files))
            
            for i, filename in enumerate(gguf_files):
                # Extract quantization type
                parts = filename.split('.')
                quant_type = parts[-2] if len(parts) > 1 else filename
                
                # Handle cases where quant_type might be None or empty
                if not quant_type or quant_type == filename:
                    quant_type = "Unknown"
                
                # Get file size
                size_bytes = file_sizes.get(filename, 0)
                if size_bytes and size_bytes > 0:
                    size_str = self.format_size(size_bytes)
                else:
                    size_str = "Unknown"
                
                # Determine quality level
                quality = self.get_quality_level(quant_type)
                
                # Get description
                description = quant_descriptions.get(quant_type, f"{quant_type} quantization")
                
                # Create table items
                self.table.setItem(i, 0, QTableWidgetItem(quant_type))
                self.table.setItem(i, 1, QTableWidgetItem(size_str))
                self.table.setItem(i, 2, QTableWidgetItem(quality))
                self.table.setItem(i, 3, QTableWidgetItem(description))
                
                # Add helpful tooltips
                tooltip = f"<b>{quant_type}</b><br>"
                tooltip += f"<b>Size:</b> {size_str}<br>"
                tooltip += f"<b>Quality:</b> {quality}<br>"
                tooltip += f"<b>Description:</b> {description}<br><br>"
                
                # Add system-specific advice
                if 'q4_k_m' in quant_type.lower():
                    tooltip += "🎯 <b>Recommended for most systems</b><br>"
                    tooltip += "• Good balance of quality and speed<br>"
                    tooltip += "• Works well on 8GB+ RAM systems<br>"
                    tooltip += "• Best choice for general use"
                elif 'q4_0' in quant_type.lower():
                    tooltip += "⚡ <b>Fast inference option</b><br>"
                    tooltip += "• Optimized for speed<br>"
                    tooltip += "• Good for real-time applications<br>"
                    tooltip += "• Slightly lower quality than Q4_K_M"
                elif 'q8_0' in quant_type.lower():
                    tooltip += "🏆 <b>Highest quality option</b><br>"
                    tooltip += "• Best quality available<br>"
                    tooltip += "• Requires 16GB+ RAM<br>"
                    tooltip += "• Slower inference speed"
                elif 'q4_k_s' in quant_type.lower():
                    tooltip += "💾 <b>Space-efficient option</b><br>"
                    tooltip += "• Smaller file size<br>"
                    tooltip += "• Good for limited storage<br>"
                    tooltip += "• Lower quality than Q4_K_M"
                
                # Set tooltip for the entire row
                for col in range(4):
                    item = self.table.item(i, col)
                    if item:
                        item.setToolTip(tooltip)
                
                # Store filename as data
                item = self.table.item(i, 0)
                if item:
                    item.setData(Qt.ItemDataRole.UserRole, filename)
                
                # Highlight recommended options with more subtle colors
                if quant_type and 'q4_k_m' in quant_type.lower():
                    for col in range(4):
                        item = self.table.item(i, col)
                        if item:
                            # Use a very subtle green background for recommended
                            item.setBackground(QColor(40, 60, 40))  # Dark green
                            item.setForeground(QColor(200, 255, 200))  # Light green text
                            # Add a star emoji to the quantization name
                            if col == 0:
                                item.setText("⭐ " + item.text())
                elif quant_type and 'q4_0' in quant_type.lower():
                    for col in range(4):
                        item = self.table.item(i, col)
                        if item:
                            # Use a very subtle yellow background for alternative
                            item.setBackground(QColor(60, 60, 40))  # Dark yellow
                            item.setForeground(QColor(255, 255, 200))  # Light yellow text
                            # Add a checkmark emoji to the quantization name
                            if col == 0:
                                item.setText("✓ " + item.text())
                        
        except Exception as e:
            self.table.setRowCount(1)
            self.table.setItem(0, 0, QTableWidgetItem(f"Error: {e}"))
            self.ok_btn.setEnabled(False)
    
    def get_quality_level(self, quant_type):
        """Get quality level description for quantization type"""
        if not quant_type:
            return "Unknown"
        
        quant_lower = quant_type.lower()
        if 'q2' in quant_lower:
            return "Low"
        elif 'q3' in quant_lower:
            return "Low-Medium"
        elif 'q4' in quant_lower:
            return "Medium"
        elif 'q5' in quant_lower:
            return "Medium-High"
        elif 'q6' in quant_lower:
            return "High"
        elif 'q8' in quant_lower:
            return "Very High"
        elif 'f16' in quant_lower or 'bf16' in quant_lower:
            return "Original"
        else:
            return "Unknown"
    
    def format_size(self, size_bytes):
        """Format size in bytes to human readable format"""
        if size_bytes == 0:
            return "0B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f}{size_names[i]}"
    
    def update_info(self):
        """Update info panel when selection changes"""
        current_row = self.table.currentRow()
        if current_row >= 0:
            item = self.table.item(current_row, 0)
            if item:
                filename = item.data(Qt.ItemDataRole.UserRole)
                quant_type = item.text()
                size_item = self.table.item(current_row, 1)
                quality_item = self.table.item(current_row, 2)
                desc_item = self.table.item(current_row, 3)
                
                size = size_item.text() if size_item else "Unknown"
                quality = quality_item.text() if quality_item else "Unknown"
                description = desc_item.text() if desc_item else "Unknown"
                
                self.selected_file = filename
                
                info_text = f"<b>Selected:</b> {filename}<br>"
                info_text += f"<b>Quantization:</b> {quant_type}<br>"
                info_text += f"<b>Size:</b> {size}<br>"
                info_text += f"<b>Quality:</b> {quality}<br>"
                info_text += f"<b>Description:</b> {description}"
                
                self.info_label.setText(info_text)
                self.ok_btn.setEnabled(True)
        else:
            self.selected_file = None
            self.info_label.setText("Please select a quantization option")
            self.ok_btn.setEnabled(False)