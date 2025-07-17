"""
Model Test Dialog - For testing downloaded models
"""

import sys
import os
import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QTextEdit,
    QComboBox,
    QProgressBar,
    QMessageBox,
    QTextBrowser,
    QTabWidget,
    QWidget,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

# Import Gradio manager
try:
    from .gradio_manager import GradioInterfaceManager

    GRADIO_AVAILABLE = True
    print("DEBUG: Gradio available")
except ImportError as e:
    GRADIO_AVAILABLE = False
    print(f"DEBUG: Gradio import failed: {e}")

# Force button to show for testing
GRADIO_AVAILABLE = True

from .loader import ModelLoadThread


class ModelTestDialog(QDialog):
    """Dialog for testing downloaded models"""

    def __init__(self, model_name: str, model_path: str, parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.model_path = model_path
        self.pipeline = None
        self.model_type = None
        self.load_thread = None
        self.gradio_manager = None

        # Initialize Gradio manager if available
        if GRADIO_AVAILABLE:
            try:
                self.gradio_manager = GradioInterfaceManager()
            except Exception as e:
                print(f"Failed to initialize Gradio manager: {e}")
                self.gradio_manager = None

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle(f"Test Model: {self.model_name}")
        self.setGeometry(200, 200, 800, 600)

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Model path display
        path_label = QLabel(f"Model Path: {self.model_path}")
        path_label.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(path_label)

        # Tab widget for different test types
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

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
        title_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #FF5722; margin: 20px;"
        )
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        # Description
        desc_label = QLabel(
            "Launch a web interface to interact with your model through a browser."
        )
        desc_label.setStyleSheet("font-size: 14px; color: #666; margin: 10px;")
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc_label)

        # Launch button
        self.web_interface_btn = QPushButton("🌐 Launch Web Interface")
        self.web_interface_btn.clicked.connect(self.launch_web_interface)
        self.web_interface_btn.setStyleSheet(
            """
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
        """
        )
        layout.addWidget(self.web_interface_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        # Status
        self.web_status_label = QLabel("Ready to launch web interface")
        self.web_status_label.setStyleSheet(
            "color: green; font-size: 12px; margin: 10px;"
        )
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
        self.chat_capabilities_label = QLabel(
            "Model capabilities will be shown here when a model is loaded"
        )
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

        # Model information
        info_text = f"""
        <h3>Model Information</h3>
        <p><strong>Name:</strong> {self.model_name}</p>
        <p><strong>Path:</strong> {self.model_path}</p>
        <p><strong>Type:</strong> <span id="model-type">Not loaded</span></p>
        <p><strong>Status:</strong> <span id="model-status">Not loaded</span></p>
        
        <h3>Usage Instructions</h3>
        <ul>
        <li>Click "Load Model" to load the model into memory</li>
        <li>Use the tabs above to test different model capabilities</li>
        <li>Text Generation: Generate text continuations</li>
        <li>Text Classification: Classify text into categories</li>
        <li>🌐 Web Interface: Launch a web interface for the model</li>
        <li>Chat: Interactive conversation with the model</li>
        </ul>
        
        <h3>Memory Requirements</h3>
        <p>Model loading requires sufficient RAM. The system will check available memory before loading.</p>
        """

        info_browser = QTextBrowser()
        info_browser.setHtml(info_text)
        layout.addWidget(info_browser)

        self.tab_widget.addTab(tab, "Model Info")

    def load_model(self):
        """Load the model in a background thread"""
        if self.load_thread and self.load_thread.isRunning():
            QMessageBox.warning(self, "Loading", "Model is already being loaded")
            return

        # Start loading thread
        self.load_thread = ModelLoadThread(self.model_path, self.model_name)
        self.load_thread.model_loaded_signal.connect(self.on_model_loaded)
        self.load_thread.loading_progress_signal.connect(self.on_loading_progress)
        self.load_thread.progress_bar_signal.connect(self.on_progress_update)
        self.load_thread.memory_check_signal.connect(self.on_memory_check)
        self.load_thread.error_signal.connect(self.on_model_load_error)

        self.load_thread.start()

        # Update UI
        self.load_btn.setEnabled(False)
        self.abort_btn.setEnabled(True)
        self.status_label.setText("Loading model...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress

    def on_model_loaded(self, pipeline_obj, model_type):
        """Handle model loaded signal"""
        self.pipeline = pipeline_obj
        self.model_type = model_type

        # Update UI
        self.load_btn.setEnabled(False)
        self.abort_btn.setEnabled(False)
        self.unload_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"✅ Model loaded successfully ({model_type})")

        # Enable appropriate buttons based on model type
        if "text-generation" in model_type or "gpt" in model_type.lower():
            self.generate_btn.setEnabled(True)
            self.send_btn.setEnabled(True)
            self.status_label.setText("✅ Text generation model loaded")
        elif "text-classification" in model_type or "bert" in model_type.lower():
            self.classify_btn.setEnabled(True)
            self.status_label.setText("✅ Text classification model loaded")
        elif "translation" in model_type or "marian" in model_type:
            self.status_label.setText("✅ Translation model loaded (use web interface)")
        else:
            # Generic model - enable all
            self.generate_btn.setEnabled(True)
            self.classify_btn.setEnabled(True)
            self.send_btn.setEnabled(True)
            self.status_label.setText("✅ Model loaded (all features available)")

        # Update model info tab
        self.update_model_info()

    def on_loading_progress(self, message):
        """Handle loading progress updates"""
        self.status_label.setText(message)

    def on_progress_update(self, value):
        """Handle progress bar updates"""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(value)

    def on_memory_check(self, is_sufficient: bool, message: str):
        """Handle memory check results"""
        self.memory_info_label.setText(message)
        if not is_sufficient:
            QMessageBox.warning(self, "Memory Warning", message)

    def abort_loading(self):
        """Abort the model loading process"""
        if self.load_thread and self.load_thread.isRunning():
            self.load_thread.abort_loading()
            self.load_thread.wait()

        # Reset UI
        self.load_btn.setEnabled(True)
        self.abort_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Model loading aborted")

    def on_model_load_error(self, error_message):
        """Handle model loading errors"""
        QMessageBox.critical(
            self, "Loading Error", f"Failed to load model: {error_message}"
        )

        # Reset UI
        self.load_btn.setEnabled(True)
        self.abort_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("❌ Model loading failed")

    def unload_model(self):
        """Unload the model and free memory"""
        if self.pipeline:
            # Clear pipeline
            self.pipeline = None
            self.model_type = None

            # Reset UI
            self.load_btn.setEnabled(True)
            self.abort_btn.setEnabled(False)
            self.unload_btn.setEnabled(False)
            self.progress_bar.setVisible(False)

            # Disable all buttons
            self.generate_btn.setEnabled(False)
            self.classify_btn.setEnabled(False)
            self.send_btn.setEnabled(False)

            self.status_label.setText("Model unloaded")
            self.memory_info_label.setText("")

            # Update model info
            self.update_model_info()

            # Clear outputs
            self.text_output.clear()
            self.classify_output.clear()
            self.chat_history.clear()

            # Also unload from Gradio interface if it's loaded there
            if self.gradio_manager and self.gradio_manager.is_model_loaded(
                self.model_name
            ):
                try:
                    self.gradio_manager.stop_interface(self.model_name)
                    self.status_label.setText(
                        "Model unloaded (including Gradio interface)"
                    )
                except Exception as e:
                    print(f"Warning: Could not unload model from Gradio: {e}")

            QMessageBox.information(
                self, "Model Unloaded", "Model has been unloaded and memory freed"
            )

    def check_memory_usage(self):
        """Check current memory usage"""
        try:
            import psutil

            memory = psutil.virtual_memory()
            used_gb = memory.used / (1024**3)
            total_gb = memory.total / (1024**3)
            percent = memory.percent

            self.memory_info_label.setText(
                f"Memory: {used_gb:.1f}GB/{total_gb:.1f}GB ({percent:.0f}%)"
            )
        except:
            self.memory_info_label.setText("Memory: Loading...")

    def show_model_type_warning(self, model_type: str):
        """Show warning for unsupported model types"""
        warning_text = f"""
        <h3>Model Type Warning</h3>
        <p>This model appears to be a <strong>{model_type}</strong> model.</p>
        <p>Some features may not work as expected:</p>
        <ul>
        <li>Text generation models work best with text generation features</li>
        <li>Classification models work best with classification features</li>
        <li>Translation models work best with translation features</li>
        </ul>
        <p>You can still try all features, but results may vary.</p>
        """

        QMessageBox.information(self, "Model Type Info", warning_text)

    def refresh_memory_info(self):
        """Refresh memory information display"""
        self.check_memory_usage()

    def update_model_info(self):
        """Update the model information tab"""
        if self.model_type:
            # Update model type in info tab
            info_text = f"""
            <h3>Model Information</h3>
            <p><strong>Name:</strong> {self.model_name}</p>
            <p><strong>Path:</strong> {self.model_path}</p>
            <p><strong>Type:</strong> {self.model_type}</p>
            <p><strong>Status:</strong> ✅ Loaded and ready</p>
            
            <h3>Available Features</h3>
            <ul>
            <li>Text Generation: {'✅ Enabled' if self.generate_btn.isEnabled() else '❌ Disabled'}</li>
            <li>Text Classification: {'✅ Enabled' if self.classify_btn.isEnabled() else '❌ Disabled'}</li>
            <li>🌐 Web Interface: ✅ Available</li>
            <li>Chat: {'✅ Enabled' if self.send_btn.isEnabled() else '❌ Disabled'}</li>
            </ul>
            """
        else:
            info_text = f"""
            <h3>Model Information</h3>
            <p><strong>Name:</strong> {self.model_name}</p>
            <p><strong>Path:</strong> {self.model_path}</p>
            <p><strong>Type:</strong> Not loaded</p>
            <p><strong>Status:</strong> Not loaded</p>
            """

        # Find the info browser in the Model Info tab
        info_tab = self.tab_widget.widget(4)  # Model Info is the 5th tab (index 4)
        if info_tab:
            info_browser = info_tab.findChild(QTextBrowser)
            if info_browser:
                info_browser.setHtml(info_text)

    def generate_text(self):
        """Generate text using the loaded model"""
        try:
            input_text = self.text_input.toPlainText().strip()
            if not input_text:
                self.text_output.setText("Please enter some text to continue")
                return

            max_length = int(self.max_length_spin.currentText())
            temperature = float(self.temperature_spin.currentText())

            result = self.pipeline(
                input_text, max_new_tokens=max_length, temperature=temperature
            )

            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0]["generated_text"]
                self.text_output.setText(generated_text)
            else:
                self.text_output.setText("No text generated")

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
                # Format classification results
                output = "Classification Results:\n\n"
                for item in result:
                    label = item.get("label", "Unknown")
                    score = item.get("score", 0)
                    output += f"• {label}: {score:.3f}\n"
                self.classify_output.setText(output)
            else:
                self.classify_output.setText("No classification results")

        except Exception as e:
            self.classify_output.setText(f"Error classifying text: {str(e)}")

    def translate_text(self):
        """Translation moved to web interface"""
        QMessageBox.information(
            self,
            "Translation",
            "Translation features are now available in the 🌐 Web Interface tab!",
        )

    def send_chat_message(self):
        """Send a chat message to the model"""
        try:
            message = self.chat_input.text().strip()
            if not message:
                return

            # Add user message to chat history
            self.chat_history.append(f"<b>You:</b> {message}")
            self.chat_input.clear()

            # Get chat parameters
            max_length = int(self.chat_max_length.currentText())
            temperature = float(self.chat_temperature.currentText())

            # Get conversation context
            context = self.get_conversation_context()

            # Generate response
            if context:
                full_prompt = context + "\n\nUser: " + message + "\nAssistant:"
            else:
                full_prompt = f"User: {message}\nAssistant:"

            result = self.pipeline(
                full_prompt, max_new_tokens=max_length, temperature=temperature
            )

            if isinstance(result, list) and len(result) > 0:
                response = result[0]["generated_text"]

                # Extract just the assistant's response
                if "Assistant:" in response:
                    response = response.split("Assistant:")[-1].strip()

                # Clean up repetitive responses
                response = self.clean_repetitive_response(response)

                # Add assistant response to chat history
                self.chat_history.append(f"<b>Assistant:</b> {response}")

                # Scroll to bottom
                self.chat_history.verticalScrollBar().setValue(
                    self.chat_history.verticalScrollBar().maximum()
                )
            else:
                self.chat_history.append(
                    "<b>Assistant:</b> Sorry, I couldn't generate a response."
                )

        except Exception as e:
            self.chat_history.append(f"<b>Error:</b> {str(e)}")

    def clean_repetitive_response(self, response: str) -> str:
        """Clean up repetitive or nonsensical responses"""
        # Remove common repetitive patterns
        lines = response.split("\n")
        cleaned_lines = []
        seen_lines = set()

        for line in lines:
            line = line.strip()
            if line and line not in seen_lines:
                cleaned_lines.append(line)
                seen_lines.add(line)

        # Limit response length
        cleaned_response = "\n".join(cleaned_lines)
        if len(cleaned_response) > 1000:
            cleaned_response = cleaned_response[:1000] + "..."

        return cleaned_response

    def get_conversation_context(self):
        """Get recent conversation context for better responses"""
        try:
            # Get the last few messages from chat history
            plain_text = self.chat_history.toPlainText()
            lines = plain_text.split("\n")

            # Get last 6 lines (3 exchanges)
            recent_lines = lines[-6:] if len(lines) > 6 else lines

            context = "\n".join(recent_lines)
            return context
        except:
            return ""

    def clear_chat_history(self):
        """Clear the chat history"""
        self.chat_history.clear()

    def save_chat_history(self):
        """Save chat history to a file"""
        try:
            from PyQt6.QtWidgets import QFileDialog

            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Chat History",
                f"chat_{self.model_name}.txt",
                "Text Files (*.txt)",
            )

            if file_path:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(self.chat_history.toPlainText())
                QMessageBox.information(
                    self, "Success", f"Chat history saved to {file_path}"
                )
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to save chat history: {str(e)}"
            )

    def load_chat_history(self):
        """Load chat history from a file"""
        try:
            from PyQt6.QtWidgets import QFileDialog

            file_path, _ = QFileDialog.getOpenFileName(
                self, "Load Chat History", "", "Text Files (*.txt)"
            )

            if file_path:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.chat_history.setPlainText(content)
                QMessageBox.information(
                    self, "Success", f"Chat history loaded from {file_path}"
                )
        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to load chat history: {str(e)}"
            )

    def save_chat_history_to_cache(self):
        """Save chat history to cache for persistence"""
        try:
            cache_dir = Path.home() / ".huggingdrive" / "chat_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)

            cache_file = cache_dir / f"{self.model_name}_chat.txt"
            with open(cache_file, "w", encoding="utf-8") as f:
                f.write(self.chat_history.toPlainText())
        except:
            pass  # Ignore cache errors

    def load_chat_history_from_cache(self):
        """Load chat history from cache"""
        try:
            cache_dir = Path.home() / ".huggingdrive" / "chat_cache"
            cache_file = cache_dir / f"{self.model_name}_chat.txt"

            if cache_file.exists():
                with open(cache_file, "r", encoding="utf-8") as f:
                    content = f.read()
                self.chat_history.setPlainText(content)
        except:
            pass  # Ignore cache errors

    def launch_web_interface(self):
        """Launch Gradio web interface for the model"""
        if not self.gradio_manager:
            QMessageBox.warning(
                self,
                "Gradio Not Available",
                "Gradio is not available. Please install it with: pip install gradio",
            )
            return

        if not self.pipeline:
            QMessageBox.warning(
                self,
                "Model Not Loaded",
                "Please load the model first before launching the web interface.",
            )
            return

        try:
            # Launch web interface
            self.gradio_manager.launch_interface(self.model_path, self.model_name)
            self.web_status_label.setText("🌐 Web Interface Active")
            self.web_interface_btn.setText("🌐 Web Interface Active")
            self.web_interface_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    font-weight: bold;
                    font-size: 16px;
                    padding: 20px;
                    border: 3px solid #45a049;
                    border-radius: 10px;
                    min-width: 300px;
                    min-height: 60px;
                    margin: 20px;
                }
            """
            )

            QMessageBox.information(
                self,
                "Web Interface Launched",
                "Gradio web interface has been launched!\n\n"
                "You can now interact with your model through a web browser.",
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to launch web interface: {str(e)}"
            )

    def closeEvent(self, event):
        """Handle dialog close event"""
        # Save chat history to cache
        self.save_chat_history_to_cache()

        # Stop any active Gradio interfaces for this model
        if self.gradio_manager:
            try:
                self.gradio_manager.stop_interface(self.model_name)
            except Exception as e:
                print(
                    f"Warning: Could not stop Gradio interface for {self.model_name}: {e}"
                )

        # Close dialog
        event.accept()
