"""
Authentication Manager - Handles Hugging Face Hub authentication
"""

import os
import json
import base64
from pathlib import Path
from typing import Optional, Tuple
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox, QTextEdit
from PyQt6.QtCore import Qt


class HuggingFaceAuthManager:
    """Manages Hugging Face Hub authentication"""
    
    def __init__(self):
        self.config_dir = Path.home() / ".huggingdrive"
        self.config_file = self.config_dir / "auth_config.json"
        self.token = None
        self.username = None
        self._load_auth_config()
    
    def _load_auth_config(self):
        """Load authentication configuration from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.token = config.get('token')
                    self.username = config.get('username')
                    
                    # Set environment variable for huggingface_hub
                    if self.token:
                        os.environ['HUGGING_FACE_HUB_TOKEN'] = self.token
        except Exception as e:
            print(f"Error loading auth config: {e}")
    
    def _save_auth_config(self):
        """Save authentication configuration to file"""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            config = {
                'token': self.token,
                'username': self.username
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Error saving auth config: {e}")
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        return self.token is not None
    
    def get_token(self) -> Optional[str]:
        """Get the current authentication token"""
        return self.token
    
    def get_username(self) -> Optional[str]:
        """Get the current username"""
        return self.username
    
    def login(self, token: str) -> Tuple[bool, str]:
        """Login with a Hugging Face token"""
        try:
            # Validate token by trying to get user info
            from huggingface_hub import HfApi
            api = HfApi(token=token)
            user_info = api.whoami()
            
            # Token is valid, save it
            self.token = token
            self.username = user_info.get('name', 'Unknown')
            self._save_auth_config()
            
            # Set environment variable
            os.environ['HUGGING_FACE_HUB_TOKEN'] = token
            
            return True, f"Successfully logged in as {self.username}"
            
        except Exception as e:
            return False, f"Login failed: {str(e)}"
    
    def logout(self):
        """Logout and clear authentication"""
        self.token = None
        self.username = None
        
        # Remove environment variable
        if 'HUGGING_FACE_HUB_TOKEN' in os.environ:
            del os.environ['HUGGING_FACE_HUB_TOKEN']
        
        # Clear config file
        try:
            if self.config_file.exists():
                self.config_file.unlink()
        except Exception as e:
            print(f"Error clearing auth config: {e}")
    
    def accept_model_terms(self, model_name: str) -> Tuple[bool, str]:
        """Accept terms of use for a specific model"""
        if not self.is_authenticated():
            return False, "Not authenticated"
        
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=self.token)
            
            # Accept terms for the model
            api.accept_terms(model_name)
            return True, f"Accepted terms for {model_name}"
            
        except Exception as e:
            return False, f"Failed to accept terms: {str(e)}"


class LoginDialog(QDialog):
    """Dialog for entering Hugging Face token"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.token = None
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Login to Hugging Face Hub")
        self.setModal(True)
        self.setFixedSize(500, 400)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("🔑 Login to Hugging Face Hub")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # Instructions
        instructions = QTextEdit()
        instructions.setPlainText(
            "To access restricted models (like Meta Llama, Microsoft Phi, etc.), "
            "you need to log in to Hugging Face Hub.\n\n"
            "1. Go to https://huggingface.co/settings/tokens\n"
            "2. Click 'New token'\n"
            "3. Give it a name (e.g., 'HuggingDrive')\n"
            "4. Select 'Read' permissions\n"
            "5. Copy the generated token\n"
            "6. Paste it below\n\n"
            "Your token will be stored securely and used for all downloads."
        )
        instructions.setReadOnly(True)
        instructions.setMaximumHeight(200)
        layout.addWidget(instructions)
        
        # Token input
        token_label = QLabel("Hugging Face Token:")
        layout.addWidget(token_label)
        
        self.token_input = QLineEdit()
        self.token_input.setPlaceholderText("hf_...")
        self.token_input.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.token_input)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.login_btn = QPushButton("Login")
        self.login_btn.clicked.connect(self.accept)
        self.login_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setStyleSheet("background-color: #f44336; color: white; padding: 8px;")
        
        button_layout.addWidget(self.login_btn)
        button_layout.addWidget(self.cancel_btn)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def get_token(self) -> Optional[str]:
        """Get the entered token"""
        return self.token_input.text().strip() if self.token_input.text().strip() else None 