"""
Drive Manager - Handles external drive detection and model management
"""

import json
import shutil
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import psutil
import os


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
        drive_path_obj = Path(drive_path)
        
        # Look for models in the default location
        default_models_dir = drive_path_obj / "huggingface_models"
        
        if default_models_dir.exists():
            for model_dir in default_models_dir.iterdir():
                if model_dir.is_dir():
                    # Check if it contains model files
                    if self.is_valid_model_directory(model_dir):
                        model_name = model_dir.name.replace("_", "/")  # Convert back from filesystem-safe name
                        # Calculate size for the model
                        size_str = self.calculate_model_size(str(model_dir))
                        
                        models.append({
                            "name": model_name,
                            "path": str(model_dir),
                            "drive": str(drive_path_obj),
                            "detected_at": str(Path().cwd()),
                            "source": "auto-detected",
                            "size": size_str
                        })
        
        return models
    
    def is_valid_model_directory(self, model_dir: Path) -> bool:
        """Check if a directory contains a valid Hugging Face model or GGUF file"""
        # Check for GGUF files first
        if isinstance(model_dir, Path):
            # If the path is a file and ends with .gguf, it's valid
            if model_dir.is_file() and model_dir.suffix.lower() == '.gguf':
                return True
            # Check directory for .gguf files
            for item in model_dir.glob('**/*.gguf'):
                if item.is_file():
                    return True
        
        # Check for common model files (Hugging Face format)
        model_files = [
            "config.json",
            "pytorch_model.bin",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "merges.txt",
            "special_tokens_map.json",
            "tokenizer.model"
        ]
        
        # If any of these files exist, consider it a valid model
        for file_name in model_files:
            if (model_dir / file_name).exists():
                return True
        
        # Also check for subdirectories that might contain model files
        for subdir in model_dir.iterdir():
            if subdir.is_dir():
                for file_name in model_files:
                    if (subdir / file_name).exists():
                        return True
        
        # Check for model shards (split model files)
        if any(model_dir.glob("pytorch_model-*.bin")) or any(model_dir.glob("model-*.safetensors")):
                        return True
        
        return False
    
    def scan_custom_folder_for_models(self, folder_path: str) -> List[Dict]:
        """Scan a custom folder for models"""
        models = []
        folder_path_obj = Path(folder_path)
        
        if folder_path_obj.exists() and folder_path_obj.is_dir():
            for model_dir in folder_path_obj.iterdir():
                if model_dir.is_dir():
                    if self.is_valid_model_directory(model_dir):
                        model_name = model_dir.name.replace("_", "/")  # Convert back from filesystem-safe name
                        # Calculate size for the model
                        size_str = self.calculate_model_size(str(model_dir))
                        
                        models.append({
                            "name": model_name,
                            "path": str(model_dir),
                            "drive": str(folder_path_obj.parent),
                            "detected_at": str(Path().cwd()),
                            "source": "custom-folder",
                            "size": size_str
                        })
        
        return models
    
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
        from huggingface_hub import snapshot_download
        
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
    
    def calculate_model_size(self, model_path: str) -> str:
        """Calculate the size of a model directory"""
        try:
            import os
            from pathlib import Path
            
            path = Path(model_path)
            if not path.exists():
                return "Unknown"
            
            total_size = 0
            file_count = 0
            
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        total_size += os.path.getsize(file_path)
                        file_count += 1
                    except (OSError, FileNotFoundError):
                        continue
            
            if total_size == 0:
                return "Unknown"
            
            # Format size
            if total_size >= 1024**3:
                return f"{total_size/(1024**3):.1f}GB"
            elif total_size >= 1024**2:
                return f"{total_size/(1024**2):.1f}MB"
            elif total_size >= 1024:
                return f"{total_size/1024:.1f}KB"
            else:
                return f"{total_size}B"
                
        except Exception as e:
            print(f"Error calculating model size for {model_path}: {e}")
            return "Unknown"
    
    def remove_model(self, model_name: str) -> bool:
        """Remove an installed model"""
        if model_name in self.config["installed_models"]:
            model_path = self.config["installed_models"][model_name]["path"]
            try:
                # Check if the path exists before trying to delete
                if os.path.exists(model_path):
                    print(f"Removing model directory: {model_path}")
                    
                    # Use robust deletion method that handles macOS hidden files
                    self._robust_remove_directory(model_path)
                    
                    # Verify deletion
                    if os.path.exists(model_path):
                        print(f"Warning: Model directory still exists after deletion: {model_path}")
                        return False
                    else:
                        print(f"Successfully deleted model directory: {model_path}")
                else:
                    print(f"Model path does not exist (may have been deleted manually): {model_path}")
                
                # Remove from config regardless of whether files existed
                del self.config["installed_models"][model_name]
                self.save_config()
                print(f"Removed model '{model_name}' from configuration")
                return True
                
            except Exception as e:
                print(f"Error removing model '{model_name}': {e}")
                # Still remove from config even if file deletion failed
                try:
                    del self.config["installed_models"][model_name]
                    self.save_config()
                    print(f"Removed model '{model_name}' from configuration despite file deletion error")
                    return True
                except Exception as config_error:
                    print(f"Error updating configuration: {config_error}")
                    return False
        else:
            print(f"Model '{model_name}' not found in configuration")
            return False
    
    def _robust_remove_directory(self, path: str):
        """Robustly remove directory and all contents, handling macOS hidden files"""
        import stat
        import subprocess
        import sys
        
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
                print(f"Warning: Could not remove hidden files: {e}")
        
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
                        print(f"Warning: Could not remove file {file_path}: {e}")
            
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
                        print(f"Warning: Could not remove directory {dir_path}: {e}")
        
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
                print(f"Warning: Could not remove root directory {path}: {e}") 