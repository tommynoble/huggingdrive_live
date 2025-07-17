"""
Model Search - Handles searching for models on Hugging Face Hub
"""

import json
import os
import time
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal


# Model categories for filtering
MODEL_CATEGORIES = {
    "text": ["text-generation", "text-classification", "question-answering", "summarization"],
    "vision": ["image-classification", "object-detection", "image-segmentation"],
    "audio": ["automatic-speech-recognition", "audio-classification", "text-to-speech"],
    "multimodal": ["image-to-text", "text-to-image", "visual-question-answering"]
}


class ModelSearchThread(QThread):
    """Thread for searching models without blocking the GUI"""
    results_signal = pyqtSignal(list)
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(str)
    
    def __init__(self, query: str, limit: int = 10, category: str = None, sort_by: str = "downloads", use_actual_sizes: bool = False):
        super().__init__()
        self.query = query
        self.limit = limit
        self.category = category
        self.sort_by = sort_by
        self.use_actual_sizes = use_actual_sizes
        
        # Set up cache directory
        self.cache_dir = Path.home() / ".huggingdrive_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.size_cache_file = self.cache_dir / "model_sizes.json"
        self.size_cache = self.load_size_cache()
    
    def load_size_cache(self) -> dict:
        """Load cached model sizes"""
        try:
            if self.size_cache_file.exists():
                with open(self.size_cache_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}
    
    def save_size_cache(self):
        """Save cached model sizes"""
        try:
            with open(self.size_cache_file, 'w') as f:
                json.dump(self.size_cache, f, indent=2)
        except Exception:
            pass
    
    def get_cached_size(self, model_id: str) -> str:
        """Get cached size for a model"""
        return self.size_cache.get(model_id, None)
    
    def cache_size(self, model_id: str, size: str):
        """Cache size for a model"""
        self.size_cache[model_id] = size
    
    def run(self):
        try:
            self.progress_signal.emit("Searching Hugging Face Hub for models...")
            
            from huggingface_hub import HfApi
            api = HfApi()
            
            # Build search parameters - search for any models
            search_params = {
                "limit": self.limit,
                "full": False  # Don't fetch full model info for performance
            }
            
            # Add search query if provided
            if self.query and self.query.strip():
                search_params["search"] = self.query.strip()
            
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
            
            # Use all models found (no GGUF filtering)
            # This allows users to find any model and convert to GGUF if needed
            
            # Sort models by downloads first, then by size
            if self.sort_by == "downloads":
                models.sort(key=lambda x: getattr(x, 'downloads', 0) or 0, reverse=True)
            elif self.sort_by == "likes":
                models.sort(key=lambda x: getattr(x, 'likes', 0) or 0, reverse=True)
            
            # Format results with size information
            results = []
            for i, model in enumerate(models):
                self.progress_signal.emit(f"Processing model {i+1}/{len(models)}: {model.modelId}")
                
                # Get size information
                if self.use_actual_sizes:
                    # Check cache first
                    cached_size = self.get_cached_size(model.modelId)
                    if cached_size:
                        size_info = cached_size
                    else:
                        # Fetch actual size
                        size_info = self.get_actual_model_size(model.modelId)
                        if size_info != "Unknown":
                            self.cache_size(model.modelId, size_info)
                else:
                    # Use estimated size for speed
                    size_info = self.estimate_model_size(model.modelId, getattr(model, 'tags', []))
                
                # Check if model might require authentication
                model_id_lower = getattr(model, 'modelId', '').lower()
                requires_auth = self.check_if_likely_requires_auth(model_id_lower)
                
                results.append({
                    "name": model.modelId,
                    "size": size_info,
                    "downloads": getattr(model, 'downloads', 0) or 0,
                    "likes": getattr(model, 'likes', 0) or 0,
                    "tags": getattr(model, 'tags', []),
                    "requires_auth": requires_auth,
                    "size_type": "actual" if self.use_actual_sizes else "estimated",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
            
            # Save cache if we fetched actual sizes
            if self.use_actual_sizes:
                self.save_size_cache()
            
            self.progress_signal.emit(f"Found {len(results)} models")
            self.results_signal.emit(results)
            
        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            self.error_signal.emit(error_msg)
    
    def check_if_likely_requires_auth(self, model_id: str) -> bool:
        """Check if a model is likely to require authentication based on its name"""
        model_lower = model_id.lower()
        
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

    def estimate_model_size(self, model_id: str, tags: list) -> str:
        """Estimate model size based on model type and tags"""
        tags_lower = [tag.lower() for tag in tags]
        
        # Check for specific model types in the name
        model_lower = model_id.lower()
        
        # Large language models
        if any(large_model in model_lower for large_model in ["llama-2-70b", "llama-2-13b", "mistral-7b", "falcon-40b", "bloom-176b"]):
            return "13GB+"
        elif any(large_model in model_lower for large_model in ["llama-2-7b", "mistral-7b", "falcon-7b", "bloom-7b"]):
            return "7GB"
        elif any(large_model in model_lower for large_model in ["llama-2-3b", "mistral-3b", "falcon-3b", "bloom-3b"]):
            return "3GB"
        elif any(large_model in model_lower for large_model in ["llama-2-1b", "mistral-1b", "falcon-1b", "bloom-1b"]):
            return "1GB"
        
        # Check for specific architectures
        if "gpt2" in model_lower:
            if "large" in model_lower:
                return "1.5GB"
            elif "medium" in model_lower:
                return "800MB"
            elif "small" in model_lower:
                return "500MB"
            else:
                return "500MB"  # Default GPT-2
        
        # Check for DialoGPT models specifically
        if "dialogpt" in model_lower:
            if "large" in model_lower:
                return "5.2GB"  # Actual size ~5.2GB
            elif "medium" in model_lower:
                return "5.2GB"  # Actual size ~5.2GB (includes all formats)
            elif "small" in model_lower:
                return "1.5GB"  # Actual size ~1.5GB
            else:
                return "5.2GB"  # Default DialoGPT
        
        if "bert" in model_lower:
            if "large" in model_lower:
                return "1.3GB"
            elif "base" in model_lower:
                return "500MB"
            else:
                return "500MB"  # Default BERT
        
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
    
    def get_actual_model_size(self, model_id: str) -> str:
        """Get the actual size of a model from Hugging Face Hub"""
        try:
            from huggingface_hub import model_info
            info = model_info(model_id)
            
            # Try to get size from model files
            total_size = 0
            
            # Check for safetensors files
            if hasattr(info, 'safetensors') and info.safetensors:
                for file_info in info.safetensors.values():
                    if isinstance(file_info, dict) and 'size' in file_info:
                        total_size += file_info['size']
            
            # Check for pytorch model files
            if hasattr(info, 'pytorch_model') and info.pytorch_model:
                for file_info in info.pytorch_model.values():
                    if isinstance(file_info, dict) and 'size' in file_info:
                        total_size += file_info['size']
            
            # Check for other model files
            if hasattr(info, 'model_files') and info.model_files:
                for file_info in info.model_files:
                    if isinstance(file_info, dict) and 'size' in file_info:
                        total_size += file_info['size']
            
            if total_size > 0:
                return self.format_size(total_size)
            else:
                return "Unknown"
                
        except Exception as e:
            print(f"Error getting size for {model_id}: {e}")
            return "Unknown" 