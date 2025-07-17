"""
Model Loader - Handles loading models for testing and interaction
"""

import time
import json
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal
import os


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
        
        # Increase timeout for large models
        if any(large_model in model_name.lower() for large_model in ["llama", "mistral", "falcon", "bloom", "gpt-neo"]):
            self.timeout_seconds = 600  # 10 minutes for large models
        else:
            self.timeout_seconds = 300  # 5 minutes for regular models
        
    def abort_loading(self):
        """Allow user to abort the loading process"""
        self.should_abort = True
        
    def estimate_model_size(self, model_path: str) -> float:
        """Estimate model size in GB"""
        try:
            total_size = 0.0  # Initialize as float
            path = Path(model_path)
            
            # For GGUF files, just get the file size
            if path.is_file() and path.suffix.lower() == '.gguf':
                return float(path.stat().st_size) / (1024**3)
            
            # For directories, sum up all file sizes
            if path.is_dir():
                for file_path in path.rglob('*'):
                    if file_path.is_file():
                        total_size += float(file_path.stat().st_size)
            
            return total_size / (1024**3)  # Convert to GB
            
        except Exception as e:
            self.loading_progress_signal.emit(f"Warning: Could not estimate model size: {e}")
            return 0.0  # Return as float
    
    def check_memory_requirements(self, model_type: str, model_size_gb: float = 0.0) -> tuple[bool, str]:
        """Check if system has sufficient memory for the model"""
        try:
            import psutil
            import torch
            
            # Get system memory info
            ram = psutil.virtual_memory()
            ram_gb = ram.total / (1024**3)
            ram_available_gb = ram.available / (1024**3)
            
            # Enhanced memory requirements for large models
            if model_size_gb > 0.0:
                # For large models, we need more memory for loading and inference
                if model_size_gb > 10.0:  # Very large models (like Llama 8B+)
                    min_ram_gb = model_size_gb * 3.0  # Need 3x model size for very large models
                elif model_size_gb > 5.0:  # Large models
                    min_ram_gb = model_size_gb * 2.5  # Need 2.5x model size for large models
                else:
                    min_ram_gb = model_size_gb * 2.0  # Standard 2x model size
            else:
                # Default requirements based on model type
                if model_type in ["llama", "mistral", "falcon", "bloom"]:
                    min_ram_gb = 16.0  # Large language models need significant RAM
                elif model_type in ["gpt2", "gpt", "causal_lm"]:
                    min_ram_gb = 4.0
                elif model_type in ["bert", "roberta", "distilbert"]:
                    min_ram_gb = 2.0
                elif "translation" in model_type or "marian" in model_type:
                    min_ram_gb = 3.0
                elif model_type in ["t5", "bart"]:
                    min_ram_gb = 4.0
                elif model_type == "gguf":
                    min_ram_gb = 2.0  # GGUF models are optimized for low memory usage
                else:
                    min_ram_gb = 3.0  # Default for unknown types
            
            ram_sufficient = ram_available_gb >= min_ram_gb
            
            if ram_sufficient:
                message = f"✅ Sufficient memory available ({ram_available_gb:.1f}GB free, need {min_ram_gb:.1f}GB)"
            else:
                message = f"⚠️ Low memory warning: {ram_available_gb:.1f}GB available, need {min_ram_gb:.1f}GB"
                if ram_gb < min_ram_gb:
                    message += f"\nTotal system RAM ({ram_gb:.1f}GB) is less than recommended ({min_ram_gb:.1f}GB)"
            
            return ram_sufficient, message
            
        except Exception as e:
            return True, f"Could not check memory requirements: {e}"
    
    def run(self):
        try:
            start_time = time.time()
            import torch
            import shutil
            
            self.progress_bar_signal.emit(5)
            self.loading_progress_signal.emit("Analyzing model...")
            
            if self.should_abort:
                return
            
            # Check if this is a GGUF file or a directory containing a GGUF file
            model_path = Path(self.model_path)
            
            # Case 1: Direct GGUF file
            if model_path.is_file() and model_path.suffix.lower() == '.gguf':
                gguf_path = model_path
                is_gguf = True
                self.loading_progress_signal.emit(f"Loading GGUF file directly: {gguf_path}")
            # Case 2: Directory containing a GGUF file
            elif model_path.is_dir():
                # First check for config.json (Hugging Face format)
                if (model_path / "config.json").exists():
                    is_gguf = False
                else:
                    # No config.json, look for GGUF files recursively
                    gguf_files = [f for f in model_path.rglob("*.gguf") if f.is_file()]
                    if gguf_files:
                        # Pick the largest GGUF file (by size)
                        gguf_path = max(gguf_files, key=lambda f: f.stat().st_size)
                        self.loading_progress_signal.emit(f"Selected largest GGUF file: {gguf_path} ({gguf_path.stat().st_size / (1024**3):.2f} GB)")
                        is_gguf = True
                    else:
                        is_gguf = False
                        self.error_signal.emit(f"No GGUF files found in directory: {model_path}")
                        return
            else:
                self.error_signal.emit(f"Model path must be either a .gguf file or a directory containing a GGUF file or config.json. Got: {model_path}")
                return
            
            if is_gguf:
                self.loading_progress_signal.emit("GGUF model detected")
                self.progress_bar_signal.emit(15)
                
                # Check memory requirements
                model_size_gb = float(gguf_path.stat().st_size) / (1024**3)
                if model_size_gb > 0:
                    self.loading_progress_signal.emit(f"Model size: {model_size_gb:.2f}GB")
                
                self.progress_bar_signal.emit(25)
                self.loading_progress_signal.emit("Checking memory requirements...")
                memory_sufficient, memory_message = self.check_memory_requirements("gguf", model_size_gb)
                self.memory_check_signal.emit(memory_sufficient, memory_message)
                
                if self.should_abort:
                    return
                
                # Load GGUF model using llama-cpp-python
                try:
                    self.loading_progress_signal.emit("Loading GGUF model with llama-cpp-python...")
                    self.progress_bar_signal.emit(50)
                    
                    # Import and initialize llama-cpp-python
                    from llama_cpp import Llama
                    
                    # Create a wrapper class that mimics the transformers pipeline interface
                    class GGUFPipeline:
                        def __init__(self, model_path, use_gpu=False, progress_callback=None):
                            if progress_callback:
                                progress_callback(f"Loading GGUF model from: {model_path}")
                            
                            # Verify source file exists and is readable
                            if not model_path.exists():
                                raise FileNotFoundError(f"Model file not found: {model_path}")
                            if not os.access(model_path, os.R_OK):
                                raise PermissionError(f"Cannot read model file: {model_path}")
                            if not model_path.is_file():
                                raise ValueError(f"Path exists but is not a file: {model_path}")
                            
                            # Validate the GGUF file before loading
                            if progress_callback:
                                progress_callback("Validating GGUF file...")
                            
                            # Check if the file starts with the GGUF magic number
                            try:
                                with open(model_path, 'rb') as f:
                                    magic = f.read(4)
                                    if magic != b'GGUF':
                                        raise ValueError(f"Invalid GGUF file: expected magic number 'GGUF', got {magic}")
                                if progress_callback:
                                    progress_callback("GGUF file validation passed")
                            except Exception as validation_error:
                                raise ValueError(f"GGUF file validation failed: {validation_error}")
                            
                            # Get CPU count safely
                            cpu_count = os.cpu_count()
                            if cpu_count is None:
                                cpu_count = 4  # Default to 4 threads if can't detect
                            n_threads = max(1, cpu_count - 1)  # Use all but one CPU core
                            
                            try:
                                # Initialize model with detailed error handling
                                if progress_callback:
                                    progress_callback(f"Loading model from {model_path} with {n_threads} threads...")
                                
                                # Add more detailed error handling
                                try:
                                    self.llm = Llama(
                                        model_path=str(model_path),
                                        n_ctx=2048,  # Context window
                                        n_threads=n_threads,
                                        n_gpu_layers=64 if use_gpu else 0  # Use GPU if available
                                    )
                                except Exception as init_error:
                                    # Try with different parameters if the first attempt fails
                                    if progress_callback:
                                        progress_callback(f"First initialization failed: {init_error}")
                                        progress_callback("Trying with reduced parameters...")
                                    
                                    # Try with fewer GPU layers and smaller context
                                    self.llm = Llama(
                                        model_path=str(model_path),
                                        n_ctx=1024,  # Smaller context window
                                        n_threads=max(1, n_threads // 2),  # Fewer threads
                                        n_gpu_layers=0  # CPU only
                                    )
                                
                                if progress_callback:
                                    progress_callback("Model loaded successfully")
                            except Exception as e:
                                error_msg = f"Failed to initialize model: {str(e)}"
                                if progress_callback:
                                    progress_callback(f"Error details: {error_msg}")
                                raise Exception(error_msg)
                        
                        def cleanup(self):
                            """Clean up model"""
                            try:
                                # Clean up the model
                                if hasattr(self, 'llm'):
                                    # Delete the model reference
                                    del self.llm
                                    
                                # Force garbage collection
                                import gc
                                gc.collect()
                            except Exception as e:
                                print(f"Error during cleanup: {e}")
                        
                        def __del__(self):
                            """Ensure cleanup on deletion"""
                            self.cleanup()
                        
                        def __call__(self, prompt, **kwargs):
                            if not hasattr(self, 'llm'):
                                raise RuntimeError("Model not initialized or already cleaned up")
                            
                            # Extract relevant kwargs
                            max_tokens = int(kwargs.get('max_new_tokens', 128))
                            temperature = float(kwargs.get('temperature', 0.7))
                            top_p = float(kwargs.get('top_p', 0.9))
                            
                            # Generate response
                            response = self.llm(
                                prompt,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                echo=False
                            )
                            
                            # Format response to match transformers pipeline output
                            if isinstance(response, dict) and 'choices' in response:
                                text = response['choices'][0]['text']
                            else:
                                # Handle streaming response
                                text = ''.join(chunk['choices'][0]['text'] for chunk in response)
                            
                            return [{
                                'generated_text': text
                            }]
                        
                        def create_chat_completion(self, messages, **kwargs):
                            """Create chat completion for GGUF models"""
                            if not hasattr(self, 'llm'):
                                raise RuntimeError("Model not initialized or already cleaned up")
                            
                            # Extract parameters
                            max_tokens = int(kwargs.get('max_tokens', 150))
                            temperature = float(kwargs.get('temperature', 0.7))
                            top_p = float(kwargs.get('top_p', 0.9))
                            stop = kwargs.get('stop', [])
                            
                            # Use the llama-cpp-python chat completion
                            response = self.llm.create_chat_completion(
                                messages=messages,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                stop=stop
                            )
                            
                            return response
                    
                    # Create pipeline with GPU support if available
                    use_gpu = torch.cuda.is_available()
                    if use_gpu:
                        gpu_name = torch.cuda.get_device_name(0)
                        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                        self.loading_progress_signal.emit(f"Found GPU: {gpu_name} with {vram_gb:.1f}GB VRAM")
                    else:
                        self.loading_progress_signal.emit("No GPU found, using CPU only")
                    
                    self.loading_progress_signal.emit(f"Initializing model {'with GPU acceleration' if use_gpu else 'on CPU'}...")
                    pipeline_obj = GGUFPipeline(gguf_path, use_gpu=use_gpu, progress_callback=self.loading_progress_signal.emit)
                    self.progress_bar_signal.emit(90)
                    self.model_loaded_signal.emit(pipeline_obj, "gguf")
                    return  # Exit successfully after loading GGUF model
                    
                except ImportError:
                    # Try to install llama-cpp-python with CUDA support if GPU is available
                    self.loading_progress_signal.emit("Installing llama-cpp-python...")
                    import subprocess
                    try:
                        if use_gpu:
                            # Install with CUDA support
                            subprocess.check_call([
                                "pip", "install", "llama-cpp-python[cuda]", "--upgrade",
                                "--extra-index-url", "https://download.pytorch.org/whl/cu118"  # Use CUDA 11.8
                            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        else:
                            # Install CPU-only version
                            subprocess.check_call([
                                "pip", "install", "llama-cpp-python", "--upgrade"
                            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        
                        # Retry loading after installation
                        from llama_cpp import Llama
                        pipeline_obj = GGUFPipeline(gguf_path, use_gpu=use_gpu, progress_callback=self.loading_progress_signal.emit)
                        self.progress_bar_signal.emit(90)
                        self.model_loaded_signal.emit(pipeline_obj, "gguf")
                        return  # Exit successfully after installation and loading
                        
                    except Exception as e:
                        raise Exception(f"Failed to install llama-cpp-python: {e}")
                    
                    return
            
            # Not a GGUF file or directory with GGUF, check for Hugging Face model format
            if (model_path / "config.json").exists():
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
                model_size_gb = self.estimate_model_size(str(model_path))
                if model_size_gb > 0:
                    self.loading_progress_signal.emit(f"Model size: {model_size_gb:.2f}GB")
                
                # Check memory requirements
                self.progress_bar_signal.emit(25)
                self.loading_progress_signal.emit("Checking memory requirements...")
                memory_sufficient, memory_message = self.check_memory_requirements(model_type, model_size_gb)
                self.memory_check_signal.emit(memory_sufficient, memory_message)
                
                if self.should_abort:
                    return
                
                # Create temporary directory in system memory for faster access
                import tempfile
                temp_dir = tempfile.mkdtemp(prefix='huggingdrive_')
                
                try:
                    # Copy model files to temporary directory
                    self.loading_progress_signal.emit("Copying model to system memory for faster access...")
                    import shutil
                    temp_model_path = Path(temp_dir) / model_path.name
                    shutil.copytree(model_path, temp_model_path)
                    
                    # Register cleanup
                    import atexit
                    atexit.register(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
                    
                    # Load model based on type
                    if model_type in ["gpt2", "gpt", "causal_lm", "llama", "mistral", "falcon", "bloom", "opt"]:
                        self.loading_progress_signal.emit("Loading text generation pipeline...")
                        self.progress_bar_signal.emit(50)
                        
                        if self.should_abort:
                            return
                        
                        # Check timeout
                        if time.time() - start_time > self.timeout_seconds:
                            self.error_signal.emit(f"Model loading timed out after {self.timeout_seconds} seconds")
                            return
                        
                        from transformers.pipelines import pipeline
                        pipeline_obj = pipeline("text-generation", model=str(temp_model_path), device=0 if torch.cuda.is_available() else -1)
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
                        
                        from transformers.pipelines import pipeline
                        pipeline_obj = pipeline("text-classification", model=str(temp_model_path), device=0 if torch.cuda.is_available() else -1)
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
                        
                        from transformers.pipelines import pipeline
                        pipeline_obj = pipeline("translation", model=str(temp_model_path), device=0 if torch.cuda.is_available() else -1)
                        self.progress_bar_signal.emit(90)
                        self.model_loaded_signal.emit(pipeline_obj, model_type)
                        
                    elif model_type in ["t5", "bart"]:
                        self.loading_progress_signal.emit("Loading text-to-text pipeline...")
                        self.progress_bar_signal.emit(50)
                        
                        if self.should_abort:
                            return
                        
                        # Check timeout
                        if time.time() - start_time > self.timeout_seconds:
                            self.error_signal.emit(f"Model loading timed out after {self.timeout_seconds} seconds")
                            return
                        
                        from transformers.pipelines import pipeline
                        pipeline_obj = pipeline("text2text-generation", model=str(temp_model_path), device=0 if torch.cuda.is_available() else -1)
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
                        
                        from transformers.pipelines import pipeline
                        pipeline_obj = pipeline("image-classification", model=str(temp_model_path), device=0 if torch.cuda.is_available() else -1)
                        self.progress_bar_signal.emit(90)
                        self.model_loaded_signal.emit(pipeline_obj, model_type)
                        
                    else:
                        self.error_signal.emit(f"Model type '{model_type}' not supported for testing")
                        
                except Exception as e:
                    # Clean up temporary directory on error
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    raise e
                    
            else:
                self.error_signal.emit("Model path must be either a .gguf file or a directory containing a GGUF file or config.json")
                
        except Exception as e:
            self.error_signal.emit(f"Error loading model: {str(e)}") 