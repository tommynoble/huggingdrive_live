"""
GGUF Converter - Proper implementation using llama.cpp conversion tools
"""

import os
import sys
import time
import json
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict
from PyQt6.QtCore import QThread, pyqtSignal


class GGUFConverter(QThread):
    """Thread for converting models to GGUF format using llama.cpp tools"""

    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    progress_bar_signal = pyqtSignal(int)
    conversion_stats_signal = pyqtSignal(str)

    def __init__(
        self,
        model_path: str,
        output_path: Optional[str] = None,
        quantization: str = "q4_K_M",
        context_length: int = 4096,
    ):
        super().__init__()
        self.model_path = model_path
        self.output_path = output_path or self._get_default_output_path(model_path)
        self.quantization = quantization
        self.context_length = context_length

        # Set up cache directories
        home_dir = Path.home()
        self.cache_dir = home_dir / ".huggingdrive_cache"
        self.conversion_cache_dir = self.cache_dir / "conversion_cache"

        try:
            self.conversion_cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create conversion cache directory: {e}")

    def _get_default_output_path(self, model_path: str) -> str:
        """Generate default output path for converted model"""
        model_dir = Path(model_path)
        model_name = model_dir.name
        parent_dir = model_dir.parent
        return str(parent_dir / f"{model_name}_gguf")

    def _find_llama_cpp_converter(self) -> Optional[str]:
        """Find the llama.cpp conversion script"""
        possible_paths = [
            # Check if llama.cpp is in the project directory
            "llama.cpp/convert.py",
            "llama.cpp/convert-llama-gguf.py",
            "llama.cpp/convert-hf-to-gguf.py",
            # Check if installed via pip
            "convert.py",
            "convert-llama-gguf.py",
            "convert-hf-to-gguf.py",
            # Check in Python path
            shutil.which("convert.py"),
            shutil.which("convert-llama-gguf.py"),
            shutil.which("convert-hf-to-gguf.py"),
        ]

        for path in possible_paths:
            if path and Path(path).exists():
                return str(Path(path).resolve())

        return None

    def _install_llama_cpp_tools(self) -> bool:
        """Install llama.cpp conversion tools"""
        try:
            self.progress_signal.emit("Installing llama.cpp conversion tools...")

            # Try to install llama-cpp-python with conversion tools
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "llama-cpp-python[conversion]",
                    "--upgrade",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Also try to install the conversion script directly
            try:
                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "llama-cpp-python",
                        "--upgrade",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            except:
                pass

            return True

        except subprocess.CalledProcessError as e:
            self.progress_signal.emit(f"Failed to install llama.cpp tools: {e}")
            return False

    def _get_model_info(self, model_path: str) -> dict:
        """Extract model information from config.json"""
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                return {
                    "model_type": config.get("model_type", "unknown"),
                    "architectures": config.get("architectures", []),
                    "vocab_size": config.get("vocab_size", 0),
                    "hidden_size": config.get("hidden_size", 0),
                    "num_attention_heads": config.get("num_attention_heads", 0),
                    "num_hidden_layers": config.get("num_hidden_layers", 0),
                }
            except Exception as e:
                self.progress_signal.emit(f"Warning: Could not read model config: {e}")
        return {}

    def _validate_model(self, model_path: str) -> bool:
        """Validate that the model can be converted"""
        required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
        model_files = ["pytorch_model.bin", "model.safetensors"]

        path = Path(model_path)

        # Check for required files
        for file in required_files:
            if not (path / file).exists():
                self.progress_signal.emit(f"❌ Missing required file: {file}")
                return False

        # Check for model weights
        has_weights = any((path / file).exists() for file in model_files)
        if not has_weights:
            self.progress_signal.emit(
                "❌ No model weights found (pytorch_model.bin or model.safetensors)"
            )
            return False

        return True

    def _determine_conversion_script(self, model_info: dict) -> str:
        """Determine which conversion script to use based on model type"""
        model_type = model_info.get("model_type", "").lower()
        architectures = [arch.lower() for arch in model_info.get("architectures", [])]

        # Check for specific model types
        if any("llama" in arch for arch in architectures) or "llama" in model_type:
            return "convert-llama-gguf.py"
        elif any("qwen" in arch for arch in architectures) or "qwen" in model_type:
            return "convert-qwen-gguf.py"
        elif (
            any("mistral" in arch for arch in architectures) or "mistral" in model_type
        ):
            return "convert-mistral-gguf.py"
        elif any("gpt2" in arch for arch in architectures) or "gpt2" in model_type:
            return "convert-gpt2-gguf.py"
        elif any("gpt" in arch for arch in architectures) or "gpt" in model_type:
            return "convert-gpt-gguf.py"
        else:
            return "convert.py"  # Generic converter

    def run(self):
        try:
            self.progress_signal.emit(
                f"Starting GGUF conversion of {self.model_path}..."
            )
            self.progress_signal.emit(f"Output directory: {self.output_path}")
            self.progress_bar_signal.emit(5)

            # Validate model path
            if not os.path.exists(self.model_path):
                raise ValueError(f"Model path does not exist: {self.model_path}")

            # Validate the model
            if not self._validate_model(self.model_path):
                raise ValueError("Model validation failed")

            # Get model info
            model_info = self._get_model_info(self.model_path)
            if model_info:
                self.progress_signal.emit(
                    f"Model type: {model_info.get('model_type', 'unknown')}"
                )
                self.progress_signal.emit(
                    f"Architectures: {', '.join(model_info.get('architectures', []))}"
                )

            self.progress_bar_signal.emit(10)

            # Find conversion script
            conversion_script = self._find_llama_cpp_converter()
            if not conversion_script:
                self.progress_signal.emit(
                    "llama.cpp conversion tools not found, attempting to install..."
                )
                if not self._install_llama_cpp_tools():
                    raise RuntimeError("Failed to install llama.cpp conversion tools")

                # Try to find the script again
                conversion_script = self._find_llama_cpp_converter()
                if not conversion_script:
                    raise RuntimeError(
                        "Could not find conversion script after installation"
                    )

            self.progress_signal.emit(f"Using conversion script: {conversion_script}")
            self.progress_bar_signal.emit(20)

            # Create output directory
            try:
                os.makedirs(self.output_path, exist_ok=True)
                self.progress_signal.emit("Created output directory")
            except Exception as e:
                raise Exception(f"Failed to create output directory: {str(e)}")

            self.progress_bar_signal.emit(30)

            # Determine output filename
            model_name = Path(self.model_path).name
            output_filename = f"{model_name}.{self.quantization}.gguf"
            output_file = Path(self.output_path) / output_filename

            # Start conversion
            start_time = time.time()

            self.progress_signal.emit(
                f"Converting to GGUF format with {self.quantization} quantization..."
            )
            self.progress_signal.emit(f"Context length: {self.context_length}")

            # Build conversion command
            cmd = [
                sys.executable,
                conversion_script,
                str(self.model_path),
                "--outfile",
                str(output_file),
                "--outtype",
                self.quantization,
                "--context-length",
                str(self.context_length),
            ]

            self.progress_signal.emit(f"Running conversion command: {' '.join(cmd)}")
            self.progress_bar_signal.emit(40)

            # Run conversion
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            # Monitor conversion progress
            if process.stdout:
                while True:
                    output = process.stdout.readline()
                    if output == "" and process.poll() is not None:
                        break
                    if output:
                        self.progress_signal.emit(output.strip())
                        # Update progress based on output
                        if "Converting" in output:
                            self.progress_bar_signal.emit(60)
                        elif "Saving" in output:
                            self.progress_bar_signal.emit(80)

            # Check if conversion was successful
            if process.returncode != 0:
                raise RuntimeError(
                    f"Conversion failed with return code: {process.returncode}"
                )

            conversion_time = time.time() - start_time

            # Calculate conversion statistics
            try:
                input_size = sum(
                    f.stat().st_size
                    for f in Path(self.model_path).rglob("*")
                    if f.is_file()
                )
                output_size = output_file.stat().st_size if output_file.exists() else 0

                input_mb = input_size / (1024 * 1024)
                output_mb = output_size / (1024 * 1024)
                compression_ratio = input_mb / output_mb if output_mb > 0 else 0

                stats_msg = f"Conversion completed in {conversion_time:.1f}s"
                stats_msg += f"\nInput size: {input_mb:.1f}MB"
                stats_msg += f"\nOutput size: {output_mb:.1f}MB"
                stats_msg += f"\nCompression ratio: {compression_ratio:.2f}x"

                self.conversion_stats_signal.emit(stats_msg)

            except Exception as e:
                self.progress_signal.emit(
                    f"Could not calculate conversion statistics: {e}"
                )

            self.progress_bar_signal.emit(100)
            self.progress_signal.emit("✅ GGUF conversion completed successfully!")

            success_msg = f"Model converted to GGUF format successfully: {output_file}"
            self.finished_signal.emit(True, success_msg)

        except Exception as e:
            error_msg = f"❌ GGUF conversion failed: {str(e)}"
            self.progress_signal.emit(error_msg)
            self.finished_signal.emit(False, error_msg)


class GGUFConverterManager:
    """Manager class for GGUF conversion operations"""

    def __init__(self):
        self.supported_quantizations = {
            "q4_0": "4-bit quantization, fast inference, moderate quality",
            "q4_1": "4-bit quantization, balanced speed and quality",
            "q4_K_M": "4-bit quantization, good quality, recommended",
            "q4_K_S": "4-bit quantization, smaller size, lower quality",
            "q5_0": "5-bit quantization, better quality than q4",
            "q5_1": "5-bit quantization, balanced 5-bit option",
            "q5_K_M": "5-bit quantization, good quality",
            "q5_K_S": "5-bit quantization, smaller size",
            "q8_0": "8-bit quantization, high quality, larger size",
        }

    def get_quantization_options(self) -> dict:
        """Get available quantization options"""
        return self.supported_quantizations.copy()

    def validate_model_for_conversion(self, model_path: str) -> Tuple[bool, str]:
        """Validate if a model can be converted to GGUF"""
        try:
            path = Path(model_path)
            if not path.exists():
                return False, "Model path does not exist"

            # Check for required files
            required_files = ["config.json", "tokenizer.json"]
            for file in required_files:
                if not (path / file).exists():
                    return False, f"Missing required file: {file}"

            # Check for model weights
            model_files = ["pytorch_model.bin", "model.safetensors"]
            has_weights = any((path / file).exists() for file in model_files)
            if not has_weights:
                return False, "No model weights found"

            return True, "Model is valid for conversion"

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def estimate_conversion_size(
        self, model_path: str, quantization: str
    ) -> Tuple[float, float]:
        """Estimate input and output sizes for conversion"""
        try:
            input_size = sum(
                f.stat().st_size for f in Path(model_path).rglob("*") if f.is_file()
            )
            input_mb = input_size / (1024 * 1024)

            # Rough estimation based on quantization
            compression_ratios = {
                "q4_0": 4.0,
                "q4_1": 4.0,
                "q4_K_M": 4.0,
                "q4_K_S": 4.0,
                "q5_0": 3.2,
                "q5_1": 3.2,
                "q5_K_M": 3.2,
                "q5_K_S": 3.2,
                "q8_0": 2.0,
            }

            compression_ratio = compression_ratios.get(quantization, 4.0)
            output_mb = input_mb / compression_ratio

            return input_mb, output_mb

        except Exception:
            return 0.0, 0.0
