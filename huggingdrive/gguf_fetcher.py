"""
GGUF File Fetcher - Background thread for fetching GGUF files from repositories
"""

from PyQt6.QtCore import QThread, pyqtSignal
from typing import List, Dict
import requests


class GGUFFileFetcherThread(QThread):
    """Thread for fetching GGUF files without blocking the GUI"""

    files_found_signal = pyqtSignal(list)  # List of file info dicts
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(str)

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

    def run(self):
        try:
            self.progress_signal.emit(f"Fetching GGUF files for {self.model_name}...")

            # Use huggingface_hub to get repository files
            from huggingface_hub import list_repo_files

            # Get all files in the repository
            files = list_repo_files(self.model_name)

            # Filter for GGUF files
            gguf_files = [f for f in files if f.endswith(".gguf")]

            if not gguf_files:
                self.error_signal.emit("No GGUF files found in this repository.")
                return

            self.progress_signal.emit(
                f"Found {len(gguf_files)} GGUF files. Getting file sizes..."
            )

            # Get file sizes for each GGUF file
            file_info_list = []
            for i, file in enumerate(gguf_files):
                self.progress_signal.emit(
                    f"Checking file {i+1}/{len(gguf_files)}: {file}"
                )

                try:
                    from huggingface_hub import hf_hub_url

                    url = hf_hub_url(self.model_name, file)
                    response = requests.head(url, timeout=10)

                    if response.status_code == 200:
                        size = int(response.headers.get("content-length", 0))
                        size_gb = size / (1024**3)
                        display_text = f"{file} ({size_gb:.2f}GB)"
                    else:
                        display_text = file
                        size_gb = 0
                except Exception as e:
                    display_text = file
                    size_gb = 0

                file_info_list.append(
                    {
                        "filename": file,
                        "display_text": display_text,
                        "size_gb": size_gb,
                        "url": f"https://huggingface.co/{self.model_name}/resolve/main/{file}",
                    }
                )

            self.progress_signal.emit(
                f"Successfully fetched {len(file_info_list)} GGUF files"
            )
            self.files_found_signal.emit(file_info_list)

        except Exception as e:
            self.error_signal.emit(f"Error fetching GGUF files: {str(e)}")
