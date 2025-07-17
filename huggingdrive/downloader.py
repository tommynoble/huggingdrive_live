"""
Model Downloader - Handles downloading models from Hugging Face Hub
"""

import os
import time
import requests
import threading
import queue
import concurrent.futures
import atexit
import signal
from pathlib import Path
from typing import Optional
from PyQt6.QtCore import QThread, pyqtSignal


class ModelDownloader(QThread):
    """Thread for downloading models without blocking the GUI"""

    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    progress_bar_signal = pyqtSignal(int)  # NEW: signal for progress bar
    download_stats_signal = pyqtSignal(str)  # NEW: signal for download statistics

    # Class-level tracking of active resources
    _active_downloaders = set()
    _cleanup_lock = threading.Lock()

    @classmethod
    def cleanup_resources(cls):
        """Clean up any active downloaders"""
        with cls._cleanup_lock:
            for downloader in list(cls._active_downloaders):
                try:
                    downloader.stop()
                except:
                    pass
            cls._active_downloaders.clear()

    def __init__(
        self, model_name: str, target_path: str, model_type: str = "transformers"
    ):
        super().__init__()
        self.model_name = model_name
        self.target_path = target_path
        self.model_type = model_type
        self._should_stop = False
        self._is_stopping = False
        self._executor = None
        self._writer_thread = None
        self._cleanup_required = False
        self._partial_file = None

        # Track this instance
        with self._cleanup_lock:
            self._active_downloaders.add(self)

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
        os.environ["TRANSFORMERS_CACHE"] = str(self.model_cache_dir)
        os.environ["HF_HOME"] = str(self.cache_dir)
        os.environ["HF_DATASETS_CACHE"] = str(self.datasets_cache_dir)

        # Download settings
        self.chunk_size = 8 * 1024 * 1024  # 8MB chunks for better throughput
        self.max_workers = 4  # Number of parallel download threads
        self.write_queue = queue.Queue(maxsize=10)  # Queue for coordinating writes
        self.downloaded_chunks = {}  # Track downloaded chunks
        self.total_downloaded = 0
        self.total_size = 0
        self.start_time = 0

    def __del__(self):
        """Clean up resources when the object is deleted"""
        try:
            # Don't emit signals if the object is being deleted
            self._is_stopping = True
            self._should_stop = True

            # Clear the write queue to prevent blocking
            while not self.write_queue.empty():
                try:
                    self.write_queue.get_nowait()
                    self.write_queue.task_done()
                except queue.Empty:
                    break

            # Cancel any running executor
            if hasattr(self, "_executor") and self._executor:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None

            # Stop writer thread
            if (
                hasattr(self, "_writer_thread")
                and self._writer_thread
                and self._writer_thread.is_alive()
            ):
                self._writer_thread.join(timeout=0.5)  # Shorter timeout
                self._writer_thread = None

            # Clean up partial downloads
            if (
                hasattr(self, "_cleanup_required")
                and self._cleanup_required
                and hasattr(self, "_partial_file")
                and self._partial_file
            ):
                try:
                    if os.path.exists(self._partial_file):
                        os.remove(self._partial_file)
                except:
                    pass  # Ignore cleanup errors

        except Exception as e:
            # Don't print during deletion to avoid noise
            pass
        finally:
            with self._cleanup_lock:
                self._active_downloaders.discard(self)

    def stop(self):
        """Stop the download gracefully"""
        if self._is_stopping:
            return

        self._is_stopping = True
        self._should_stop = True
        self.progress_signal.emit("Stopping download gracefully...")

        try:
            # Clear the write queue to prevent blocking
            while not self.write_queue.empty():
                try:
                    self.write_queue.get_nowait()
                    self.write_queue.task_done()
                except queue.Empty:
                    break

            # Cancel any running executor
            if self._executor:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None

            # Stop writer thread
            if self._writer_thread and self._writer_thread.is_alive():
                self._writer_thread.join(timeout=1)
                self._writer_thread = None

            # Clean up partial downloads
            if self._cleanup_required and self._partial_file:
                try:
                    if os.path.exists(self._partial_file):
                        os.remove(self._partial_file)
                except:
                    pass  # Ignore cleanup errors

            self.progress_signal.emit("Download stopped.")

        except Exception as e:
            self.progress_signal.emit(f"Error during stop: {str(e)}")
        finally:
            self._is_stopping = False
            with self._cleanup_lock:
                self._active_downloaders.discard(self)

    def _download_chunk(self, url: str, start: int, end: int, chunk_id: int):
        """Download a specific chunk of the file"""
        if self._should_stop:
            return 0

        headers = {"Range": f"bytes={start}-{end}"}
        try:
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()

            if self._should_stop:
                return 0

            chunk_data = response.content
            self.write_queue.put((chunk_id, chunk_data))
            return len(chunk_data)
        except Exception as e:
            if not self._should_stop:  # Only log error if not stopping
                self.progress_signal.emit(f"Chunk {chunk_id} download error: {str(e)}")
            return 0

    def _write_chunks_to_file(self, target_file: str, total_chunks: int):
        """Write downloaded chunks to file in order"""
        self._partial_file = target_file
        self._cleanup_required = True

        try:
            with open(target_file, "wb") as f:
                for chunk_id in range(total_chunks):
                    if self._should_stop:
                        return

                    try:
                        chunk_id, chunk_data = self.write_queue.get(
                            timeout=5
                        )  # 5 second timeout
                        f.write(chunk_data)
                        self.total_downloaded += len(chunk_data)

                        # Calculate and emit progress
                        if self.total_size > 0:
                            progress = (self.total_downloaded / self.total_size) * 100
                            self.progress_bar_signal.emit(int(progress))

                            # Calculate speed
                            elapsed = time.time() - self.start_time
                            if elapsed > 0:
                                speed = self.total_downloaded / (
                                    1024 * 1024 * elapsed
                                )  # MB/s
                                self.download_stats_signal.emit(
                                    f"Downloaded: {self.total_downloaded/(1024*1024):.1f}MB / "
                                    f"{self.total_size/(1024*1024):.1f}MB ({speed:.1f}MB/s)"
                                )

                        self.write_queue.task_done()
                    except queue.Empty:
                        if self._should_stop:
                            return
                        continue  # Try next chunk if timeout
                    except Exception as e:
                        if not self._should_stop:
                            self.progress_signal.emit(
                                f"Error writing chunk {chunk_id}: {str(e)}"
                            )
                        return

            # If we completed successfully, don't delete the file
            self._cleanup_required = False

        except Exception as e:
            if not self._should_stop:
                self.progress_signal.emit(f"Error in writer thread: {str(e)}")

    def _download_file(self, url: str, target_path: str):
        """Download a single file with parallel chunks"""
        try:
            # Set up headers for Hugging Face
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)",
                "Accept": "*/*",
            }

            # First try to get the direct download URL with proper redirect handling
            self.progress_signal.emit("Resolving download URL...")
            response = requests.head(
                url, headers=headers, timeout=30, allow_redirects=True
            )

            # Check if we're being redirected to login or error pages
            if "huggingface.co/login" in response.url:
                raise ValueError(
                    "This model requires Hugging Face authentication. Please:\n"
                    "1. Download the model directly from Hugging Face website\n"
                    "2. Place the .gguf file in your models directory\n"
                    "3. Use the local file path instead of the URL"
                )

            # Check for other error redirects
            if any(
                error_page in response.url.lower()
                for error_page in ["error", "404", "403", "unauthorized"]
            ):
                raise ValueError(f"URL redirects to error page: {response.url}")

            # Get the final URL after redirects
            final_url = response.url
            self.progress_signal.emit(f"Resolved URL: {final_url}")

            # Verify we got a valid response
            if response.status_code != 200:
                raise ValueError(f"Failed to access file: HTTP {response.status_code}")

            # Check content type to ensure we're getting a file, not HTML
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" in content_type or "text/plain" in content_type:
                raise ValueError(
                    f"URL returns HTML/text instead of a file (content-type: {content_type})"
                )

            # Get file size
            self.total_size = int(response.headers.get("content-length", 0))

            # Get filename from URL
            filename = os.path.basename(final_url.split("?")[0])
            if not filename:
                filename = "model.gguf"

            # Ensure filename has .gguf extension
            if not filename.endswith(".gguf"):
                filename += ".gguf"

            # Construct full target file path
            target_file = os.path.join(target_path, filename)

            # Create parent directory if it doesn't exist
            parent_dir = os.path.dirname(target_file)
            if parent_dir:  # Only create if there's a parent directory
                os.makedirs(parent_dir, exist_ok=True)

            if self.total_size == 0:
                # If server doesn't support HEAD or content-length, try a direct download
                self.progress_signal.emit(
                    "Server doesn't support partial downloads, using direct download..."
                )
                response = requests.get(
                    final_url,
                    headers=headers,
                    stream=True,
                    timeout=60,
                    allow_redirects=True,
                )
                response.raise_for_status()

                # Download with progress tracking
                self.start_time = time.time()
                self._partial_file = target_file
                self._cleanup_required = True

                with open(target_file, "wb") as f:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if self._should_stop:
                            return False
                        f.write(chunk)
                        self.total_downloaded += len(chunk)

                        # Calculate progress and speed
                        if self.total_size > 0:
                            progress = (self.total_downloaded / self.total_size) * 100
                            self.progress_bar_signal.emit(int(progress))

                            # Calculate speed
                            elapsed = time.time() - self.start_time
                            if elapsed > 0:
                                speed = self.total_downloaded / (
                                    1024 * 1024 * elapsed
                                )  # MB/s
                                self.download_stats_signal.emit(
                                    f"Downloaded: {self.total_downloaded/(1024*1024):.1f}MB / "
                                    f"{self.total_size/(1024*1024):.1f}MB ({speed:.1f}MB/s)"
                                )

                self._cleanup_required = False
                return True

            # Calculate chunk ranges
            chunk_size = min(self.chunk_size, self.total_size // self.max_workers)
            chunks = []
            for i in range(0, self.total_size, chunk_size):
                end = min(i + chunk_size - 1, self.total_size - 1)
                chunks.append((i, end, len(chunks)))

            # Create parent directory
            os.makedirs(os.path.dirname(target_file), exist_ok=True)

            # Start writer thread
            self._writer_thread = threading.Thread(
                target=self._write_chunks_to_file, args=(target_file, len(chunks))
            )
            self._writer_thread.start()

            # Download chunks in parallel
            self.start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                self._executor = executor
                futures = []
                for start, end, chunk_id in chunks:
                    if self._should_stop:
                        return False
                    futures.append(
                        executor.submit(
                            self._download_chunk, final_url, start, end, chunk_id
                        )
                    )

                # Wait for all chunks to download
                for future in concurrent.futures.as_completed(futures):
                    if self._should_stop:
                        return False
                    try:
                        chunk_size = future.result()
                        if chunk_size == 0 and not self._should_stop:
                            raise Exception("Chunk download failed")
                    except Exception as e:
                        if not self._should_stop:
                            raise Exception(f"Download failed: {str(e)}")

            # Wait for writer to finish
            if self._writer_thread:
                self._writer_thread.join(timeout=5)  # 5 second timeout

            # Verify the downloaded file is a valid GGUF file
            self.progress_signal.emit("Verifying downloaded file...")
            try:
                with open(target_file, "rb") as f:
                    magic = f.read(4)
                    if magic != b"GGUF":
                        raise ValueError(
                            f"Downloaded file is not a valid GGUF model file (detected type: {magic})"
                        )
                self.progress_signal.emit(
                    "✅ File verification passed - valid GGUF file"
                )
            except Exception as e:
                # Clean up the invalid file
                if os.path.exists(target_file):
                    os.remove(target_file)
                raise ValueError(
                    f"Downloaded file is not a valid GGUF model file: {str(e)}"
                )

            return True

        except Exception as e:
            # Clean up any partial file
            if os.path.exists(target_file):
                os.remove(target_file)
            raise e

    def check_auth_requirements(self, url: str) -> tuple[bool, str]:
        """
        Check if a model requires authentication before downloading
        Returns: (requires_auth, message)
        """
        try:
            # Set up headers for Hugging Face
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)",
                "Accept": "*/*",
            }

            # Make a HEAD request to check access
            response = requests.head(
                url, headers=headers, timeout=10, allow_redirects=True
            )

            # Check for authentication redirects
            if "huggingface.co/login" in response.url:
                return True, "🔒 This model requires Hugging Face authentication"

            # Check for other authentication issues
            if response.status_code == 401:
                return (
                    True,
                    "🔒 Model is private or requires authentication (401 Unauthorized)",
                )
            elif response.status_code == 403:
                return (
                    True,
                    "🔒 Access forbidden - model may be private (403 Forbidden)",
                )
            elif response.status_code == 404:
                return True, "❌ Model not found (404) - check the URL"

            # Check if we got HTML instead of a file (indicates login page or error)
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" in content_type:
                return True, "🔒 Redirected to login page - authentication required"

            # Success - no authentication required
            return False, "✅ Model is publicly accessible"

        except requests.exceptions.RequestException as e:
            return True, f"❌ Network error checking access: {str(e)}"
        except Exception as e:
            return True, f"❌ Error checking access: {str(e)}"

    def check_model_access(self, model_name: str) -> tuple[bool, str]:
        """
        Check if user has access to a specific model
        Returns: (has_access, message)
        """
        try:
            token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if not token:
                return False, "🔒 No authentication token found. Please log in first."

            from huggingface_hub import HfApi

            api = HfApi(token=token)

            # Try to get model info
            try:
                info = api.model_info(model_name)
                return True, f"✅ Access granted to {model_name}"
            except Exception as e:
                error_msg = str(e)
                if "403" in error_msg or "forbidden" in error_msg.lower():
                    return (
                        False,
                        f"🔒 Access denied to {model_name}. You need to request access.",
                    )
                elif "404" in error_msg:
                    return False, f"❌ Model {model_name} not found"
                else:
                    return False, f"❌ Error checking access: {error_msg}"

        except Exception as e:
            return False, f"❌ Error checking model access: {str(e)}"

    def run(self):
        try:
            self.progress_signal.emit(f"Starting download of {self.model_name}...")
            self.progress_signal.emit(f"Target directory: {self.target_path}")

            # Reset stop flags
            self._should_stop = False
            self._is_stopping = False

            # Validate target path
            if not self.target_path:
                raise ValueError("Target path is empty")

            # Create target directory
            try:
                # Check if target_path is already a file
                if os.path.isfile(self.target_path):
                    error_msg = f"❌ Target path {self.target_path} is a file, but should be a directory"
                    self.progress_signal.emit(error_msg)
                    self.finished_signal.emit(False, error_msg)
                    return

                os.makedirs(self.target_path, exist_ok=True)
                self.progress_signal.emit("Created target directory")
            except PermissionError as e:
                error_msg = f"❌ Permission denied: Cannot write to drive. The drive may be read-only or you don't have write permissions."
                self.progress_signal.emit(error_msg)
                self.progress_signal.emit(
                    "Please check drive permissions or select a different drive."
                )
                self.finished_signal.emit(False, error_msg)
                return
            except OSError as e:
                if e.errno == 30:  # Read-only file system
                    error_msg = (
                        f"❌ Drive is read-only: Cannot write to {self.target_path}"
                    )
                    self.progress_signal.emit(error_msg)
                    self.progress_signal.emit(
                        "Please select a different drive or check drive permissions."
                    )
                    self.finished_signal.emit(False, error_msg)
                    return
                else:
                    raise Exception(f"Failed to create target directory: {str(e)}")
            except Exception as e:
                raise Exception(f"Failed to create target directory: {str(e)}")

            # Get the URL - either direct or from repository
            download_url = self.model_name
            if not download_url.startswith(("http://", "https://")):
                # Convert repository reference to URL
                if "/" in download_url:
                    parts = download_url.split("/")
                    if len(parts) >= 2:
                        repo_name = f"{parts[0]}/{parts[1]}"

                        # Check if a specific file was provided
                        if len(parts) > 2 and parts[-1].endswith(".gguf"):
                            # Specific GGUF file provided
                            file_name = parts[-1]
                            download_url = f"https://huggingface.co/{repo_name}/resolve/main/{file_name}"
                            self.progress_signal.emit(f"Resolved URL: {download_url}")
                        else:
                            # Regular repository provided - download the entire model
                            self.progress_signal.emit(
                                f"Repository provided: {repo_name}"
                            )
                            self.progress_signal.emit(
                                "Downloading entire model repository..."
                            )

                            # Check authentication requirements for repository
                            self.progress_signal.emit("Checking repository access...")

                            # Check if we have a token
                            token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
                            if token:
                                self.progress_signal.emit(
                                    "✅ Authentication token found"
                                )
                            else:
                                self.progress_signal.emit(
                                    "⚠️ No authentication token found"
                                )

                            try:
                                from huggingface_hub import model_info

                                # Use token explicitly for the check
                                info = model_info(repo_name, token=token)
                                self.progress_signal.emit("✅ Repository is accessible")
                            except Exception as e:
                                error_msg = str(e)
                                if (
                                    "403" in error_msg
                                    or "forbidden" in error_msg.lower()
                                ):
                                    if token:
                                        auth_message = f"🔒 Access forbidden - {repo_name} requires model approval"
                                        self.progress_signal.emit(auth_message)
                                        self.progress_signal.emit(
                                            "💡 You're logged in, but need to request access to this model"
                                        )
                                        self.progress_signal.emit(
                                            "   Visit: https://huggingface.co/"
                                            + repo_name
                                        )
                                        self.progress_signal.emit(
                                            "   Click 'Request access' and wait for approval"
                                        )
                                    else:
                                        auth_message = f"🔒 Access forbidden - {repo_name} requires authentication"
                                        self.progress_signal.emit(auth_message)
                                        self.progress_signal.emit(
                                            "💡 Please log in using the '🔑 Login to HF' button"
                                        )
                                    self.finished_signal.emit(False, auth_message)
                                    return
                                elif (
                                    "404" in error_msg
                                    or "not found" in error_msg.lower()
                                ):
                                    auth_message = (
                                        f"❌ Repository not found: {repo_name}"
                                    )
                                    self.progress_signal.emit(auth_message)
                                    self.finished_signal.emit(False, auth_message)
                                    return
                                else:
                                    raise ValueError(
                                        f"Failed to access repository: {error_msg}"
                                    )

                            # Use huggingface_hub to download the model
                            try:
                                from huggingface_hub import snapshot_download

                                self.progress_signal.emit(
                                    "Using huggingface_hub to download model..."
                                )

                                # Track download progress
                                self.start_time = time.time()
                                self.total_downloaded = 0
                                self.total_size = 0

                                # Create a background thread to simulate progress updates
                                def update_progress():
                                    progress = 10
                                    while progress < 90 and not self._should_stop:
                                        time.sleep(2)  # Update every 2 seconds
                                        if not self._should_stop:
                                            progress += 5  # Increment by 5% each update
                                            if progress < 90:
                                                self.progress_bar_signal.emit(progress)
                                                self.progress_signal.emit(
                                                    f"Downloading... {progress}% complete"
                                                )

                                # Start progress update thread
                                progress_thread = threading.Thread(
                                    target=update_progress, daemon=True
                                )
                                progress_thread.start()

                                # Start progress bar
                                self.progress_bar_signal.emit(10)

                                # Download the model to the target path
                                try:
                                    # Get token from environment variable
                                    token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
                                    if token:
                                        self.progress_signal.emit(
                                            "Using authentication token for download..."
                                        )

                                    snapshot_download(
                                        repo_id=repo_name,
                                        local_dir=self.target_path,
                                        local_dir_use_symlinks=False,
                                        resume_download=True,
                                        tqdm_class=None,  # Disable tqdm to use our own progress reporting
                                        token=token,  # Explicitly pass token
                                    )
                                except Exception as e:
                                    # Stop the progress thread if download fails
                                    self._should_stop = True
                                    error_msg = str(e)
                                    if (
                                        "403" in error_msg
                                        or "forbidden" in error_msg.lower()
                                    ):
                                        auth_message = f"🔒 Access forbidden - {repo_name} requires authentication"
                                        self.progress_signal.emit(auth_message)
                                        self.progress_signal.emit(
                                            "💡 Tip: Try searching for public models instead"
                                        )
                                        self.progress_signal.emit(
                                            "   Popular public models: TinyLlama, Rocket-3B, StableLM"
                                        )
                                        self.finished_signal.emit(False, auth_message)
                                        return
                                    else:
                                        raise e

                                # Since we can't easily hook into snapshot_download progress,
                                # we'll simulate progress updates during the download
                                # The actual progress will be shown in the status messages

                                # Final progress update
                                self.progress_bar_signal.emit(95)
                                self.progress_signal.emit(
                                    "✅ Model downloaded successfully!"
                                )

                                # Calculate final statistics
                                try:
                                    end_time = time.time()
                                    duration = end_time - self.start_time
                                    if duration > 0:
                                        # Calculate total downloaded size
                                        total_size = sum(
                                            f.stat().st_size
                                            for f in Path(self.target_path).rglob("*")
                                            if f.is_file()
                                        )
                                        size_mb = total_size / (1024 * 1024)
                                        speed = (
                                            size_mb / duration if duration > 0 else 0
                                        )

                                        self.download_stats_signal.emit(
                                            f"Download completed in {duration:.1f}s | "
                                            f"Total: {size_mb:.1f}MB | "
                                            f"Average speed: {speed:.1f}MB/s"
                                        )
                                except Exception as e:
                                    self.progress_signal.emit(
                                        f"Could not calculate final statistics: {e}"
                                    )

                                self.progress_bar_signal.emit(100)
                                self.finished_signal.emit(
                                    True, f"Model downloaded to {self.target_path}"
                                )
                                return

                            except ImportError:
                                self.progress_signal.emit(
                                    "huggingface_hub not available, falling back to manual download..."
                                )
                                # Fallback: try to find and download individual files
                                raise ValueError(
                                    f"Please install huggingface_hub: pip install huggingface_hub"
                                )
                            except Exception as e:
                                raise ValueError(f"Failed to download model: {str(e)}")

            # Check authentication requirements before starting download
            self.progress_signal.emit("Checking model access...")
            requires_auth, auth_message = self.check_auth_requirements(download_url)

            if requires_auth:
                self.progress_signal.emit(auth_message)
                self.progress_signal.emit(
                    "💡 Tip: Try searching for public GGUF models instead"
                )
                self.progress_signal.emit(
                    "   Popular public models: TinyLlama, Rocket-3B, StableLM"
                )
                self.finished_signal.emit(False, auth_message)
                return

            self.progress_signal.emit(
                "✅ Model is publicly accessible - starting download..."
            )

            # Extract filename from URL
            filename = os.path.basename(download_url.split("?")[0])
            if not filename:
                filename = "model.gguf"  # Default name if none found

            # Ensure we're downloading to a file, not a directory
            target_file = os.path.join(self.target_path, filename)

            # Make sure target_path is a directory, not a file
            if os.path.isfile(self.target_path):
                raise ValueError(
                    f"Target path {self.target_path} is a file, but should be a directory"
                )

            # Check if file already exists
            if os.path.exists(target_file):
                self.progress_signal.emit(
                    f"File {filename} already exists, skipping download"
                )
                success_msg = f"File already exists at {target_file}"
                self.finished_signal.emit(True, success_msg)
                return

            # Download the file
            start_time = time.time()
            if not self._download_file(download_url, self.target_path):
                if self._should_stop:
                    self.progress_signal.emit("Download stopped by user")
                    self.finished_signal.emit(False, "Download stopped by user")
                else:
                    raise Exception("Failed to download file")
                return

            if self._should_stop:
                self.progress_signal.emit("Download stopped by user")
                self.finished_signal.emit(False, "Download stopped by user")
                return

            download_time = time.time() - start_time

            # Calculate download statistics
            try:
                total_size = os.path.getsize(target_file)
                size_mb = total_size / (1024 * 1024)
                speed_mbps = size_mb / download_time if download_time > 0 else 0

                stats_msg = f"Download completed in {download_time:.1f}s"
                stats_msg += f"\nSize: {size_mb:.1f}MB ({total_size / (1024**3):.2f}GB)"
                stats_msg += f"\nAverage speed: {speed_mbps:.1f}MB/s"

                self.download_stats_signal.emit(stats_msg)

            except Exception as e:
                self.progress_signal.emit(
                    f"Could not calculate download statistics: {e}"
                )

            self.progress_bar_signal.emit(100)
            self.progress_signal.emit("✅ Download completed successfully!")

            success_msg = f"File downloaded successfully to {target_file}"
            self.finished_signal.emit(True, success_msg)

        except InterruptedError:
            error_msg = "Download stopped by user"
            self.progress_signal.emit(error_msg)
            self.finished_signal.emit(False, error_msg)
        except Exception as e:
            if self._should_stop:
                error_msg = "Download stopped by user"
            else:
                error_msg = f"❌ Download failed: {str(e)}"
            self.progress_signal.emit(error_msg)
            self.finished_signal.emit(False, error_msg)
        finally:
            # Clean up resources
            self._cleanup_required = False
            self._partial_file = None
            self._executor = None
            self._writer_thread = None
            self._is_stopping = False

    def _find_best_gguf_file(self, repo_name: str) -> Optional[str]:
        """
        Find the best GGUF file in a repository by checking common naming patterns
        Returns the filename of the best GGUF file, or None if not found
        """
        try:
            # Common GGUF file patterns to try, in order of preference
            common_patterns = [
                # Q4_K_M is usually the best balance of size and quality
                "*.Q4_K_M.gguf",
                "*.q4_k_m.gguf",
                # Q4_0 is also good
                "*.Q4_0.gguf",
                "*.q4_0.gguf",
                # Q5_K_M is higher quality but larger
                "*.Q5_K_M.gguf",
                "*.q5_k_m.gguf",
                # Q8_0 is highest quality but largest
                "*.Q8_0.gguf",
                "*.q8_0.gguf",
                # Fallback to any GGUF file
                "*.gguf",
            ]

            # For specific repositories, we can hardcode the best file
            repo_lower = repo_name.lower()
            if "tinyllama" in repo_lower and "chat" in repo_lower:
                return "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
            elif "tinyllama" in repo_lower:
                return "tinyllama-1.1b.Q4_K_M.gguf"
            elif "llama-2-7b" in repo_lower and "chat" in repo_lower:
                return "llama-2-7b-chat.Q4_K_M.gguf"
            elif "llama-2-13b" in repo_lower and "chat" in repo_lower:
                return "llama-2-13b-chat.Q4_K_M.gguf"
            elif "mistral" in repo_lower and "instruct" in repo_lower:
                return "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
            elif "codellama" in repo_lower and "python" in repo_lower:
                return "codellama-7b-python.Q4_K_M.gguf"

            # For other repositories, try to fetch the file list
            try:
                import requests

                api_url = f"https://huggingface.co/api/repos/{repo_name}/tree/main"
                response = requests.get(api_url, timeout=10)
                if response.status_code == 200:
                    files = response.json()
                    gguf_files = [
                        f["path"] for f in files if f["path"].endswith(".gguf")
                    ]

                    if gguf_files:
                        # Sort by preference (Q4_K_M first, then Q4_0, etc.)
                        def sort_key(filename):
                            filename_lower = filename.lower()
                            if "q4_k_m" in filename_lower:
                                return 0
                            elif "q4_0" in filename_lower:
                                return 1
                            elif "q5_k_m" in filename_lower:
                                return 2
                            elif "q8_0" in filename_lower:
                                return 3
                            else:
                                return 4

                        gguf_files.sort(key=sort_key)
                        return gguf_files[0]  # Return the best one

            except Exception as e:
                self.progress_signal.emit(f"Could not fetch file list: {e}")

            # If all else fails, return None
            return None

        except Exception as e:
            self.progress_signal.emit(f"Error finding GGUF file: {e}")
            return None
