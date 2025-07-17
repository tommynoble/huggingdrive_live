#!/usr/bin/env python3
"""
Test script to verify that HuggingDrive can now download regular models
"""

import sys
import os
from pathlib import Path

# Add the huggingdrive directory to the path
sys.path.insert(0, str(Path(__file__).parent / "huggingdrive"))

from downloader import ModelDownloader
from PyQt6.QtCore import QCoreApplication


def test_regular_model_download():
    """Test downloading a regular Hugging Face model"""

    # Create QApplication for Qt signals
    app = QCoreApplication(sys.argv)

    # Test with a small, public model
    model_name = "microsoft/DialoGPT-small"  # Small model for testing
    target_path = "/tmp/test_download"

    print(f"Testing download of {model_name} to {target_path}")

    # Create downloader
    downloader = ModelDownloader(model_name, target_path)

    # Connect signals
    def on_progress(message):
        print(f"Progress: {message}")

    def on_finished(success, message):
        print(f"Download finished: {success} - {message}")
        app.quit()

    downloader.progress_signal.connect(on_progress)
    downloader.finished_signal.connect(on_finished)

    # Start download
    downloader.start()

    # Run the event loop
    app.exec()

    print("Test completed!")


if __name__ == "__main__":
    test_regular_model_download()
