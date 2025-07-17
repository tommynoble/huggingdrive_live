#!/usr/bin/env python3
"""
HuggingDrive CLI - Command line interface for HuggingDrive
"""

import sys
from PyQt6.QtWidgets import QApplication
from huggingdrive.gui import HuggingDriveGUI


def main():
    """Main function to run the application"""
    app = QApplication(sys.argv)
    app.setApplicationName("HuggingDrive")
    app.setApplicationVersion("1.0.0")

    # Set application style
    app.setStyle("Fusion")

    # Create and show main window using the modular structure
    window = HuggingDriveGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
