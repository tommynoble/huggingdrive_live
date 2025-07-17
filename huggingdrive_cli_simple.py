#!/usr/bin/env python3
"""
HuggingDrive Simple CLI - Command line interface without GUI dependencies
"""

import sys
import argparse
import os
from huggingdrive.drive_manager import DriveManager
from huggingdrive.downloader import ModelDownloader


def main():
    """Main function to run the CLI application"""
    parser = argparse.ArgumentParser(
        description="HuggingDrive CLI - Manage Hugging Face models on external drives"
    )
    parser.add_argument(
        "--list-drives", 
        action="store_true", 
        help="List available external drives"
    )
    parser.add_argument(
        "--search", 
        type=str, 
        help="Search for models on Hugging Face"
    )
    parser.add_argument(
        "--download", 
        type=str, 
        help="Download a specific model (provide model name)"
    )
    parser.add_argument(
        "--drive", 
        type=str, 
        help="Specify external drive path for operations"
    )
    parser.add_argument(
        "--list-models", 
        action="store_true", 
        help="List downloaded models on external drive"
    )
    
    args = parser.parse_args()
    
    if not any([args.list_drives, args.search, args.download, args.list_models]):
        parser.print_help()
        return
    
    # Initialize managers
    drive_manager = DriveManager()
    
    if args.list_drives:
        print("🔍 Scanning for external drives...")
        drives = drive_manager.get_external_drives()
        if drives:
            print(f"✅ Found {len(drives)} external drive(s):")
            for drive in drives:
                print(f"   - {drive['mountpoint']} ({drive['free_gb']:.1f}GB free)")
        else:
            print("❌ No external drives found")
    
    elif args.search:
        print(f"🔍 Search functionality requires GUI version")
        print("Please use the GUI version for model searching")
    
    elif args.download:
        if not args.drive:
            print("❌ Please specify a drive with --drive")
            return
        
        print(f"📥 Downloading model: {args.download}")
        print(f"📍 Target drive: {args.drive}")
        
        try:
            # Create downloader instance with required parameters
            downloader = ModelDownloader(args.download, args.drive)
            downloader.run()
            print("✅ Download completed successfully!")
        except Exception as e:
            print(f"❌ Download failed: {e}")
    
    elif args.list_models:
        if not args.drive:
            print("❌ Please specify a drive with --drive")
            return
        
        print(f"📋 Listing models on drive: {args.drive}")
        models = drive_manager.scan_for_models(args.drive)
        if models:
            print(f"✅ Found {len(models)} models:")
            for model in models:
                print(f"   - {model['name']} ({model['size']})")
        else:
            print("❌ No models found on drive")


if __name__ == "__main__":
    main() 