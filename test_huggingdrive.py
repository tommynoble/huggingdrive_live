#!/usr/bin/env python3
"""
Test script for HuggingDrive
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import psutil
        print("✅ psutil imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import psutil: {e}")
        return False
    
    try:
        from huggingface_hub import snapshot_download, HfApi
        print("✅ huggingface_hub imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import huggingface_hub: {e}")
        return False
    
    try:
        from PyQt6.QtWidgets import QApplication
        print("✅ PyQt6 imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import PyQt6: {e}")
        return False
    
    try:
        import torch
        print("✅ torch imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import torch: {e}")
        return False
    
    return True

def test_drive_detection():
    """Test external drive detection"""
    print("\nTesting drive detection...")
    
    try:
        from huggingdrive_cli import HuggingDriveCLI
        cli = HuggingDriveCLI()
        drives = cli.get_external_drives()
        
        if drives:
            print(f"✅ Found {len(drives)} external drive(s):")
            for drive in drives:
                print(f"   - {drive['mountpoint']} ({drive['free_gb']:.1f}GB free)")
        else:
            print("⚠️  No external drives detected (this is normal if no drives are connected)")
        
        return True
    except Exception as e:
        print(f"❌ Drive detection failed: {e}")
        return False

def test_config_management():
    """Test configuration management"""
    print("\nTesting configuration management...")
    
    try:
        from huggingdrive_cli import HuggingDriveCLI
        cli = HuggingDriveCLI()
        
        # Test config loading
        config = cli.config
        print("✅ Configuration loaded successfully")
        
        # Test config saving
        cli.save_config()
        print("✅ Configuration saved successfully")
        
        return True
    except Exception as e:
        print(f"❌ Configuration management failed: {e}")
        return False

def test_model_search():
    """Test model search functionality"""
    print("\nTesting model search...")
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        models_gen = api.list_models(search="gpt2", limit=3)
        models = list(models_gen)  # Convert generator to list
        
        if models:
            print(f"✅ Found {len(models)} models in search")
            for model in models[:2]:  # Show first 2
                print(f"   - {model.modelId}")
        else:
            print("⚠️  No models found in search")
        
        return True
    except Exception as e:
        print(f"❌ Model search failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 HuggingDrive Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_drive_detection,
        test_config_management,
        test_model_search
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! HuggingDrive is ready to use.")
        print("\nNext steps:")
        print("1. Connect an external drive")
        print("2. Run: python3 huggingdrive.py (for GUI)")
        print("3. Run: python3 huggingdrive_cli.py --help (for CLI)")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("You may need to install missing dependencies:")
        print("  pip3 install -r requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 