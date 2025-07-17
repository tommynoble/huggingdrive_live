#!/usr/bin/env python3
"""
Test script for Gradio integration with HuggingDrive
"""

import sys
import os
from pathlib import Path

# Add the huggingdrive directory to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_gradio_import():
    """Test if Gradio can be imported"""
    try:
        import gradio as gr
        print("✅ Gradio imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import Gradio: {e}")
        return False

def test_gradio_manager():
    """Test if the Gradio manager can be imported"""
    try:
        from huggingdrive.gradio_manager import GradioInterfaceManager
        print("✅ GradioInterfaceManager imported successfully")
        
        # Test creating an instance
        manager = GradioInterfaceManager()
        print("✅ GradioInterfaceManager instance created successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import GradioInterfaceManager: {e}")
        return False
    except Exception as e:
        print(f"❌ Error creating GradioInterfaceManager: {e}")
        return False

def test_model_type_detection():
    """Test model type detection"""
    try:
        from huggingdrive.gradio_manager import GradioInterfaceManager
        manager = GradioInterfaceManager()
        
        # Test with a dummy config
        test_config = {
            "model_type": "gpt2",
            "architectures": ["GPT2LMHeadModel"]
        }
        
        # Create a temporary test directory
        test_dir = Path("test_model")
        test_dir.mkdir(exist_ok=True)
        
        config_file = test_dir / "config.json"
        import json
        with open(config_file, 'w') as f:
            json.dump(test_config, f)
        
        # Test detection
        model_type = manager.detect_model_type(str(test_dir))
        print(f"✅ Model type detection works: {model_type}")
        
        # Clean up
        import shutil
        shutil.rmtree(test_dir)
        
        return True
    except Exception as e:
        print(f"❌ Error testing model type detection: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Gradio Integration for HuggingDrive")
    print("=" * 50)
    
    tests = [
        ("Gradio Import", test_gradio_import),
        ("Gradio Manager Import", test_gradio_manager),
        ("Model Type Detection", test_model_type_detection),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Testing: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Gradio integration is ready.")
    else:
        print("⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main() 