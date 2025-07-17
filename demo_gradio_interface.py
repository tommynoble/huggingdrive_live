#!/usr/bin/env python3
"""
Demo script showing Gradio integration with HuggingDrive
This script demonstrates how to launch a web interface for a model
"""

import sys
from pathlib import Path

# Add the huggingdrive directory to the path
sys.path.insert(0, str(Path(__file__).parent))


def demo_gradio_interface():
    """Demonstrate the Gradio interface functionality"""

    print("🚀 HuggingDrive Gradio Integration Demo")
    print("=" * 50)

    try:
        from huggingdrive.gradio_manager import GradioInterfaceManager

        # Create the manager
        manager = GradioInterfaceManager()
        print("✅ Gradio manager created")

        # Example model paths (you would replace these with actual downloaded models)
        example_models = [
            {"name": "gpt2", "path": "models/gpt2", "type": "text-generation"},
            {
                "name": "bert-base-uncased",
                "path": "models/bert-base-uncased",
                "type": "text-classification",
            },
            {
                "name": "Helsinki-NLP/opus-mt-en-fr",
                "path": "models/Helsinki-NLP/opus-mt-en-fr",
                "type": "translation",
            },
        ]

        print("\n📋 Available interface types:")
        print("1. Text Generation - for GPT, LLaMA, Mistral models")
        print("2. Text Classification - for BERT, RoBERTa models")
        print("3. Translation - for T5, Marian models")
        print("4. Chat Interface - default for most models")

        print("\n🔧 How to use:")
        print("1. Download a model using HuggingDrive")
        print("2. Click 'Test Model' in the main interface")
        print("3. Click '🌐 Launch Web Interface' button")
        print("4. A web browser will open with the model interface")
        print("5. Share the URL with others to let them use your model!")

        print("\n💡 Features:")
        print("• Automatic model type detection")
        print("• Model-specific interfaces")
        print("• Real-time parameter adjustment")
        print("• Mobile-friendly web interface")
        print("• Shareable URLs")

        print("\n🎯 Example workflow:")
        print("1. Download 'gpt2' model")
        print("2. Launch web interface")
        print("3. Get URL like: http://localhost:7860")
        print("4. Share URL with friends")
        print("5. Everyone can use your model through the web!")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure Gradio is installed: pip install gradio")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def show_installation_instructions():
    """Show installation instructions"""
    print("\n📦 Installation Instructions:")
    print("1. Install Gradio: pip install gradio")
    print("2. Install HuggingDrive dependencies: pip install -r requirements.txt")
    print("3. Run HuggingDrive: python3 huggingdrive.py")
    print("4. Download a model and click 'Launch Web Interface'")


def main():
    """Main demo function"""
    success = demo_gradio_interface()

    if not success:
        show_installation_instructions()

    print("\n✨ Demo completed!")


if __name__ == "__main__":
    main()
