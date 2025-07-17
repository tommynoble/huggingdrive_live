#!/usr/bin/env python3
"""
Example usage of HuggingDrive programmatically
"""

from huggingdrive_cli import HuggingDriveCLI
import sys


def main():
    """Example of using HuggingDrive in Python scripts"""

    print("🚀 HuggingDrive Programmatic Usage Example")
    print("=" * 50)

    # Initialize HuggingDrive
    cli = HuggingDriveCLI()

    # 1. List available external drives
    print("\n1. Checking for external drives...")
    drives = cli.get_external_drives()

    if not drives:
        print(
            "❌ No external drives found. Please connect an external drive and try again."
        )
        return

    print(f"✅ Found {len(drives)} external drive(s):")
    for drive in drives:
        print(f"   - {drive['mountpoint']} ({drive['free_gb']:.1f}GB free)")

    # 2. Set default drive (use the first available drive)
    default_drive = drives[0]["mountpoint"]
    print(f"\n2. Setting default drive to: {default_drive}")
    cli.set_default_drive(default_drive)

    # 3. Search for a model
    print("\n3. Searching for GPT-2 models...")
    cli.search_models("gpt2", limit=3)

    # 4. Install a small model (for demonstration)
    model_name = "gpt2"  # Small model for testing
    print(f"\n4. Installing model: {model_name}")

    try:
        cli.install_model(model_name, default_drive)
        print(f"✅ Successfully installed {model_name}")
    except Exception as e:
        print(f"❌ Failed to install {model_name}: {e}")
        print("This might be due to network issues or insufficient disk space.")
        return

    # 5. List installed models
    print("\n5. Listing installed models...")
    cli.list_models()

    # 6. Export configuration
    config_file = "example_config.json"
    print(f"\n6. Exporting configuration to {config_file}...")
    cli.export_config(config_file)

    print("\n🎉 Example completed successfully!")
    print("\nYou can now:")
    print(f"1. Use the GUI: python3 huggingdrive.py")
    print(f"2. Use the CLI: python3 huggingdrive_cli.py --help")
    print(f"3. Check your configuration: cat {config_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Example interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        sys.exit(1)
