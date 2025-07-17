#!/usr/bin/env python3
"""
HuggingDrive GGUF Converter CLI
Standalone command-line tool for converting Hugging Face models to GGUF format
"""

import argparse
import sys
import os
from pathlib import Path

# Add the huggingdrive package to the path
sys.path.insert(0, str(Path(__file__).parent))

from huggingdrive.gguf_converter import AdvancedGGUFConverter, GGUFConversionManager


def main():
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face models to GGUF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/model --output /path/to/output
  %(prog)s /path/to/model --quantization q4_K_M --context-length 8192
  %(prog)s /path/to/model --list-quantizations
  %(prog)s /path/to/model --validate
        """
    )
    
    parser.add_argument(
        "model_path",
        nargs="?",
        help="Path to the Hugging Face model directory"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output directory for the converted GGUF model (auto-generated if not specified)"
    )
    
    parser.add_argument(
        "--quantization", "-q",
        default="q4_K_M",
        help="Quantization method (default: q4_K_M)"
    )
    
    parser.add_argument(
        "--context-length", "-c",
        type=int,
        default=4096,
        help="Context length in tokens (default: 4096)"
    )
    
    parser.add_argument(
        "--method",
        choices=["llama-cpp-python", "ctransformers", "auto"],
        default="auto",
        help="Conversion method (default: auto)"
    )
    
    parser.add_argument(
        "--list-quantizations",
        action="store_true",
        help="List available quantization options and exit"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate model for conversion and exit"
    )
    
    parser.add_argument(
        "--estimate-size",
        action="store_true",
        help="Estimate conversion size and exit"
    )
    
    parser.add_argument(
        "--recommendations",
        action="store_true",
        help="Show conversion recommendations and exit"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Initialize the conversion manager
    manager = GGUFConversionManager()
    
    # List quantizations
    if args.list_quantizations:
        print("Available quantization options:")
        print("=" * 50)
        for key, description in manager.get_quantization_options().items():
            print(f"{key:12} - {description}")
        return
    
    # Check if model path is provided
    if not args.model_path:
        parser.error("Model path is required (unless using --list-quantizations)")
    
    model_path = Path(args.model_path)
    
    # Validate model
    if args.validate:
        is_valid, message = manager.validate_model_for_conversion(str(model_path))
        if is_valid:
            print(f"✅ {message}")
            return 0
        else:
            print(f"❌ {message}")
            return 1
    
    # Estimate size
    if args.estimate_size:
        input_mb, output_mb = manager.estimate_conversion_size(str(model_path), args.quantization)
        if input_mb > 0 and output_mb > 0:
            compression_ratio = input_mb / output_mb
            space_saved = input_mb - output_mb
            print(f"Input size:  {input_mb:.1f}MB")
            print(f"Output size: {output_mb:.1f}MB")
            print(f"Compression: {compression_ratio:.2f}x")
            print(f"Space saved: {space_saved:.1f}MB")
        else:
            print("Could not estimate conversion size")
        return 0
    
    # Show recommendations
    if args.recommendations:
        recommendations = manager.get_conversion_recommendations(str(model_path))
        if recommendations:
            print("Conversion Recommendations:")
            print("=" * 30)
            for key, value in recommendations.items():
                print(f"{key}: {value}")
        else:
            print("No recommendations available")
        return 0
    
    # Validate model before conversion
    is_valid, message = manager.validate_model_for_conversion(str(model_path))
    if not is_valid:
        print(f"❌ Model validation failed: {message}")
        return 1
    
    print(f"✅ {message}")
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = str(model_path.parent / f"{model_path.name}_gguf")
    
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    print(f"Quantization: {args.quantization}")
    print(f"Context length: {args.context_length}")
    print(f"Method: {args.method}")
    print("-" * 50)
    
    # Create converter
    use_llama_cpp = args.method in ["llama-cpp-python", "auto"]
    
    converter = AdvancedGGUFConverter(
        model_path=str(model_path),
        output_path=output_path,
        quantization=args.quantization,
        context_length=args.context_length,
        use_llama_cpp=use_llama_cpp
    )
    
    # Set up progress callbacks
    def progress_callback(message):
        if args.verbose:
            print(f"[PROGRESS] {message}")
    
    def stats_callback(stats):
        print(f"[STATS] {stats}")
    
    converter.progress_signal.connect(progress_callback)
    converter.conversion_stats_signal.connect(stats_callback)
    
    # Start conversion
    print("Starting GGUF conversion...")
    converter.start()
    converter.wait()
    
    # Check result
    if converter.isFinished():
        print("✅ GGUF conversion completed successfully!")
        return 0
    else:
        print("❌ GGUF conversion failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 