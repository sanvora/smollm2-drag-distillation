#!/usr/bin/env python3
"""
export_onnx.py — Export a fine-tuned SmolLM2-360M checkpoint to ONNX int8.

Usage:
    python scripts/finetune/export_onnx.py --checkpoint ./checkpoint
    python scripts/finetune/export_onnx.py --checkpoint ./checkpoint --output ./onnx-export
    python scripts/finetune/export_onnx.py --help

Requirements:
    pip install optimum[exporters] onnx onnxruntime transformers

The exported model is compatible with @huggingface/transformers.js (WebGPU/WASM).
"""

import argparse
import os
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export fine-tuned SmolLM2-360M to ONNX int8 for browser inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint",
        default="./checkpoint",
        help="Path to the fine-tuned model checkpoint directory (default: ./checkpoint)",
    )
    parser.add_argument(
        "--output",
        default="./onnx-export",
        help="Output directory for ONNX files (default: ./onnx-export)",
    )
    parser.add_argument(
        "--task",
        default="text-generation-with-past",
        help="Optimum export task (default: text-generation-with-past)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the export command without executing it",
    )
    return parser.parse_args()


def get_optimum_cli() -> list:
    """Return the command to invoke optimum-cli, preferring the current venv."""
    scripts_dir = os.path.dirname(sys.executable)
    for name in ["optimum-cli.exe", "optimum-cli"]:
        candidate = os.path.join(scripts_dir, name)
        if os.path.exists(candidate):
            return [candidate]
    # Fallback: invoke via python -m
    return [sys.executable, "-m", "optimum.exporters.onnx.__main__"]


def check_prerequisites() -> None:
    """Ensure optimum-cli is available."""
    cli = get_optimum_cli()
    try:
        subprocess.run(cli + ["--help"], capture_output=True, text=True)
        print(f"optimum-cli: found ({cli[0]})")
    except FileNotFoundError:
        print(
            "Error: optimum-cli not found.\n"
            "Install with: pip install 'optimum[exporters]'",
            file=sys.stderr,
        )
        sys.exit(1)


def validate_checkpoint(checkpoint_path: str) -> None:
    """Verify the checkpoint directory contains expected files."""
    required = ["config.json", "tokenizer_config.json"]
    missing = [f for f in required if not os.path.exists(os.path.join(checkpoint_path, f))]
    if missing:
        print(
            f"Error: checkpoint directory '{checkpoint_path}' is missing: {', '.join(missing)}\n"
            "Ensure you have a valid fine-tuned model saved at that path.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Detect weight format
    has_safetensors = any(f.endswith(".safetensors") for f in os.listdir(checkpoint_path))
    has_bin = any(f.endswith(".bin") for f in os.listdir(checkpoint_path))
    if not (has_safetensors or has_bin):
        print(
            f"Warning: No .safetensors or .bin weight files found in '{checkpoint_path}'.",
            file=sys.stderr,
        )


def export(args: argparse.Namespace) -> None:
    checkpoint = os.path.abspath(args.checkpoint)
    output = os.path.abspath(args.output)

    cmd = get_optimum_cli() + [
        "export",
        "onnx",
        "--task", args.task,
        "--int8",    # transformer-aware int8 quantization (replaces quantize_onnx.py)
        "--model", checkpoint,
        output,
    ]

    print("Export command:")
    print(" ", " ".join(cmd))
    print()

    if args.dry_run:
        print("Dry run — skipping execution.")
        return

    validate_checkpoint(checkpoint)
    os.makedirs(output, exist_ok=True)

    print(f"Exporting '{checkpoint}' -> '{output}' ...")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nExport failed with exit code {result.returncode}.", file=sys.stderr)
        sys.exit(result.returncode)

    # Verify output
    onnx_files = [f for f in os.listdir(output) if f.endswith(".onnx")]
    if not onnx_files:
        print("Warning: No .onnx files found in output directory.", file=sys.stderr)
        sys.exit(1)

    print("\nExport complete. Files:")
    for fname in sorted(os.listdir(output)):
        size_mb = os.path.getsize(os.path.join(output, fname)) / 1e6
        print(f"  {fname}  ({size_mb:.1f} MB)")

    print(f"\nNext step: push to HuggingFace Hub:")
    print(f"  huggingface-cli upload <your-hf-username>/smollm2-drag {output}")


def main() -> None:
    args = parse_args()

    if not args.dry_run:
        check_prerequisites()

    export(args)


if __name__ == "__main__":
    main()
