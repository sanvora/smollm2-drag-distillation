#!/usr/bin/env python3
"""
quantize_onnx.py — Post-export int8 static quantization of an ONNX model.

Produces onnx/model_quantized.onnx (~400MB from 1.6GB fp32) which
@huggingface/transformers loads when dtype='q8' or dtype='int8'.

Usage:
    uv run python scripts/finetune/quantize_onnx.py
    uv run python scripts/finetune/quantize_onnx.py --input scripts/finetune/onnx-export/onnx/model.onnx
"""
import argparse
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default="scripts/finetune/onnx-export/onnx/model.onnx")
    p.add_argument("--output", default="scripts/finetune/onnx-export/onnx/model_quantized.onnx")
    return p.parse_args()


def main():
    args = parse_args()
    input_path  = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {input_path}  ({input_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Output: {output_path}")
    print("Quantizing to int8 (dynamic)...")

    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        str(input_path),
        str(output_path),
        weight_type=QuantType.QInt8,
    )

    out_size = output_path.stat().st_size / 1e6
    print(f"\nDone. Output: {output_path}  ({out_size:.1f} MB)")
    print("Reduction: {:.0f}%".format(
        100 * (1 - out_size / (input_path.stat().st_size / 1e6))
    ))


if __name__ == "__main__":
    main()
