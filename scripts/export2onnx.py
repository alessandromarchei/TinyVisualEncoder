#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import torch
import onnx
import onnxruntime as ort

from models.tiny_visual_frontend import TinyVisualFrontend


def parse_args():
    p = argparse.ArgumentParser(
        description="Export TinyVisualFrontend to dynamic-shape ONNX"
    )

    p.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Path to .pt/.pth weights",
    )

    p.add_argument(
        "--output",
        required=True,
        type=str,
        help="Output .onnx",
    )

    p.add_argument(
        "--embedding-dim",
        type=int,
        default=512,
    )

    p.add_argument(
        "--temporal-kernel",
        type=int,
        default=5,
    )

    p.add_argument(
        "--stem-channels",
        type=int,
        default=16,
    )

    p.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=[24, 32, 64, 96],
    )

    p.add_argument(
        "--expand",
        type=float,
        default=2.0,
    )

    p.add_argument(
        "--opset",
        type=int,
        default=17,
    )

    return p.parse_args()


def extract_state_dict(ckpt):

    if not isinstance(ckpt, dict):
        raise RuntimeError(
            f"Checkpoint must be a dict, got {type(ckpt)}"
        )

    # Common checkpoint containers
    for key in (
        "state_dict",
        "model_state_dict",
        "model",
        "visual_frontend",
    ):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]

    # Otherwise assume checkpoint itself is state_dict
    return ckpt


def clean_state_dict(state_dict):

    cleaned = {}

    prefixes = (
        "module.",
        "visual_frontend.",
    )

    for key, value in state_dict.items():

        new_key = key

        changed = True

        while changed:
            changed = False

            for prefix in prefixes:

                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True

        cleaned[new_key] = value

    return cleaned


def main():

    args = parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print(" TinyVisualFrontend dynamic ONNX export")
    print("=" * 72)

    print(f"Checkpoint       : {checkpoint_path}")
    print(f"Output           : {output_path}")
    print(f"Embedding dim    : {args.embedding_dim}")
    print(f"Temporal kernel  : {args.temporal_kernel}")
    print(f"Stem channels    : {args.stem_channels}")
    print(f"Widths           : {args.widths}")
    print(f"Expand           : {args.expand}")
    print(f"Opset            : {args.opset}")

    # ============================================================
    # Model
    # ============================================================

    model = TinyVisualFrontend(
        embedding_dim=args.embedding_dim,
        temporal_kernel=args.temporal_kernel,
        stem_channels=args.stem_channels,
        widths=tuple(args.widths),
        expand=args.expand,
    )

    # ============================================================
    # Load weights
    # ============================================================

    print("\nLoading weights...")

    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = extract_state_dict(ckpt)
    state_dict = clean_state_dict(state_dict)

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    print("Weights loaded successfully.")

    # ============================================================
    # Dummy input
    #
    # IMPORTANT:
    #
    # T = arbitrary tracing value.
    # It will NOT be fixed in the exported ONNX.
    #
    # B = 1 fixed
    # C = 1 fixed
    # H = 112 fixed
    # W = 112 fixed
    #
    # ============================================================

    dummy_T = 25

    dummy = torch.randn(
        dummy_T,
        1,
        1,
        112,
        112,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        y = model(dummy)

    print()
    print("PyTorch")
    print("-" * 72)
    print(f"Input  : {tuple(dummy.shape)}")
    print(f"Output : {tuple(y.shape)}")

    # ============================================================
    # Export
    # ============================================================

    print()
    print("Exporting dynamic ONNX...")

    torch.onnx.export(
        model,
        dummy,
        str(output_path),

        input_names=[
            "visual_input",
        ],

        output_names=[
            "visual_embedding",
        ],

        # ========================================================
        # CRITICAL PART
        #
        # Input:
        #   [T, 1, 1, 112, 112]
        #
        # Output:
        #   [T, 1, D]
        #
        # axis 0 is dynamic on BOTH.
        # ========================================================

        dynamic_axes={
            "visual_input": {
                0: "num_frames",
                1: "batch_size",
            },
            "visual_embedding": {
                0: "num_frames",
                1: "batch_size",
            },
        },

        opset_version=args.opset,
        do_constant_folding=True,
    )

    print(f"Saved: {output_path}")

    # ============================================================
    # ONNX validation
    # ============================================================

    print()
    print("Checking ONNX model...")

    onnx_model = onnx.load(
        str(output_path)
    )

    onnx.checker.check_model(
        onnx_model
    )

    print("ONNX checker: OK")

    # ============================================================
    # Print ONNX interface
    # ============================================================

    session = ort.InferenceSession(
        str(output_path),
        providers=[
            "CPUExecutionProvider",
        ],
    )

    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]

    print()
    print("ONNX interface")
    print("-" * 72)

    print(
        f"Input  : {input_info.name} "
        f"{input_info.shape}"
    )

    print(
        f"Output : {output_info.name} "
        f"{output_info.shape}"
    )

    # ============================================================
    # IMPORTANT:
    # Test MANY temporal lengths
    # ============================================================

    test_lengths = [
        1,
        2,
        3,
        5,
        7,
        10,
        13,
        25,
        37,
        50,
        73,
        100,
        127,
    ]

    print()
    print("=" * 72)
    print(" Dynamic temporal-shape validation")
    print("=" * 72)

    max_global_error = 0.0

    for T in test_lengths:

        x = torch.randn(
            T,
            1,
            1,
            112,
            112,
            dtype=torch.float32,
        )

        # --------------------------------------------------------
        # PyTorch
        # --------------------------------------------------------

        with torch.inference_mode():
            y_pt = model(x).cpu().numpy()

        # --------------------------------------------------------
        # ONNX
        # --------------------------------------------------------

        y_onnx = session.run(
            ["visual_embedding"],
            {
                "visual_input": x.numpy(),
            },
        )[0]

        # --------------------------------------------------------
        # Shape check
        # --------------------------------------------------------

        expected_shape = (
            T,
            1,
            args.embedding_dim,
        )

        if y_onnx.shape != expected_shape:
            raise RuntimeError(
                f"T={T}: expected {expected_shape}, "
                f"got {y_onnx.shape}"
            )

        # --------------------------------------------------------
        # Numerical comparison
        # --------------------------------------------------------

        abs_error = np.abs(
            y_pt - y_onnx
        )

        max_error = float(
            abs_error.max()
        )

        mean_error = float(
            abs_error.mean()
        )

        max_global_error = max(
            max_global_error,
            max_error,
        )

        print(
            f"T={T:4d} | "
            f"shape={str(y_onnx.shape):18s} | "
            f"max_err={max_error:.3e} | "
            f"mean_err={mean_error:.3e}"
        )

        np.testing.assert_allclose(
            y_pt,
            y_onnx,
            rtol=1e-4,
            atol=1e-5,
        )

    print()
    print("=" * 72)
    print(" EXPORT SUCCESSFUL")
    print("=" * 72)
    print(f"ONNX             : {output_path}")
    print(f"Max global error : {max_global_error:.3e}")
    print()
    print("Temporal dimension T is dynamic.")
    print(
        "Accepted input shape: "
        "[T, 1, 1, 112, 112]"
    )
    print(
        "Output shape:         "
        f"[T, 1, {args.embedding_dim}]"
    )


if __name__ == "__main__":
    main()