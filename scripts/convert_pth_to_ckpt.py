#!/usr/bin/env python3
"""
Convert a .pth checkpoint to .ckpt format expected by TransfuserAgent.
The agent expects a PyTorch Lightning checkpoint with state_dict under "state_dict" key.
"""

import torch
import argparse
from pathlib import Path


def convert_pth_to_ckpt(pth_path: str, ckpt_path: str, prefix: str = "agent."):
    """
    Convert .pth file to .ckpt format.

    Args:
        pth_path: Path to input .pth file
        ckpt_path: Path to output .ckpt file
        prefix: Prefix to add to state dict keys (default: "agent.")
    """
    print(f"Loading {pth_path}...")

    # Load the .pth file
    state_dict = torch.load(pth_path, map_location='cpu')

    # Check if it's already in the right format
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        print("Input already has 'state_dict' key, using as-is")
        checkpoint = state_dict
    else:
        # Wrap it in the expected format
        print(f"Wrapping state_dict and adding '{prefix}' prefix to keys...")

        # Add prefix to all keys if needed
        if prefix:
            state_dict = {f"{prefix}{k}": v for k, v in state_dict.items()}

        checkpoint = {
            "state_dict": state_dict,
            "pytorch-lightning_version": "2.0.0",  # Add metadata for compatibility
        }

    # Save as .ckpt
    print(f"Saving to {ckpt_path}...")
    torch.save(checkpoint, ckpt_path)
    print("Done!")

    # Print some info
    num_params = sum(p.numel() for p in checkpoint["state_dict"].values() if isinstance(p, torch.Tensor))
    print(f"Total parameters: {num_params:,}")
    print(f"Number of keys: {len(checkpoint['state_dict'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .pth to .ckpt format")
    parser.add_argument("pth_path", type=str, help="Path to input .pth file")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Path to output .ckpt file (default: same as input with .ckpt extension)")
    parser.add_argument("--prefix", "-p", type=str, default="agent.",
                        help="Prefix to add to state dict keys (default: 'agent.')")
    parser.add_argument("--no-prefix", action="store_true",
                        help="Don't add any prefix to keys")

    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        output_path = Path(args.pth_path).with_suffix('.ckpt')
    else:
        output_path = Path(args.output)

    # Set prefix
    prefix = "" if args.no_prefix else args.prefix

    # Convert
    convert_pth_to_ckpt(args.pth_path, str(output_path), prefix=prefix)
