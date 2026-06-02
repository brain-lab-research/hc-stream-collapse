#!/usr/bin/env python3
"""Load an mhc_lite_lss checkpoint and print LSS coefficients (per-stream scaling before layer 0)."""
import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description="Inspect LSS weights from mhc_lite_lss checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pt file")
    parser.add_argument("--key", type=str, default="expand_stream.lss", help="State dict key for LSS")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    sd = ckpt.get("model", ckpt)
    if not isinstance(sd, dict):
        raise SystemExit("Checkpoint has no 'model' state dict")

    key = args.key
    if key not in sd:
        print(f"Key '{key}' not found. Available keys (first 30):")
        for k in list(sd.keys())[:30]:
            print(f"  {k}")
        if len(sd) > 30:
            print(f"  ... and {len(sd) - 30} more")
        raise SystemExit(1)

    h = sd[key]
    print(f"{key}: shape {h.shape} (num_streams={h.shape[0]}, dim={h.shape[1]})")
    print(f"  mean={h.float().mean().item():.6f}, std={h.float().std().item():.6f}")
    print(f"  min={h.float().min().item():.6f}, max={h.float().max().item():.6f}")


if __name__ == "__main__":
    main()
