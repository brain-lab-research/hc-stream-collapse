"""Zero-shot perplexity on held-out benchmarks (WikiText-103, C4)."""
import os
import math
from typing import Dict, Optional

import numpy as np
import torch

BENCHMARKS = {
    "wt_val": "data/wikitext/val.bin",
    "wt_test": "data/wikitext/test.bin",
    "c4_test": "data/c4/test.bin",
}


def available_benchmarks() -> Dict[str, str]:
    return {name: path for name, path in BENCHMARKS.items() if os.path.exists(path)}


def loss_to_ppl(loss: float) -> float:
    return math.exp(min(20.0, loss))


@torch.no_grad()
def estimate_benchmark_loss(
    model,
    data_path: str,
    *,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    device,
    ctx,
    seed: int = 1337,
) -> float:
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    if len(data) <= block_size + 1:
        return float("nan")

    rng = np.random.default_rng(seed)
    losses = []
    model.eval()
    for _ in range(eval_iters):
        ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
        x = torch.stack([
            torch.from_numpy(data[i : i + block_size].astype(np.int64).copy())
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64).copy())
            for i in ix
        ])
        x, y = x.to(device), y.to(device)
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def evaluate_benchmarks(
    model,
    *,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    device,
    ctx,
    seed: int = 1337,
    benchmarks: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    benchmarks = benchmarks or available_benchmarks()
    out = {}
    for name, path in benchmarks.items():
        out[name] = estimate_benchmark_loss(
            model,
            path,
            eval_iters=eval_iters,
            batch_size=batch_size,
            block_size=block_size,
            device=device,
            ctx=ctx,
            seed=seed,
        )
    return out


if __name__ == "__main__":
    import argparse
    from contextlib import nullcontext

    from model import GPT, GPTConfig

    parser = argparse.ArgumentParser(description="Evaluate checkpoint on WT103 / C4")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--eval_iters", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    model_args = checkpoint["model_args"]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(args.device)
    model.eval()

    device_type = "cuda" if "cuda" in args.device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    results = evaluate_benchmarks(
        model,
        eval_iters=args.eval_iters,
        batch_size=args.batch_size,
        block_size=args.block_size,
        device=args.device,
        ctx=ctx,
    )
    if not results:
        raise SystemExit("No benchmark .bin files found. Run data/wikitext/prepare.py and data/c4/prepare.py first.")

    print(f"Checkpoint: {ckpt_path}")
    for name, loss in results.items():
        print(f"{name:8s}  loss={loss:.4f}  ppl={loss_to_ppl(loss):.2f}")
