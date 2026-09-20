"""
Sanity-check and benchmark the connectome trunk before committing a training run to it.

Checks, in order:
    1. the connectome loads and the super_class groups are populated
    2. a forward pass produces finite output at the expected shape
    3. activity neither dies nor diverges across settle steps
    4. gradients actually reach the connectome weights (the sparse autograd path works)
    5. forward+backward wall time and peak GPU memory at the real training batch size
    6. the same numbers for the dense trunk it replaces, for comparison

Run:  python scripts/tools/flywire/bench_flybrain.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from trackmania_rl.agents.flybrain import FlyBrain  # noqa: E402


def human(n: int) -> str:
    return f"{n:,}"


def bench(fn, n_warmup: int = 3, n_iter: int = 10):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / n_iter
    peak = torch.cuda.max_memory_allocated() / 2**30
    return dt, peak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--connectome", type=Path, default=Path(__file__).resolve().parents[3] / "data" / "flywire" / "connectome_783.npz")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--settle-steps", type=int, default=4)
    parser.add_argument("--readout-dim", type=int, default=512)
    args = parser.parse_args()

    if not args.connectome.exists():
        raise SystemExit(f"missing {args.connectome}\nRun scripts/tools/flywire/build_connectome.py first.")

    torch.manual_seed(0)
    dev = "cuda"

    print("=" * 78)
    print("1. loading connectome")
    print("=" * 78)
    brain = FlyBrain(
        connectome_path=args.connectome,
        visual_channels=32,
        visual_h=11,
        visual_w=16,
        float_dim=256,
        readout_dim=args.readout_dim,
        n_settle_steps=args.settle_steps,
    ).to(dev)
    print(brain.extra_repr())

    n_params = sum(p.numel() for p in brain.parameters())
    print()
    print(f"  recurrent weights (connectome edges) : {human(brain.recurrent.values.numel())}")
    print(f"  optic injection weights              : {human(brain.optic_in.values.numel())}")
    print(f"  sensory projection                   : {human(sum(p.numel() for p in brain.sensory_proj.parameters()))}")
    print(f"  descending readout                   : {human(sum(p.numel() for p in brain.readout.parameters()))}")
    print(f"  per-neuron bias + leak               : {human(brain.neuron_bias.numel() + brain.neuron_leak.numel())}")
    print(f"  TOTAL                                : {human(n_params)}")

    # What the dense trunk it replaces would cost.
    dense_trunk = 5888 * 512 * 2 + 64 * 5888
    print(f"  (dense trunk being replaced          : {human(dense_trunk)})")

    print()
    print("=" * 78)
    print("2/3. forward pass, activity statistics across settle steps")
    print("=" * 78)
    B = args.batch_size
    vis = torch.randn(B, 5632, device=dev)
    flo = torch.randn(B, 256, device=dev)

    # Re-run the settle loop manually so we can watch it evolve.
    with torch.no_grad():
        optic_drive = brain.optic_in(vis.t().contiguous())
        sensory_drive = brain.sensory_proj(flo).t()
        drive = vis.new_zeros((brain.n_neurons, B))
        drive.index_copy_(0, brain.optic_idx, optic_drive)
        drive.index_copy_(0, brain.sensory_idx, sensory_drive)
        drive = drive + brain.neuron_bias.unsqueeze(1)
        values = brain.recurrent_values()
        leak = torch.sigmoid(brain.neuron_leak).unsqueeze(1)
        x = torch.zeros_like(drive)
        for t in range(brain.n_settle_steps):
            rec = brain.recurrent(x, values=values)
            x = torch.nn.functional.leaky_relu(leak * x + rec + drive, negative_slope=0.01)
            if brain.step_norm:
                x = x * torch.rsqrt(x.pow(2).mean(dim=0, keepdim=True) + 1e-6) * brain.step_gain
            frac_active = (x > 0).float().mean().item()
            print(
                f"  step {t}: rms={x.pow(2).mean().sqrt().item():8.4f}  "
                f"max|x|={x.abs().max().item():9.4f}  active={frac_active * 100:5.1f}%"
            )

    out = brain(vis, flo)
    print(f"\n  output shape {tuple(out.shape)}  finite={torch.isfinite(out).all().item()}  std={out.std().item():.4f}")
    assert out.shape == (B, args.readout_dim)
    assert torch.isfinite(out).all(), "non-finite output"

    print()
    print("=" * 78)
    print("4. gradient reaches the connectome weights")
    print("=" * 78)
    brain.zero_grad(set_to_none=True)
    brain(vis, flo).pow(2).mean().backward()
    for name, p in [
        ("connectome weights", brain.recurrent.values),
        ("optic injection", brain.optic_in.values),
        ("neuron_bias", brain.neuron_bias),
        ("readout.weight", brain.readout.weight),
    ]:
        g = p.grad
        ok = g is not None and torch.isfinite(g).all() and g.abs().max() > 0
        nz = (g != 0).float().mean().item() * 100 if g is not None else 0.0
        print(f"  {name:26s} grad_ok={bool(ok)}  |g|max={g.abs().max().item():.3e}  nonzero={nz:5.1f}%")
        assert ok, f"no usable gradient for {name}"

    print()
    print("=" * 78)
    print(f"5. timing at batch_size={B}")
    print("=" * 78)

    def fwd_only():
        with torch.no_grad():
            brain(vis, flo)

    def fwd_bwd():
        brain.zero_grad(set_to_none=True)
        brain(vis, flo).pow(2).mean().backward()

    dt_f, mem_f = bench(fwd_only)
    dt_fb, mem_fb = bench(fwd_bwd)
    print(f"  forward           : {dt_f * 1e3:8.2f} ms   peak {mem_f:5.2f} GiB")
    print(f"  forward+backward  : {dt_fb * 1e3:8.2f} ms   peak {mem_fb:5.2f} GiB")

    vis1 = torch.randn(1, 5632, device=dev)
    flo1 = torch.randn(1, 256, device=dev)

    def infer1():
        with torch.no_grad():
            brain(vis1, flo1)

    dt_i, _ = bench(infer1, n_iter=30)
    print(f"  inference (batch 1): {dt_i * 1e3:7.2f} ms  -> {1 / dt_i:7.1f} actions/s")
    print("     (rollouts need ~20 actions/s per game instance at running_speed 80)")

    print()
    print("=" * 78)
    print("6. dense trunk baseline, same batch")
    print("=" * 78)
    a_head = torch.nn.Sequential(torch.nn.Linear(5888, 512), torch.nn.LeakyReLU(), torch.nn.Linear(512, 12)).to(dev)
    cat = torch.randn(B, 5888, device=dev)

    def dense_fwd_bwd():
        a_head.zero_grad(set_to_none=True)
        a_head(cat).pow(2).mean().backward()

    dt_d, mem_d = bench(dense_fwd_bwd)
    print(f"  dense A_head fwd+bwd: {dt_d * 1e3:8.3f} ms   peak {mem_d:5.2f} GiB")
    print(f"  fly brain is {dt_fb / max(dt_d, 1e-9):.1f}x the cost of the dense head it replaces")

    print()
    print("all checks passed.")


if __name__ == "__main__":
    main()
