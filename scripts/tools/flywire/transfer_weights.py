"""
Warm-start the connectome network from a trained dense-trunk run.

The fly network keeps Linesight's input heads unchanged -- the convolutional image head and
the float feature extractor are identical, tensor for tensor, to the baseline's. Those are
the parts that learned to *see* a Trackmania track over millions of frames, and there is no
reason to make the connectome rediscover them. The output layers that read a 512-wide
embedding match too.

What cannot transfer is exactly the part being replaced: the baseline's first dueling layers
and its IQN embedding take a 5888-wide concatenation, where the fly network takes a 512-wide
readout decoded from descending neurons. Those stay randomly initialised, which is the point
of the experiment.

    python scripts/tools/flywire/transfer_weights.py --from hocko_run1 --to flybrain_hocko3

Note this does change the question being asked, from "can a connectome learn to drive from
nothing" to "can a connectome drive given a trained visual front-end". The second is the
fairer question anyway: a fly does not learn its optic lobe from scratch either.
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def as_state_dict(obj):
    return obj.state_dict() if hasattr(obj, "state_dict") else obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default="hocko_run1", help="run name to take weights from")
    ap.add_argument("--to", dest="dst", default="flybrain_hocko3", help="run name to create")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src_path = ROOT / "save" / args.src / "weights1.torch"
    if not src_path.exists():
        raise SystemExit(f"no checkpoint at {src_path}")

    from config_files import config_copy
    from trackmania_rl.agents.iqn import make_untrained_iqn_network

    if config_copy.run_name != args.dst:
        print(f"note: config_copy.run_name is '{config_copy.run_name}', creating '{args.dst}'")
    if not getattr(config_copy, "use_fly_brain", False):
        raise SystemExit("use_fly_brain is False in config; the target network would not have a connectome")

    print("building the connectome network ...")
    _, fly = make_untrained_iqn_network(jit=False, is_inference=False)
    target = fly.state_dict()

    donor = as_state_dict(torch.load(src_path, weights_only=False, map_location="cpu"))

    matched, shape_mismatch, absent = [], [], []
    for name, tensor in target.items():
        if name not in donor:
            absent.append(name)
        elif tuple(donor[name].shape) != tuple(tensor.shape):
            shape_mismatch.append((name, tuple(donor[name].shape), tuple(tensor.shape)))
        else:
            matched.append(name)

    transferred = 0
    for name in matched:
        target[name].copy_(donor[name].to(target[name].dtype))
        transferred += target[name].numel()

    print(f"\nTRANSFERRED from '{args.src}' ({len(matched)} tensors, {transferred:,} params)")
    groups = {}
    for n in matched:
        groups.setdefault(n.split(".")[0], []).append(n)
    for g, names in groups.items():
        n_params = sum(target[n].numel() for n in names)
        print(f"   {g:28s} {len(names)} tensors  {n_params:>10,} params")

    print(f"\nLEFT RANDOM - shape differs because this is the replaced trunk ({len(shape_mismatch)})")
    for n, a, b in shape_mismatch:
        print(f"   {n:28s} donor {str(a):16s} -> fly {b}")

    fly_only = [n for n in absent if "fly_brain" in n]
    other = [n for n in absent if "fly_brain" not in n]
    fly_params = sum(target[n].numel() for n in fly_only)
    print(f"\nCONNECTOME, no donor equivalent ({len(fly_only)} tensors, {fly_params:,} params)")
    if other:
        print(f"   plus {len(other)} other tensors with no match: {other[:4]}")

    if args.dry_run:
        print("\ndry run, nothing written")
        return

    out_dir = ROOT / "save" / args.dst
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} already exists and is not empty; pick another --to name")
    out_dir.mkdir(parents=True, exist_ok=True)

    fly.load_state_dict(target)
    torch.save(fly.state_dict(), out_dir / "weights1.torch")
    shutil.copyfile(out_dir / "weights1.torch", out_dir / "weights2.torch")
    print(f"\nwrote {out_dir / 'weights1.torch'} (and weights2)")
    print(f"set run_name = \"{args.dst}\" in config_files/config.py, then start training.")
    print("Optimizer state is deliberately not copied: its moments belong to a different trunk.")


if __name__ == "__main__":
    main()
