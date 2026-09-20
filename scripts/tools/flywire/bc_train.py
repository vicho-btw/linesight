"""
Behaviour-clone a human lap: cross-entropy on the action the human actually pressed.

distill_flybrain.py cannot be reused here. It distils a NETWORK teacher, matching its Q-values
with a KL term -- but a human replay has no Q-values, only "at this frame, this key was down".
So the objective is plain supervised classification over the 12 actions, with the Q head read
as logits.

Two things about this data deserve care:

  * It is tiny. One lap at 10 ms is ~4,950 frames, where the earlier Q-distillation had orders
    of magnitude more. Overfitting is the default outcome, so the visual encoder is copied from
    an existing policy and frozen -- it already knows how to see this game -- and only the
    decision layers are trained.

  * It is a single trajectory. Behaviour cloning on one trajectory compounds error: the first
    small deviation puts the car in a state the dataset never covers, where the next error is
    likelier. Validation accuracy will look good anyway, because held-out frames come from the
    same lap. Only driving it reveals whether it holds the line.

    python scripts/tools/flywire/bc_train.py --data data/szymix_bc --out flybrain_szymix_bc
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def load_dataset(data_dirs):
    """Load every shard from one or more directories.

    The expert demonstration and the corrective states live in separate directories and are
    trained on together: the demonstration supplies the ideal line, the corrective states supply
    what to do once the car is off it. Copying hundreds of megabytes to merge them would be the
    obvious alternative and is pure waste.
    """
    if isinstance(data_dirs, (str, Path)):
        data_dirs = [data_dirs]
    shards = [s for d in data_dirs for s in sorted(Path(d).glob("*.npz"))]
    if not shards:
        raise SystemExit(f"no shards in {', '.join(str(d) for d in data_dirs)}")
    frames, floats, actions = [], [], []
    for s in shards:
        z = np.load(s)
        frames.append(z["frames"])
        floats.append(z["floats"])
        actions.append(z["actions"])
    frames = np.concatenate(frames)
    floats = np.concatenate(floats)
    actions = np.concatenate(actions)
    print(f"loaded {len(frames):,} frames from {len(shards)} shard(s) ({frames.nbytes / 2**20:.0f} MiB raw)")
    return frames, floats, actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, nargs="+", default=[ROOT / "data" / "szymix_bc"])
    ap.add_argument("--init-from", default=None, help="run whose visual encoder to copy and freeze")
    ap.add_argument("--out", default="flybrain_szymix_bc")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=1.0, help="temperature on the Q head read as logits")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from config_files import config_copy
    from trackmania_rl.agents.iqn import IQN_Network

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    frames, floats, actions = load_dataset(args.data)
    n_actions = len(config_copy.inputs)
    hist = np.bincount(actions, minlength=n_actions)
    print("action distribution in the lap:")
    for i, c in enumerate(hist):
        if c:
            a = config_copy.inputs[i]
            d = f"{'A' if a['accelerate'] else '-'}{'B' if a['brake'] else '-'}{'L' if a['left'] else ('R' if a['right'] else '-')}"
            print(f"   {i:2d} [{d}]: {c:5d} ({100 * c / len(actions):.1f}%)")
    majority = 100.0 * hist.max() / len(actions)
    print(f"majority-class baseline: {majority:.1f}%  (any model must beat this to have learned anything)")

    config_copy.use_fly_brain = True
    student = IQN_Network(
        float_inputs_dim=config_copy.float_input_dim,
        float_hidden_dim=config_copy.float_hidden_dim,
        conv_head_output_dim=config_copy.conv_head_output_dim,
        dense_hidden_dimension=config_copy.dense_hidden_dimension,
        iqn_embedding_dimension=config_copy.iqn_embedding_dimension,
        n_actions=n_actions,
        float_inputs_mean=config_copy.float_inputs_mean,
        float_inputs_std=config_copy.float_inputs_std,
    ).to(dev)

    if args.init_from:
        src = torch.load(ROOT / "save" / args.init_from / "weights1.torch", weights_only=False)
        copied = 0
        with torch.no_grad():
            for k, v in student.state_dict().items():
                if k in src and src[k].shape == v.shape and (k.startswith("img_head") or k.startswith("float_feature")):
                    v.copy_(src[k])
                    copied += v.numel()
        for name, p in student.named_parameters():
            if name.startswith("img_head") or name.startswith("float_feature"):
                p.requires_grad_(False)
        print(f"copied {copied:,} encoder params from '{args.init_from}' and froze them")

    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"trainable parameters: {trainable:,}")

    frames_t = torch.from_numpy(frames)
    floats_t = torch.from_numpy(floats)
    act_t = torch.from_numpy(actions.astype(np.int64))

    idx = rng.permutation(len(frames))
    n_val = max(1, int(args.val_frac * len(idx)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    print(f"train {len(train_idx):,}  val {len(val_idx):,}")

    opt = torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    iqn_n = config_copy.iqn_n

    def batch_to(sel):
        img = frames_t[sel].to(dev, non_blocking=True).float().sub_(128).div_(128)
        flo = floats_t[sel].to(dev, non_blocking=True)
        a = act_t[sel].to(dev, non_blocking=True)
        return img, flo, a

    @torch.no_grad()
    def evaluate(sel_all, chunk=512):
        student.eval()
        correct = seen = 0
        for i in range(0, len(sel_all), chunk):
            sel = torch.from_numpy(sel_all[i : i + chunk])
            img, flo, a = batch_to(sel)
            q, _ = student(img, flo, config_copy.iqn_k)
            q = q.reshape(config_copy.iqn_k, len(sel), -1).mean(0)
            correct += (q.argmax(1) == a).sum().item()
            seen += len(sel)
        student.train()
        return 100.0 * correct / seen

    out_dir = ROOT / "save" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0

    print(f"\nbehaviour cloning: cross-entropy over {n_actions} actions\n")
    for ep in range(args.epochs):
        t0 = time.perf_counter()
        order = rng.permutation(train_idx)
        tot = 0.0
        nb = 0
        for i in range(0, len(order), args.batch):
            sel = torch.from_numpy(order[i : i + args.batch])
            img, flo, a = batch_to(sel)
            q, _ = student(img, flo, iqn_n)
            q = q.reshape(iqn_n, len(sel), -1).mean(0)
            loss = F.cross_entropy(q / args.temp, a)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 10.0)
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        va = evaluate(val_idx)
        ta = evaluate(train_idx[: len(val_idx)])
        flag = ""
        if va > best:
            best = va
            torch.save(student.state_dict(), out_dir / "weights1.torch")
            torch.save(student.state_dict(), out_dir / "weights2.torch")
            flag = "  <- saved"
        print(
            f"  epoch {ep + 1:3d}/{args.epochs}  loss {tot / max(nb, 1):.4f}  "
            f"train_acc {ta:5.1f}%  val_acc {va:5.1f}%  ({time.perf_counter() - t0:.1f}s){flag}"
        )

    print(f"\nbest validation accuracy {best:.1f}% (majority baseline {majority:.1f}%) -> save/{args.out}")
    print("Validation accuracy is measured on frames from the SAME lap, so it overstates how well this")
    print("policy will drive. Verify with: python scripts/tools/flywire/eval_agent.py --run", args.out, "--fly")


if __name__ == "__main__":
    main()
