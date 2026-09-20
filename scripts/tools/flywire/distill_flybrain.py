"""
Distil a trained dense-trunk agent into the connectome network.

The conv image head and the float feature extractor are copied from the teacher and frozen:
they are architecturally identical in both networks and already know how to see a Trackmania
track, so the only thing under test is whether the connectome trunk can map those features
onto the teacher's decisions.

    python scripts/tools/flywire/distill_flybrain.py --data data/distill --out flybrain_distilled

Why the loss is a KL over softened Q, and not MSE
-------------------------------------------------
Measured on this teacher: Q has std ~2.7 while the advantage -- the spread *between* actions,
which is the only part that decides what the car does -- has std ~0.14, about 0.2% of the
variance. An MSE on Q therefore spends essentially all of its gradient learning how good a
situation is and almost none learning which action to take, and it looks like a triumph
while doing it (explained variance 0.99). Measured head to head on held-out data:

    MSE on Q                  argmax agreement 40.2%
    MSE on advantage          argmax agreement 47.9%
    KL on softmax(Q / 0.05)   argmax agreement 63.7%

A small auxiliary term keeps the value scale roughly right so the RL phase does not inherit
a wildly mis-scaled critic.

Reported metrics
----------------
argmax agreement understates policy quality, because disagreeing where two actions are worth
almost the same costs nothing. `regret` is the honest number: how much value the teacher
thinks the student gives up by choosing differently, in the teacher's own Q units.
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


def load_dataset(data_dir: Path, device: str, max_frames: int | None = None):
    shards = sorted(data_dir.glob("*.npz"))
    if not shards:
        raise SystemExit(f"no shards in {data_dir}; run collect_teacher_data.py first")
    frames, floats, qvals = [], [], []
    total = 0
    for s in shards:
        z = np.load(s)
        frames.append(z["frames"])
        floats.append(z["floats"])
        qvals.append(z["qvals"])
        total += len(z["frames"])
        if max_frames and total >= max_frames:
            break
    frames = np.concatenate(frames)
    floats = np.concatenate(floats)
    qvals = np.concatenate(qvals)
    print(f"loaded {len(frames):,} frames from {len(shards)} shards ({frames.nbytes / 2**20:.0f} MiB raw)")
    return frames, floats, qvals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "distill")
    ap.add_argument("--teacher", default="hocko_run1")
    ap.add_argument("--out", default="flybrain_distilled", help="run name to write weights into")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temp", type=float, default=0.05, help="softmax temperature for the KL loss")
    ap.add_argument("--value-weight", type=float, default=0.05, help="auxiliary weight on matching mean Q")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    from config_files import config_copy
    from trackmania_rl.agents.iqn import IQN_Network

    dev = "cuda"
    frames, floats, qvals = load_dataset(args.data, dev, args.max_frames)

    n = len(frames)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    n_val = int(n * args.val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    print(f"train {len(train_idx):,}  val {len(val_idx):,}")

    # Frames stay on the host as uint8 (a 90k-frame set is ~1.7 GiB); batches move per step.
    frames_t = torch.from_numpy(frames)
    floats_t = torch.from_numpy(floats)
    q_t = torch.from_numpy(qvals)

    def build(fly: bool):
        config_copy.use_fly_brain = fly
        return IQN_Network(
            float_inputs_dim=config_copy.float_input_dim,
            float_hidden_dim=config_copy.float_hidden_dim,
            conv_head_output_dim=config_copy.conv_head_output_dim,
            dense_hidden_dimension=config_copy.dense_hidden_dimension,
            iqn_embedding_dimension=config_copy.iqn_embedding_dimension,
            n_actions=len(config_copy.inputs),
            float_inputs_mean=config_copy.float_inputs_mean,
            float_inputs_std=config_copy.float_inputs_std,
        ).to(dev)

    teacher_sd = torch.load(ROOT / "save" / args.teacher / "weights1.torch", weights_only=False)
    student = build(True)

    copied = 0
    with torch.no_grad():
        for k, v in student.state_dict().items():
            if k in teacher_sd and teacher_sd[k].shape == v.shape and (
                k.startswith("img_head") or k.startswith("float_feature")
            ):
                v.copy_(teacher_sd[k])
                copied += v.numel()
    for name, p in student.named_parameters():
        if name.startswith("img_head") or name.startswith("float_feature"):
            p.requires_grad_(False)
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"copied {copied:,} params from teacher (frozen); {trainable:,} trainable")

    opt = torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    iqn_n = config_copy.iqn_n

    def batch_to(idx):
        img = frames_t[idx].to(dev, non_blocking=True).float().sub_(128).div_(128)
        flo = floats_t[idx].to(dev, non_blocking=True)
        tq = q_t[idx].to(dev, non_blocking=True)
        return img, flo, tq

    @torch.no_grad()
    def evaluate(idx_all, chunk=512):
        student.eval()
        agree = 0
        regret = 0.0
        seen = 0
        for i in range(0, len(idx_all), chunk):
            idx = torch.from_numpy(idx_all[i : i + chunk])
            img, flo, tq = batch_to(idx)
            q, _ = student(img, flo, config_copy.iqn_k)
            q = q.reshape(config_copy.iqn_k, len(idx), -1).mean(0)
            sa = q.argmax(1)
            ta = tq.argmax(1)
            agree += (sa == ta).sum().item()
            # value the teacher assigns to its own choice vs the student's choice
            regret += (tq.gather(1, ta[:, None]) - tq.gather(1, sa[:, None])).sum().item()
            seen += len(idx)
        student.train()
        return 100.0 * agree / seen, regret / seen

    print(f"\ndistilling: KL(softmax(Q/{args.temp})) + {args.value_weight} * MSE(mean Q)")
    best = -1.0
    out_dir = ROOT / "save" / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(args.epochs):
        t0 = time.perf_counter()
        order = rng.permutation(train_idx)
        tot = 0.0
        for i in range(0, len(order), args.batch):
            idx = torch.from_numpy(order[i : i + args.batch])
            img, flo, tq = batch_to(idx)
            q, _ = student(img, flo, iqn_n)
            q = q.reshape(iqn_n, len(idx), -1).mean(0)
            kl = F.kl_div(F.log_softmax(q / args.temp, 1), F.softmax(tq / args.temp, 1), reduction="batchmean")
            val = F.mse_loss(q.mean(1), tq.mean(1))
            loss = kl + args.value_weight * val
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 10.0)
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        acc, reg = evaluate(val_idx)
        flag = ""
        if acc > best:
            best = acc
            torch.save(student.state_dict(), out_dir / "weights1.torch")
            torch.save(student.state_dict(), out_dir / "weights2.torch")
            flag = "  <- saved"
        print(
            f"  epoch {ep:3d}  loss {tot / len(order):8.5f}  "
            f"val agreement {acc:5.1f}%  mean regret {reg:7.4f}  ({time.perf_counter() - t0:.0f}s){flag}"
        )

    print(f"\nbest val agreement {best:.1f}%  ->  {out_dir}")
    print("Hand to RL by setting run_name to this and starting with LOW epsilon and lr;")
    print("a fresh run restarts epsilon at 1.0 and would randomise this away immediately.")


if __name__ == "__main__":
    main()
