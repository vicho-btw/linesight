"""
Track the fly-brain run's progress and benchmark it against the stock-architecture baseline.

Reads the tensorboard scalars both runs write, so it needs nothing from the training
processes and cannot disturb them. Appends one row per call to a CSV, so calling it on a
timer gives an hourly progress history.

Usage:
    python scripts/tools/flywire/track_progress.py              # print + append a row
    python scripts/tools/flywire/track_progress.py --no-append  # print only
    python scripts/tools/flywire/track_progress.py --history    # show the CSV so far
"""

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[3]
DNF = 300.0  # race time logged when the agent fails to finish

MAP = "hock"
TOTAL_ZONES = 8062  # len(maps/ESL-Hockolicious_0.5m_cl2.npy)
WORLD_RECORD = 53.76  # ESL-Hockolicious, tastyy
AUTHOR_TIME = 55.10


def load_scalars(run_dir: Path):
    if not run_dir.exists():
        return {}
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        out[tag] = (np.array([e.step for e in events]), np.array([e.value for e in events]))
    return out


def best_time(scalars, tag):
    """Best (lowest) finished race time, or None if the agent has never finished."""
    if tag not in scalars:
        return None, None
    steps, vals = scalars[tag]
    finished = vals < DNF
    if not finished.any():
        return None, None
    i = int(np.argmin(np.where(finished, vals, np.inf)))
    return float(vals[i]), int(steps[i])


def first_finish_step(scalars, tag):
    if tag not in scalars:
        return None
    steps, vals = scalars[tag]
    finished = np.flatnonzero(vals < DNF)
    return int(steps[finished[0]]) if len(finished) else None


def at_step(scalars, tag, step, window=0.1):
    """Baseline value near a given step, averaged over a +/-10% window to smooth noise."""
    if tag not in scalars:
        return None
    steps, vals = scalars[tag]
    lo, hi = step * (1 - window), step * (1 + window)
    sel = (steps >= lo) & (steps <= hi)
    if not sel.any():
        if step < steps.min():
            return None
        sel = steps <= step
        if not sel.any():
            return None
        return float(vals[sel][-1])
    return float(vals[sel].mean())


def fmt_time(t):
    return "--" if t is None else f"{t:.3f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="flybrain_hocko")
    ap.add_argument("--baseline", default="hocko_run1")
    ap.add_argument("--csv", type=Path, default=ROOT / "logs" / "flybrain_progress.csv")
    ap.add_argument("--no-append", action="store_true")
    ap.add_argument("--history", action="store_true")
    args = ap.parse_args()

    if args.history:
        if args.csv.exists():
            print(args.csv.read_text())
        else:
            print(f"no history yet at {args.csv}")
        return

    fly = load_scalars(ROOT / "tensorboard" / args.run)
    base = load_scalars(ROOT / "tensorboard" / args.baseline)

    if not fly:
        raise SystemExit(f"no tensorboard data for run '{args.run}' yet")

    zone_tag = f"single_zone_reached_trained_{MAP}"
    eval_tag = f"eval_race_time_trained_{MAP}"
    explo_tag = f"explo_race_time_trained_{MAP}"
    ratio_tag = f"race_time_ratio_{MAP}"

    steps, _ = fly[zone_tag] if zone_tag in fly else (np.array([0]), None)
    frames = int(steps.max())

    zone_now = at_step(fly, zone_tag, frames, window=0.25) or 0.0
    zone_max = float(fly[zone_tag][1].max()) if zone_tag in fly else 0.0
    ratio = at_step(fly, ratio_tag, frames, window=0.3)
    q_val = at_step(fly, f"avg_Q_trained_{MAP}", frames, window=0.5)

    fly_best, fly_best_step = best_time(fly, eval_tag)
    fly_first = first_finish_step(fly, eval_tag)

    base_zone = at_step(base, zone_tag, frames, window=0.25)
    base_best_here, _ = None, None
    if eval_tag in base:
        bsteps, bvals = base[eval_tag]
        sel = bsteps <= frames
        fin = bvals[sel] < DNF
        base_best_here = float(bvals[sel][fin].min()) if fin.any() else None
    base_first = first_finish_step(base, eval_tag)
    base_final, base_final_step = best_time(base, eval_tag)

    now = datetime.now(timezone.utc).astimezone()

    print("=" * 74)
    print(f"  FLY BRAIN on ESL-Hockolicious          {now:%Y-%m-%d %H:%M:%S}")
    print("=" * 74)
    print(f"  frames played            {frames:>12,}")
    print(f"  game speed               {ratio:>12.2f}x real time" if ratio else "  game speed                       --")
    print(f"  track progress (now)     {zone_now:>12,.0f} / {TOTAL_ZONES:,} zones  ({100 * zone_now / TOTAL_ZONES:5.1f}%)")
    print(f"  track progress (best)    {zone_max:>12,.0f} / {TOTAL_ZONES:,} zones  ({100 * zone_max / TOTAL_ZONES:5.1f}%)")
    if q_val is not None:
        print(f"  avg Q                    {q_val:>12.4f}")
    print()
    print(f"  best completed lap       {fmt_time(fly_best):>12}" + (f"   at {fly_best_step:,} frames" if fly_best_step else ""))
    print(f"  first completed lap      {(f'{fly_first:,} frames' if fly_first else 'not yet'):>12}")
    print()
    print(f"  --- baseline '{args.baseline}' for comparison ---")
    if base_zone is not None:
        delta = zone_now - base_zone
        sign = "+" if delta >= 0 else ""
        print(f"  baseline zones @ {frames:,} frames: {base_zone:,.0f}  ({sign}{delta:,.0f} for the fly)")
    else:
        print(f"  baseline has no data this early")
    print(f"  baseline first completed lap : {base_first:,} frames" if base_first else "  baseline first lap: n/a")
    print(f"  baseline best                : {fmt_time(base_final)}" + (f" at {base_final_step:,} frames" if base_final_step else ""))
    print()
    print(f"  targets: author {AUTHOR_TIME}s | baseline {base_final if base_final else float('nan'):.3f}s | WORLD RECORD {WORLD_RECORD}s")
    print("=" * 74)

    if not args.no_append:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        new = not args.csv.exists()
        with args.csv.open("a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(
                    ["timestamp", "unix", "frames", "game_speed_x", "zone_now", "zone_max",
                     "pct_track", "avg_q", "best_lap", "first_finish_frames", "baseline_zone_here"]
                )
            w.writerow([
                now.isoformat(timespec="seconds"), int(time.time()), frames,
                f"{ratio:.3f}" if ratio else "", f"{zone_now:.0f}", f"{zone_max:.0f}",
                f"{100 * zone_max / TOTAL_ZONES:.2f}", f"{q_val:.4f}" if q_val is not None else "",
                f"{fly_best:.3f}" if fly_best else "", fly_first or "",
                f"{base_zone:.0f}" if base_zone is not None else "",
            ])
        print(f"appended to {args.csv}")


if __name__ == "__main__":
    main()
