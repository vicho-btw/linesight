"""
Build a pace reference from a replay ghost, so the agent is paced against a human's lap.

The pace reward asks one question at every step: "at this point on the track, what time was the
reference at?" Normally the reference is the agent's own best lap, which means it can only ever
chase itself. A ghost from a faster driver replaces that with a target the agent has never
reached -- without requiring it to imitate the inputs, which it may not even be able to express.

The one subtlety is matching ghost positions to track zones. Naive nearest-neighbour fails:
where the track doubles back, the closest centreline point can belong to a different part of
the lap, and a racing line sitting metres off the centreline snaps to the wrong branch. On
szymix's 49.47 line that happens 207 times. So matching walks FORWARD only, exactly as
update_current_zone_idx does during a rollout, and never jumps back.

    python scripts/tools/flywire/ghost_to_pace.py --replay "<path>.Replay.Gbx" --run flybrain_rl
    python scripts/tools/flywire/ghost_to_pace.py --replay "<path>" --dry-run
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def ghost_trace(replay_path: Path):
    from pygbx import Gbx, GbxType

    g = Gbx(str(replay_path))
    ghosts = g.get_classes_by_ids([GbxType.CTN_GHOST, GbxType.CTN_GHOST_OLD])
    if not ghosts:
        raise SystemExit(f"no ghost in {replay_path}")
    gh = ghosts[0]
    pos = np.array([[r.position.x, r.position.y, r.position.z] for r in gh.records], dtype=np.float64)
    times = np.arange(len(pos), dtype=np.float64) * gh.sample_period
    # A ghost keeps recording past the finish line. Those samples are in the run-off area, far from
    # the racing line, and their timestamps exceed race_time -- so a pace table built from them ends
    # after the lap does and labels the final zones with inputs made after the race was over.
    keep = times <= gh.race_time
    if not keep.all():
        print(f"  dropping {int((~keep).sum())} sample(s) recorded after the finish ({gh.race_time / 1000:.3f}s)")
    return gh, pos[keep], times[keep]


def match_forward(pos, zone_centers, start_zone, window=400):
    """Zone index per sample, searching forward only -- never back onto an earlier branch."""
    idx = np.empty(len(pos), dtype=np.int64)
    cur = start_zone
    for i, p in enumerate(pos):
        hi = min(cur + window, len(zone_centers))
        seg = zone_centers[cur:hi]
        d = np.linalg.norm(seg - p, axis=1)
        cur = cur + int(np.argmin(d))
        idx[i] = cur
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--run", default="flybrain_rl", help="run whose reference_pace.npy to write")
    ap.add_argument("--dry-run", action="store_true", help="report without writing")
    ap.add_argument("--zone-centers", default=None,
                    help="centreline file to match against. A ghost that takes a shortcut leaves the normal "
                         "route entirely, so matching it to the normal centreline yields nonsense exactly "
                         "where it matters; use a centreline built from that ghost's own line.")
    args = ap.parse_args()

    from config_files import config_copy
    from trackmania_rl.map_loader import load_next_map_zone_centers

    # Load zone centres exactly as training does, extrapolation included, so the indices in the
    # table mean the same thing as rollout_results["current_zone_idx"].
    from itertools import chain, cycle

    if args.zone_centers:
        zcf = Path(args.zone_centers).name
    else:
        _, _, zcf, _, _ = next(cycle(chain(*config_copy.map_cycle)))
    zone_centers = load_next_map_zone_centers(zcf, ROOT)
    start_zone = config_copy.n_zone_centers_extrapolate_before_start_of_map
    print(f"zone centres : {len(zone_centers)} (start index {start_zone})")

    gh, pos, times = ghost_trace(Path(args.replay))
    print(f"ghost        : {len(pos)} samples every {gh.sample_period}ms, race_time {gh.race_time / 1000:.3f}s")

    zones = match_forward(pos, zone_centers, start_zone)
    back = int((np.diff(zones) < 0).sum())
    print(f"matched zones: {zones.min()}..{zones.max()}   backward steps: {back} (must be 0)")
    resid = np.linalg.norm(zone_centers[zones] - pos, axis=1)
    print(f"match residual: median {np.median(resid):.2f}m  p90 {np.percentile(resid, 90):.2f}m  max {resid.max():.2f}m")

    # Interpolate a time for every zone between the first and last the ghost reached.
    lo, hi = int(zones[0]), int(zones[-1])
    table = np.full(hi + 1, np.inf, dtype=np.float64)
    # first arrival wins, so a zone the car passes twice keeps the earlier time
    for z, t in zip(zones, times):
        if t < table[z]:
            table[z] = t
    known = np.flatnonzero(np.isfinite(table))
    table[lo : hi + 1] = np.interp(np.arange(lo, hi + 1), known, table[known])
    table[:lo] = 0.0

    if not np.all(np.diff(table) >= -1e-9):
        bad = int((np.diff(table) < 0).sum())
        print(f"WARNING: table is not monotonic in time ({bad} decreases) -- pace deltas would be wrong")

    print(f"pace table   : {len(table)} zones, finishes at {table[-1] / 1000:.3f}s")
    for frac in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        z = int(lo + frac * (hi - lo))
        print(f"   {frac * 100:5.0f}% of track (zone {z:5d}): {table[z] / 1000:6.2f}s")

    if args.dry_run:
        print("\ndry run: nothing written")
        return

    out = ROOT / "save" / args.run / "reference_pace.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        prev = np.load(out)
        backup = out.with_name("reference_pace.previous.npy")
        np.save(backup, prev)
        print(f"\nexisting reference ({prev[-1] / 1000:.3f}s over {len(prev)} zones) backed up -> {backup.name}")
    tmp = out.with_name("reference_pace.tmp.npy")  # np.save appends .npy unless it is already there
    np.save(tmp, table)
    tmp.replace(out)
    print(f"wrote {out}  ({table[-1] / 1000:.3f}s target)")


if __name__ == "__main__":
    main()
