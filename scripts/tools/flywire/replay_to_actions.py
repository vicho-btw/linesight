"""
Turn a human replay into a sequence of agent actions.

A TMNF keyboard replay stores input CHANGES -- "SteerLeft on at 1240 ms", "SteerLeft off at
1320 ms" -- not a per-tick state. The four keys it records (Accelerate, Brake, SteerLeft,
SteerRight) map exactly onto this project's 12 discrete actions, which are every combination of
steer x accelerate x brake, so nothing is lost in WHAT is pressed.

What can be lost is WHEN. The replay resolves input changes to 10 ms; the agent only chooses an
action every ms_per_action (50 ms). Any key press shorter than one action step, or landing
mid-step, cannot be reproduced exactly. This script reports that infidelity explicitly, because
it bounds how faithfully the lap can ever be reproduced -- and a teacher trajectory that does
not actually finish the track is worthless for distillation.

    python scripts/tools/flywire/replay_to_actions.py --replay "<path>.Replay.Gbx"
    python scripts/tools/flywire/replay_to_actions.py --replay "<path>" --out data/szymix_actions.npy
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

KEYS = ("accelerate", "brake", "left", "right")
EVENT_TO_KEY = {
    "Accelerate": "accelerate",
    "Brake": "brake",
    "SteerLeft": "left",
    "SteerRight": "right",
}
TICK_MS = 10  # the resolution the replay itself stores


def load_events(replay_path: Path):
    from pygbx import Gbx, GbxType

    g = Gbx(str(replay_path))
    ghosts = g.get_classes_by_ids([GbxType.CTN_GHOST, GbxType.CTN_GHOST_OLD])
    if not ghosts:
        raise SystemExit(f"no ghost in {replay_path}")
    gh = ghosts[0]

    # Event times are NOT guaranteed to start at zero. A replay saved mid-session carries the
    # session clock: one 54.370s lap here begins at t=65535 and ends at t=119905, and 119905 -
    # 65535 is exactly its race_time. Reading those times as absolute produces a timeline made
    # entirely of the period before the race, so the car never even accelerates. The race start
    # is marked by _FakeIsRaceRunning; fall back to the earliest event if it is absent.
    start = next((e.time for e in gh.control_entries if e.event_name == "_FakeIsRaceRunning" and e.enabled), None)
    if start is None:
        start = min((e.time for e in gh.control_entries), default=0)

    # Refuse analog replays rather than silently discarding them. A gamepad/wheel run records a
    # single "Steer" axis (values across +/-65536) instead of SteerLeft/SteerRight presses. Those
    # events match nothing in EVENT_TO_KEY, so filtering leaves a near-empty timeline -- and the
    # fidelity check downstream then reports 0.0% error against that empty timeline, which looks
    # like a perfect extraction. The agent has three steering states (full left, none, full right)
    # and cannot express partial lock, so such a replay is not imitable here at all.
    analog = [e for e in gh.control_entries if e.event_name == "Steer"]
    usable = [e for e in gh.control_entries if e.event_name in EVENT_TO_KEY]
    if len(analog) > len(usable):
        raise SystemExit(
            f"{replay_path.name} uses ANALOG steering: {len(analog)} 'Steer' axis events vs "
            f"{len(usable)} digital key events. "
            "This agent's action space has only full-left / none / full-right, so partial lock "
            "cannot be reproduced. Use a keyboard replay, or extend the action space with steering "
            "magnitude (needs Python_Link.as, the wire format, config inputs, and a retrain)."
        )

    events = [e for e in gh.control_entries if e.event_name in EVENT_TO_KEY]
    events.sort(key=lambda e: e.time)
    if start:
        print(f"  (replay clock starts at {start} ms; shifting events to a zero origin)")
        for e in events:
            e.time -= start
        events = [e for e in events if e.time >= 0]
    return gh, events


def build_timeline(events, duration_ms: int):
    """Per-10ms boolean state of each key, by replaying the change events in order."""
    n = duration_ms // TICK_MS + 1
    timeline = np.zeros((n, len(KEYS)), dtype=bool)
    state = dict.fromkeys(KEYS, False)
    ei = 0
    for i in range(n):
        t = i * TICK_MS
        while ei < len(events) and events[ei].time <= t:
            state[EVENT_TO_KEY[events[ei].event_name]] = bool(events[ei].enabled)
            ei += 1
        timeline[i] = [state[k] for k in KEYS]
    return timeline


def action_index(inputs_cfg, accelerate, brake, left, right):
    for i, a in enumerate(inputs_cfg):
        if (
            bool(a["accelerate"]) == accelerate
            and bool(a["brake"]) == brake
            and bool(a["left"]) == left
            and bool(a["right"]) == right
        ):
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--out", default=None, help="where to save the action sequence (.npy)")
    ap.add_argument("--step", type=int, default=None, help="action step in ms (default: config ms_per_action)")
    args = ap.parse_args()

    from config_files import config_copy

    step = args.step or config_copy.ms_per_action
    inputs_cfg = config_copy.inputs

    gh, events = load_events(Path(args.replay))
    duration = gh.race_time
    print(f"replay        : {Path(args.replay).name}")
    print(f"race_time     : {duration / 1000:.3f}s   ({len(events)} key events)")
    print(f"action step   : {step} ms  ->  {duration // step + 1} actions")

    timeline = build_timeline(events, duration)

    # Sample the input state at each action boundary; that action is then HELD for the step.
    n_actions = duration // step + 1
    actions, held = [], np.zeros_like(timeline)
    unmapped = 0
    for k in range(n_actions):
        t_idx = min(k * step // TICK_MS, len(timeline) - 1)
        acc, brk, lft, rgt = timeline[t_idx]
        if lft and rgt:
            # Both steer keys down cancel out in TMNF; the action space has no such entry.
            lft = rgt = False
        idx = action_index(inputs_cfg, bool(acc), bool(brk), bool(lft), bool(rgt))
        if idx is None:
            unmapped += 1
            idx = 0
        actions.append(idx)
        lo, hi = k * step // TICK_MS, min((k + 1) * step // TICK_MS, len(timeline))
        held[lo:hi] = [acc, brk, lft, rgt]

    actions = np.asarray(actions, dtype=np.int64)

    # How much of the lap does a 50 ms agent actually get right?
    mism = (held[: len(timeline)] != timeline).any(axis=1)
    per_key = {k: int((held[: len(timeline), i] != timeline[:, i]).sum()) for i, k in enumerate(KEYS)}
    print(f"unmapped combos: {unmapped}")
    print()
    print(f"fidelity at {step} ms:")
    print(f"  ticks differing from the true input: {mism.sum()}/{len(timeline)} ({100 * mism.mean():.1f}%)")
    for k, v in per_key.items():
        print(f"    {k:11s}: {v:5d} ticks ({100 * v / len(timeline):.1f}%)")

    short = [
        (e1.time, e2.time - e1.time, e1.event_name)
        for e1, e2 in zip(events, events[1:])
        if e1.enabled and not e2.enabled and e1.event_name == e2.event_name and (e2.time - e1.time) < step
    ]
    print(f"  presses shorter than one action step: {len(short)}")
    for t, d, name in short[:10]:
        print(f"    {name} held {d} ms at t={t}")

    print()
    print("action histogram:")
    for idx, cnt in sorted(Counter(actions.tolist()).items()):
        a = inputs_cfg[idx]
        desc = f"{'A' if a['accelerate'] else '-'}{'B' if a['brake'] else '-'}{'L' if a['left'] else ('R' if a['right'] else '-')}"
        print(f"  {idx:2d} [{desc}]: {cnt:5d}  ({100 * cnt / len(actions):.1f}%)")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, actions)
        print(f"\nsaved {len(actions)} actions -> {out}")


if __name__ == "__main__":
    main()
