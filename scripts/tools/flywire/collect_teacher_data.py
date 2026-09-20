"""
Collect a supervised dataset from a trained dense-trunk agent.

Runs the teacher policy in the game and saves, for every action it takes, the observation it
saw and the Q values it assigned to all twelve actions. The connectome is then distilled
against those Q values (see distill_flybrain.py) before being handed back to RL.

    python scripts/tools/flywire/collect_teacher_data.py --teacher hocko_run1 --laps 80

Nothing in the game code needs changing: rollout_results already carries frames,
state_float and q_values; this only persists them.

On epsilon
----------
The labels are always the teacher's greedy opinion -- its full Q vector, which is what
teaches the racing line. Epsilon affects only which *states* get visited. TrackMania is
deterministic, so a purely greedy teacher drives an identical lap every time and yields
about a thousand distinct frames, repeated: too few to fit a 7M-parameter trunk, and with no
coverage of what to do when the car is a metre off the racing line. A small epsilon scatters
the visited states around that line while leaving every label untouched. Pass --epsilon 0
for strictly on-policy data.
"""

import argparse
import sys
import time
from datetime import datetime
from itertools import chain, cycle
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def usable(row) -> bool:
    """Rollout lists are padded with np.nan where the race ended mid-step."""
    return isinstance(row, np.ndarray)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="hocko_run1", help="run name whose weights1.torch is the teacher")
    ap.add_argument("--laps", type=int, default=80, help="number of rollouts to collect")
    ap.add_argument("--epsilon", type=float, default=0.03, help="exploration during collection; labels stay greedy")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "distill")
    args = ap.parse_args()

    from multiprocessing import Lock

    from config_files import config_copy

    # The teacher is the dense-trunk network, whatever the current config says.
    config_copy.use_fly_brain = False

    from trackmania_rl.agents import iqn as iqn
    from trackmania_rl.map_loader import load_next_map_zone_centers
    from trackmania_rl.tmi_interaction import game_instance_manager

    weights = ROOT / "save" / args.teacher / "weights1.torch"
    if not weights.exists():
        raise SystemExit(f"no teacher weights at {weights}")

    args.out.mkdir(parents=True, exist_ok=True)

    net, _ = iqn.make_untrained_iqn_network(jit=False, is_inference=True)
    net.load_state_dict(torch.load(f=weights, weights_only=False))
    net.eval()
    print(f"teacher loaded from {weights}")

    inferer = iqn.Inferer(net, config_copy.iqn_k, config_copy.tau_epsilon_boltzmann)
    inferer.epsilon = args.epsilon
    inferer.epsilon_boltzmann = 0.0
    inferer.is_explo = args.epsilon > 0
    inferer.tau_epsilon_boltzmann = config_copy.tau_epsilon_boltzmann

    tmi = game_instance_manager.GameInstanceManager(
        game_spawning_lock=Lock(),
        running_speed=config_copy.running_speed,
        run_steps_per_action=config_copy.tm_engine_step_per_action,
        max_overall_duration_ms=config_copy.cutoff_rollout_if_race_not_finished_within_duration_ms,
        max_minirace_duration_ms=config_copy.cutoff_rollout_if_no_vcp_passed_within_duration_ms,
        tmi_port=config_copy.base_tmi_port,
    )

    map_cycle_iter = cycle(chain(*config_copy.map_cycle))
    map_name, map_path, zone_centers_filename, _, _ = next(map_cycle_iter)
    zone_centers = load_next_map_zone_centers(zone_centers_filename, ROOT)
    print(f"map: {map_name}  epsilon: {args.epsilon}  target laps: {args.laps}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_frames = 0
    kept_laps = 0
    t0 = time.perf_counter()

    for lap in range(1, args.laps + 1):
        try:
            rr, end_stats = tmi.rollout(
                exploration_policy=inferer.get_exploration_action,
                map_path=map_path,
                zone_centers=zone_centers,
                update_network=lambda: None,
            )
        except Exception as e:
            print(f"  lap {lap}: rollout failed ({type(e).__name__}), restarting the game")
            try:
                tmi.close_game()
            except Exception:
                pass
            tmi.iface = None
            tmi.last_rollout_crashed = True
            time.sleep(5)
            continue

        if tmi.last_rollout_crashed:
            print(f"  lap {lap}: crashed, discarded")
            continue

        n = min(len(rr["frames"]), len(rr["state_float"]), len(rr["q_values"]), len(rr["actions"]))
        keep = [i for i in range(n) if usable(rr["frames"][i]) and usable(rr["state_float"][i]) and usable(rr["q_values"][i])]
        if not keep:
            print(f"  lap {lap}: nothing usable")
            continue

        frames = np.stack([rr["frames"][i] for i in keep]).astype(np.uint8)          # (T, 1, H, W)
        floats = np.stack([rr["state_float"][i] for i in keep]).astype(np.float32)   # (T, float_dim)
        qvals = np.stack([rr["q_values"][i] for i in keep]).astype(np.float32)       # (T, 12)
        actions = np.array([rr["actions"][i] for i in keep], dtype=np.int16)
        greedy = np.array([bool(rr["action_was_greedy"][i]) for i in keep], dtype=np.bool_)

        shard = args.out / f"{args.teacher}_{stamp}_{lap:04d}.npz"
        np.savez_compressed(shard, frames=frames, floats=floats, qvals=qvals, actions=actions, greedy=greedy)

        total_frames += len(keep)
        kept_laps += 1
        race_time = end_stats.get("race_time", None)
        rt = f"{race_time / 1000:.3f}s" if isinstance(race_time, (int, float)) else "DNF"
        rate = total_frames / max(time.perf_counter() - t0, 1e-9)
        print(
            f"  lap {lap:3d}/{args.laps}: {len(keep):5d} frames  race {rt:>8}  "
            f"total {total_frames:,}  ({rate:.0f} frames/s)  -> {shard.name}"
        )

    dt = time.perf_counter() - t0
    size = sum(f.stat().st_size for f in args.out.glob("*.npz")) / 2**20
    print(f"\ncollected {total_frames:,} frames over {kept_laps} laps in {dt / 60:.1f} min")
    print(f"dataset: {args.out}  ({size:.0f} MiB)")
    try:
        tmi.close_game()
    except Exception:
        pass


if __name__ == "__main__":
    main()
