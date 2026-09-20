"""
Collect corrective data: let the policy drive, and label what it sees with what the human did there.

Cloning one lap produces a policy that scores 97% on held-out frames and drives 6% of the track.
The reason is not capacity: one demonstration contains zero examples of being OFF the line, so
the first small deviation puts the car in states the dataset never covered. The fix is DAgger
(Ross et al., 2011) -- train on the states the policy actually visits, labelled by the expert.

Our "expert" is a recorded lap, not a queryable policy, so the label is recovered by position:

    zone on szymix's centreline  ->  pace table gives his race time at that zone
                                 ->  his action timeline gives the key he held then

That is an approximation worth stating: it answers "what was he doing HERE", not "what should be
done from this exact state". Off-line with the wrong angle or speed, his input at that position
may not be a good recovery. It should pull the policy back toward the line; it is not an optimal
controller.

Rollouts start from states banked along the scripted world-record lap rather than from the start
line. Otherwise a policy that fails at 6% would only ever generate data about the first 6%, and
coverage of the rest of the track would have to wait for it to improve.

    python scripts/tools/flywire/dagger_collect.py --policy flybrain_szymix_bc --laps 20
"""

import argparse
import sys
from itertools import chain, cycle
from multiprocessing import Lock
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="flybrain_szymix_bc", help="run whose weights drive the rollouts")
    ap.add_argument("--actions", default=str(ROOT / "data" / "szymix_actions_10ms.npy"))
    ap.add_argument("--pace", default=str(ROOT / "data" / "szymix_pace.npy"))
    ap.add_argument("--map", default=None, help="map path to request, overriding the config map cycle")
    ap.add_argument("--zone-centers", default="ESL-Hockolicious_szymix_0.5m.npy")
    ap.add_argument("--out", default=str(ROOT / "data" / "szymix_dagger"))
    ap.add_argument("--laps", type=int, default=20)
    ap.add_argument("--buckets", type=int, default=40, help="how many places along the lap to restart from")
    ap.add_argument("--epsilon", type=float, default=0.02, help="small noise so repeated starts diverge differently")
    args = ap.parse_args()

    from config_files import config_copy

    config_copy.use_fly_brain = True
    config_copy.ms_per_action = 10
    config_copy.tm_engine_step_per_action = 1
    config_copy.exploring_starts_n_buckets = args.buckets
    config_copy.exploring_starts_prob = 1.0

    from trackmania_rl.agents import iqn
    from trackmania_rl.map_loader import load_next_map_zone_centers
    from trackmania_rl.tmi_interaction import game_instance_manager

    expert_actions = np.load(args.actions)
    pace = np.load(args.pace)
    print(f"expert: {len(expert_actions)} actions, pace table {len(pace)} zones ending {pace[-1] / 1000:.3f}s")

    def label_for_zone(z):
        """What the human was pressing at this point of the track."""
        z = int(min(max(z, 0), len(pace) - 1))
        t_ms = pace[z]
        k = int(round(t_ms / config_copy.ms_per_action))
        return int(expert_actions[min(max(k, 0), len(expert_actions) - 1)])

    net, _ = iqn.make_untrained_iqn_network(jit=False, is_inference=True)
    net.load_state_dict(torch.load(ROOT / "save" / args.policy / "weights1.torch", weights_only=False))
    net.eval()
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

    _, map_path, _, _, _ = next(cycle(chain(*config_copy.map_cycle)))
    if args.map:
        map_path = args.map
    print(f"map: {map_path}")
    zone_centers = load_next_map_zone_centers(Path(args.zone_centers).name, ROOT)
    print(f"progress measured on {args.zone_centers} ({len(zone_centers)} zones)")

    # --- first, bank states along the scripted world-record lap ---------------------------
    # rollout() banks a simulation state the first time it enters each section, but only on laps
    # that started at the line. Driving the expert's own inputs fills the bank with states ON his
    # racing line, which is exactly where we want the policy to be restarted from.
    n_act = len(config_copy.inputs)
    ctr = {"i": 0}

    def scripted(img, floats):
        i = ctr["i"] + 1  # +1: TMInterface latches a requested input on the NEXT engine tick
        ctr["i"] += 1
        a = int(expert_actions[i]) if i < len(expert_actions) else 3
        q = np.zeros(n_act, dtype=np.float32)
        q[a] = 1.0
        return a, True, 1.0, q

    print("\nbanking states along the expert lap...")
    ctr["i"] = 0
    rr, es = tmi.rollout(
        exploration_policy=scripted, map_path=map_path, zone_centers=zone_centers,
        update_network=lambda: None, exploring_starts=False,
    )
    banked = tmi.zone_states.get(map_path, {})
    print(f"  expert lap finished={es.get('race_finished')} time={es.get('race_time', 0) / 1000:.3f}s")
    print(f"  banked {len(banked)} start states: sections {sorted(banked)}")
    if not banked:
        raise SystemExit("nothing banked; cannot restart mid-lap")

    # --- now drive the policy from those states and label what it sees --------------------
    frames_all, floats_all, labels_all = [], [], []
    agree_total = seen_total = 0

    for lap in range(1, args.laps + 1):
        try:
            rr, es = tmi.rollout(
                exploration_policy=inferer.get_exploration_action, map_path=map_path,
                zone_centers=zone_centers, update_network=lambda: None, exploring_starts=True,
            )
        except Exception as e:
            print(f"  lap {lap}: rollout raised {type(e).__name__}: {e}")
            try:
                tmi.close_game()
            except Exception:
                pass
            tmi.iface = None
            tmi.last_rollout_crashed = True
            continue

        sf = rr["state_float"]
        zones = rr["current_zone_idx"]
        acts = rr["actions"]
        frames = rr["frames"]
        n = min(len(sf), len(zones), len(acts), len(frames))
        start_zone = rr.get("exploring_start_zone", 0)

        kept = 0
        agree = 0
        for i in range(n):
            f = frames[i]
            z = zones[i]
            if not isinstance(f, np.ndarray) or not isinstance(z, (int, np.integer)):
                continue
            lbl = label_for_zone(z)
            frames_all.append(np.asarray(f, dtype=np.uint8))
            floats_all.append(np.asarray(sf[i], dtype=np.float32))
            labels_all.append(lbl)
            if isinstance(acts[i], (int, np.integer)) and int(acts[i]) == lbl:
                agree += 1
            kept += 1
        agree_total += agree
        seen_total += kept
        furthest = max((z for z in zones if isinstance(z, (int, np.integer))), default=0)
        print(
            f"  lap {lap:3d}: start zone {start_zone:5d} -> furthest {furthest:5d}"
            f"  kept {kept:5d} states, policy agreed with expert {100 * agree / max(kept, 1):5.1f}%"
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames_arr = np.stack(frames_all).astype(np.uint8)
    floats_arr = np.stack(floats_all).astype(np.float32)
    labels_arr = np.asarray(labels_all, dtype=np.int64)
    np.savez_compressed(out / "dagger_000.npz", frames=frames_arr, floats=floats_arr, actions=labels_arr)
    print(f"\ncollected {len(labels_arr):,} corrective states -> {out}")
    print(f"policy agreed with the expert on {100 * agree_total / max(seen_total, 1):.1f}% of them")
    print("Low agreement is the point: those are the states the single-lap dataset never contained.")

    try:
        tmi.close_game()
    except Exception:
        pass


if __name__ == "__main__":
    main()
