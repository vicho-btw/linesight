"""
Drive a saved agent in the game and report lap times.

Per-frame validation accuracy cannot tell you whether a policy can actually complete a lap:
a lap is ~1100 sequential decisions, and one wrong action puts the car in a state slightly
off the training distribution, where the next error is more likely. Compounding error is the
classic failure of behaviour cloning and it is invisible to held-out metrics. So we drive.

    python scripts/tools/flywire/eval_agent.py --run flybrain_distilled --fly --laps 5
    python scripts/tools/flywire/eval_agent.py --run hocko_run1 --laps 3        # the teacher
"""

import argparse
import sys
import time
from itertools import chain, cycle
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

WORLD_RECORD = 53.76
TEACHER_BEST = 54.370


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name under save/")
    ap.add_argument("--fly", action="store_true", help="the checkpoint is a connectome network")
    ap.add_argument("--laps", type=int, default=5)
    ap.add_argument("--epsilon", type=float, default=0.0, help="0 = pure greedy")
    ap.add_argument("--action-ms", type=int, default=None,
                    help="drive at this action step. A policy trained at 10 ms behaves differently at 50 ms, "
                         "so this must match the rate it was trained for or the result is meaningless.")
    ap.add_argument("--deterministic", action="store_true",
                    help="use FIXED quantiles instead of random ones. IQN samples tau randomly per decision, "
                         "so a greedy policy still picks different actions on identical states and one flipped "
                         "action early sends the lap somewhere else. This makes a run reproducible.")
    ap.add_argument("--map", default=None,
                    help="map path to request, overriding the config map cycle. Must match the map the "
                         "policy was trained on, or the result says nothing.")
    ap.add_argument("--zone-centers", default=None,
                    help="centreline to measure progress against. A policy that drives a shortcut must be "
                         "scored on a centreline covering THAT route; on the normal one its progress freezes "
                         "and every lap looks like a failure. Preferred over --no-cutoff, which merely "
                         "removes the safety that ends a failed rollout quickly.")
    ap.add_argument("--no-cutoff", action="store_true",
                    help="disable the no-progress timeout. A policy that takes a shortcut leaves the "
                         "centreline, which freezes progress tracking and kills the rollout even though "
                         "the car is driving fine.")
    args = ap.parse_args()

    from multiprocessing import Lock

    from config_files import config_copy

    config_copy.use_fly_brain = bool(args.fly)
    if args.action_ms:
        assert args.action_ms % 10 == 0, "action step must be a multiple of the 10 ms engine tick"
        config_copy.ms_per_action = args.action_ms
        config_copy.tm_engine_step_per_action = args.action_ms // 10
        print(f"driving at {args.action_ms} ms per action ({config_copy.tm_engine_step_per_action} engine steps)")

    from trackmania_rl.agents import iqn as iqn
    from trackmania_rl.map_loader import load_next_map_zone_centers
    from trackmania_rl.tmi_interaction import game_instance_manager

    weights = ROOT / "save" / args.run / "weights1.torch"
    if not weights.exists():
        raise SystemExit(f"no weights at {weights}")

    net, _ = iqn.make_untrained_iqn_network(jit=False, is_inference=True)
    net.load_state_dict(torch.load(f=weights, weights_only=False))
    net.eval()
    kind = "CONNECTOME" if args.fly else "dense trunk"
    print(f"{kind} agent '{args.run}' loaded, epsilon={args.epsilon}")

    inferer = iqn.Inferer(net, config_copy.iqn_k, config_copy.tau_epsilon_boltzmann)

    policy = inferer.get_exploration_action
    if args.deterministic:
        import torch as _t

        k = config_copy.iqn_k
        # midpoints of k equal segments of [0,1]: the same quantiles every time, so the Q estimate
        # for a given state is a fixed number rather than a draw.
        fixed_tau = ((_t.arange(k, dtype=_t.float32) + 0.5) / k).reshape(k, 1).to("cuda")

        def policy(img, floats):
            q = inferer.infer_network(img, floats, tau=fixed_tau).mean(axis=0)
            a = int(np.argmax(q))
            return a, True, float(q[a]), q

        print(f"deterministic inference: {k} fixed quantiles")
    inferer.epsilon = args.epsilon
    inferer.epsilon_boltzmann = 0.0
    inferer.is_explo = args.epsilon > 0
    inferer.tau_epsilon_boltzmann = config_copy.tau_epsilon_boltzmann

    tmi = game_instance_manager.GameInstanceManager(
        game_spawning_lock=Lock(),
        running_speed=config_copy.running_speed,
        run_steps_per_action=config_copy.tm_engine_step_per_action,
        max_overall_duration_ms=config_copy.cutoff_rollout_if_race_not_finished_within_duration_ms,
        max_minirace_duration_ms=(600_000 if args.no_cutoff else config_copy.cutoff_rollout_if_no_vcp_passed_within_duration_ms),
        tmi_port=config_copy.base_tmi_port,
    )

    map_cycle_iter = cycle(chain(*config_copy.map_cycle))
    map_name, map_path, zone_centers_filename, _, _ = next(map_cycle_iter)
    if args.map:
        map_path = args.map
        print(f"map: {map_path}")
    if args.zone_centers:
        zone_centers_filename = Path(args.zone_centers).name
        print(f"measuring progress against {zone_centers_filename}")
    zone_centers = load_next_map_zone_centers(zone_centers_filename, ROOT)
    # load_next_map_zone_centers pads the centreline with extrapolated zones before the start and
    # AFTER the finish (1000 of them by default). Measuring progress against the padded length
    # understates it badly -- three quarters of a lap reads as 60% -- so report against the real
    # track and keep the padded count only for the raw zone index.
    total_zones = len(zone_centers)
    pre = config_copy.n_zone_centers_extrapolate_before_start_of_map
    real_zones = total_zones - pre - config_copy.n_zone_centers_extrapolate_after_end_of_map
    pct = lambda z: 100.0 * max(0, z - pre) / max(real_zones, 1)

    times, furthest = [], []
    for lap in range(1, args.laps + 1):
        try:
            rr, end_stats = tmi.rollout(
                exploration_policy=policy,
                map_path=map_path,
                zone_centers=zone_centers,
                update_network=lambda: None,
            )
        except Exception as e:
            print(f"  lap {lap}: rollout failed ({type(e).__name__}); restarting game")
            try:
                tmi.close_game()
            except Exception:
                pass
            tmi.iface = None
            time.sleep(5)
            continue

        rt = end_stats.get("race_time", None)
        zone = max([z for z in rr["current_zone_idx"] if isinstance(z, (int, np.integer))], default=0)
        furthest.append(zone)
        if isinstance(rt, (int, float)) and rt < 300_000:
            secs = rt / 1000
            times.append(secs)
            print(f"  lap {lap}: FINISHED {secs:8.3f}s")
        else:
            print(f"  lap {lap}: did not finish   reached zone {zone}/{pre + real_zones} ({pct(zone):.1f}% of track)")

    print()
    if times:
        b = min(times)
        print(f"  finished {len(times)}/{args.laps}   best {b:.3f}s   median {np.median(times):.3f}s")
        print(f"  teacher best {TEACHER_BEST}s   world record {WORLD_RECORD}s   gap to WR {b - WORLD_RECORD:+.3f}s")
    else:
        print(f"  finished 0/{args.laps}")
        if furthest:
            print(f"  furthest progress: zone {max(furthest)}/{pre + real_zones} ({pct(max(furthest)):.1f}% of track)")
    try:
        tmi.close_game()
    except Exception:
        pass


if __name__ == "__main__":
    main()
