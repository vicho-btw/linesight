"""
Drive a recorded action sequence in the game, and optionally capture it for behaviour cloning.

This answers the question that decides whether a human replay can be used as a teacher at all:
the agent can only change inputs every ms_per_action (50 ms), while the replay resolves them to
10 ms. Trackmania physics over a 50 second lap is chaotic, so a few percent of mistimed inputs
can either be absorbed or compound into a completely different line. Reasoning cannot settle
that -- only driving it can.

If the lap survives, the captured (frame, float, action) triples are a behaviour-cloning dataset
on a racing line no policy here has ever driven. Note there are no Q-values: a human teacher
gives action labels only, so the training objective is cross-entropy, not the KL-on-Q that
distill_flybrain.py uses for a network teacher.

    python scripts/tools/flywire/drive_replay.py --actions data/szymix_actions.npy --laps 3
    python scripts/tools/flywire/drive_replay.py --actions data/szymix_actions.npy --capture data/szymix_bc
"""

import argparse
import sys
from itertools import chain, cycle
from multiprocessing import Lock
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# Reference splits are read from the replay being driven, not hardcoded: the same script is used
# on several maps, and a constant from one of them silently mislabels every other comparison.
WR_SPLITS = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", default=str(ROOT / "data" / "szymix_actions.npy"))
    ap.add_argument("--laps", type=int, default=3)
    ap.add_argument("--capture", default=None, help="directory to write the BC dataset into")
    ap.add_argument("--action-ms", type=int, default=None, help="drive at this action step instead of the configured one")
    ap.add_argument("--reference-replay", default=None,
                    help="replay whose checkpoint times to compare against (defaults to no comparison)")
    ap.add_argument("--map", default=None, help="map path to request, overriding the config map cycle")
    ap.add_argument("--zone-centers", default=None, help="centreline file for progress tracking")
    ap.add_argument("--shift", type=int, default=0,
                    help="apply action[i+shift] at step i. If TMInterface latches a requested input on the "
                         "NEXT engine tick, every input we send is one tick late, which tracks well at first "
                         "and then fails at the first precision-critical moment. shift=1 tests that.")
    ap.add_argument("--no-cutoff", action="store_true",
                    help="disable the no-progress timeout: a shortcut leaves the centreline, which freezes "
                         "progress tracking and kills the rollout even though the car is driving fine")
    args = ap.parse_args()

    from config_files import config_copy

    config_copy.use_fly_brain = True
    config_copy.exploring_starts_prob = 0.0
    if args.action_ms:
        # The engine ticks every 10 ms, so an action every N ms means N//10 engine steps.
        assert args.action_ms % 10 == 0, "action step must be a multiple of the 10 ms engine tick"
        config_copy.ms_per_action = args.action_ms
        config_copy.tm_engine_step_per_action = args.action_ms // 10
        print(f"driving at {args.action_ms} ms per action ({config_copy.tm_engine_step_per_action} engine steps)")

    from trackmania_rl.agents import iqn
    from trackmania_rl.map_loader import load_next_map_zone_centers
    from trackmania_rl.tmi_interaction import game_instance_manager

    actions = np.load(args.actions)
    n_actions = len(actions)
    global WR_SPLITS
    if args.reference_replay:
        from pygbx import Gbx, GbxType

        _g = Gbx(args.reference_replay)
        _gh = _g.get_classes_by_ids([GbxType.CTN_GHOST, GbxType.CTN_GHOST_OLD])[0]
        WR_SPLITS = [t / 1000 for t in (_gh.cp_times or [])]
        print(f"reference splits from {Path(args.reference_replay).name}: finish {WR_SPLITS[-1]:.3f}s")
    print(f"scripted actions: {n_actions}  ({n_actions * config_copy.ms_per_action / 1000:.2f}s of driving)")

    # rollout() needs a network to exist for frame capture plumbing, but it is never consulted:
    # the policy below ignores its inputs entirely and returns the scripted action.
    net, _ = iqn.make_untrained_iqn_network(jit=False, is_inference=True)
    net.eval()
    n_act = len(config_copy.inputs)

    step_counter = {"i": 0}

    def scripted_policy(img, floats):
        i = step_counter["i"] + args.shift
        step_counter["i"] += 1
        if 0 <= i < n_actions:
            a = int(actions[i])
        else:
            # Past the end of the recording the car is somewhere the human never was. Lift off
            # rather than hold the last input, which would usually mean full throttle into a wall.
            a = 3  # no accelerate, no brake, no steer
        q = np.zeros(n_act, dtype=np.float32)
        q[a] = 1.0
        return a, True, 1.0, q

    tmi = game_instance_manager.GameInstanceManager(
        game_spawning_lock=Lock(),
        running_speed=config_copy.running_speed,
        run_steps_per_action=config_copy.tm_engine_step_per_action,
        max_overall_duration_ms=config_copy.cutoff_rollout_if_race_not_finished_within_duration_ms,
        max_minirace_duration_ms=(600_000 if args.no_cutoff else config_copy.cutoff_rollout_if_no_vcp_passed_within_duration_ms),
        tmi_port=config_copy.base_tmi_port,
    )

    map_cycle_iter = cycle(chain(*config_copy.map_cycle))
    _, map_path, zone_centers_filename, _, _ = next(map_cycle_iter)
    if args.map:
        map_path = args.map
    if args.zone_centers:
        zone_centers_filename = Path(args.zone_centers).name
    print(f"map: {map_path}")
    print(f"centreline: {zone_centers_filename}")
    zone_centers = load_next_map_zone_centers(zone_centers_filename, ROOT)
    total_zones = len(zone_centers)

    captured = []
    for lap in range(1, args.laps + 1):
        step_counter["i"] = 0
        try:
            rr, es = tmi.rollout(
                exploration_policy=scripted_policy,
                map_path=map_path,
                zone_centers=zone_centers,
                update_network=lambda: None,
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

        zones = [z for z in rr["current_zone_idx"] if isinstance(z, (int, np.integer))]
        furthest = max(zones) if zones else 0
        n_used = min(step_counter["i"], n_actions)

        if es.get("race_finished"):
            t = es["race_time"] / 1000
            ref = f"   (reference {WR_SPLITS[-1]:.3f}s, delta {t - WR_SPLITS[-1]:+.3f}s)" if WR_SPLITS else ""
            print(f"  lap {lap}: FINISHED {t:.3f}s{ref}")
            cps = [c / 1000 for c in es.get("cp_time_ms", [])]
            if cps and WR_SPLITS:
                print("    sector-by-sector vs szymix:")
                for i, (ours, theirs) in enumerate(zip(cps, WR_SPLITS)):
                    print(f"      cp{i:2d}: ours {ours:6.2f}   szymix {theirs:6.2f}   {ours - theirs:+.2f}")
            captured.append(rr)
        else:
            pct = 100 * furthest / max(total_zones, 1)
            print(
                f"  lap {lap}: did not finish - reached zone {furthest}/{total_zones} ({pct:.1f}% of track) "
                f"after {n_used}/{n_actions} scripted actions"
            )
            # Where it diverged is the useful part: it bounds what 50 ms control can reproduce.
            captured.append(rr)

    if args.capture and captured:
        out = Path(args.capture)
        out.mkdir(parents=True, exist_ok=True)
        best = max(captured, key=lambda r: max((z for z in r["current_zone_idx"] if isinstance(z, (int, np.integer))), default=0))
        frames = np.asarray(best["frames"][: len(best["state_float"])], dtype=object)
        floats = np.asarray(best["state_float"], dtype=np.float32)
        acts = np.asarray(best["actions"][: len(best["state_float"])], dtype=np.int64)
        keep = np.array([isinstance(f, np.ndarray) for f in frames])
        np.savez_compressed(
            # named after the output directory: a hardcoded name silently writes one map's data
            # under another map's label, which is only noticed when something else fails
            out / f"{out.name}_000.npz",
            # np.asarray(..., dtype=object) above is needed only to hold the np.nan the finish path
            # appends beside real frames. The stacked result must be cast back to uint8, or the npz
            # is a pickled object array that will not load without allow_pickle.
            frames=np.stack([np.asarray(f, dtype=np.uint8) for f, k in zip(frames, keep) if k]).astype(np.uint8),
            floats=floats[keep],
            actions=acts[keep],
        )
        print(f"\ncaptured {int(keep.sum())} (frame, float, action) triples -> {out}")

    try:
        tmi.close_game()
    except Exception:
        pass


if __name__ == "__main__":
    main()
