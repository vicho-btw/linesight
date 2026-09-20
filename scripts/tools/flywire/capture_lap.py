"""
Drive a policy and record both what it sees and what its connectome is doing, for video.

The FlyBrain already publishes settled activity to a live tap for the browser viewer, gated on a
watch file and throttled to every Nth inference. For a video we want every step and no gating, so
this substitutes a recorder object with the same publish() interface rather than changing
flybrain.py -- the brain calls whatever is in self.live_tap.

Two things are recorded per action:
  * the 120x160 grayscale frame the agent actually receives, which is the whole of its vision
  * the settled activity of all 139,248 neurons, quantised to uint8

    python scripts/tools/flywire/capture_lap.py --run checkpoints/flybrain_20260920_012559 --out data/lap_capture
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


class Recorder:
    """Duck-typed stand-in for _LiveTap: keeps every frame instead of publishing a sampled few."""

    def __init__(self):
        self.acts = []
        self.qs = []
        self.ok = True

    def publish(self, x_col):
        # x_col is (n_neurons,) on the GPU. Quantise here, on device, then one transfer.
        a = x_col.abs()
        scale = torch.clamp(a.max(), min=1e-6)
        self.acts.append((x_col / scale * 127.0 + 128.0).clamp(0, 255).to(torch.uint8).cpu().numpy())

    def publish_q(self, q_values):
        try:
            self.qs.append(np.asarray(q_values, dtype=np.float32).copy())
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="checkpoints/flybrain_20260920_012559")
    ap.add_argument("--map", default='"ESL-Hockolicious.Challenge.Gbx"')
    ap.add_argument("--zone-centers", default="ESL-Hockolicious_0.5m_cl2.npy")
    ap.add_argument("--action-ms", type=int, default=None, help="default: whatever the config says")
    ap.add_argument("--out", default=str(ROOT / "data" / "lap_capture"))
    ap.add_argument("--laps", type=int, default=4, help="drive this many, keep the fastest")
    args = ap.parse_args()

    from config_files import config_copy

    config_copy.use_fly_brain = True
    config_copy.exploring_starts_prob = 0.0
    config_copy.fly_live_tap = False  # we attach our own recorder instead
    if args.action_ms:
        config_copy.ms_per_action = args.action_ms
        config_copy.tm_engine_step_per_action = args.action_ms // 10
    print(f"driving at {config_copy.ms_per_action} ms per action")

    from trackmania_rl.agents import iqn
    from trackmania_rl.map_loader import load_next_map_zone_centers
    from trackmania_rl.tmi_interaction import game_instance_manager

    net, uncompiled = iqn.make_untrained_iqn_network(jit=False, is_inference=True)
    net.load_state_dict(torch.load(ROOT / "save" / args.run / "weights1.torch", weights_only=False))
    net.eval()

    # make_untrained_iqn_network returns two DISTINCT module trees, each with its own FlyBrain.
    # Inference runs through `net`; attaching the tap only to `uncompiled` records nothing at all
    # (collector_process.py sets it on both for exactly this reason).
    rec = Recorder()
    attached = 0
    for holder in (net, uncompiled):
        b = getattr(holder, "fly_brain", None)
        if b is not None:
            b.live_tap = rec
            attached += 1
            n_neurons = b.n_neurons
    if not attached:
        raise SystemExit("no fly_brain on this network")
    print(f"recorder attached to {attached} connectome instance(s), {n_neurons:,} neurons")

    inferer = iqn.Inferer(net, config_copy.iqn_k, config_copy.tau_epsilon_boltzmann)
    inferer.epsilon = 0.0
    inferer.epsilon_boltzmann = 0.0
    inferer.is_explo = False
    inferer.tau_epsilon_boltzmann = config_copy.tau_epsilon_boltzmann

    def policy(img, floats):
        a, greedy, v, q = inferer.get_exploration_action(img, floats)
        rec.publish_q(q)
        return a, greedy, v, q

    tmi = game_instance_manager.GameInstanceManager(
        game_spawning_lock=Lock(),
        running_speed=config_copy.running_speed,
        run_steps_per_action=config_copy.tm_engine_step_per_action,
        max_overall_duration_ms=config_copy.cutoff_rollout_if_race_not_finished_within_duration_ms,
        max_minirace_duration_ms=config_copy.cutoff_rollout_if_no_vcp_passed_within_duration_ms,
        tmi_port=config_copy.base_tmi_port,
    )
    _, map_path, zcf, _, _ = next(cycle(chain(*config_copy.map_cycle)))
    map_path = args.map
    zone_centers = load_next_map_zone_centers(Path(args.zone_centers).name, ROOT)
    print(f"map {map_path}  centreline {args.zone_centers}")

    best = None
    for lap in range(1, args.laps + 1):
        rec.acts.clear()
        rec.qs.clear()
        try:
            rr, es = tmi.rollout(
                exploration_policy=policy, map_path=map_path,
                zone_centers=zone_centers, update_network=lambda: None,
            )
        except Exception as e:
            print(f"  lap {lap}: {type(e).__name__}: {e}")
            try:
                tmi.close_game()
            except Exception:
                pass
            tmi.iface = None
            tmi.last_rollout_crashed = True
            continue

        fin = es.get("race_finished")
        t = es["race_time"] / 1000 if fin else None
        zones = [z for z in rr["current_zone_idx"] if isinstance(z, (int, np.integer))]
        print(f"  lap {lap}: {'FINISHED ' + format(t, '.3f') + 's' if fin else 'did not finish'}"
              f"   {len(rec.acts)} brain frames recorded")
        if fin and (best is None or t < best[0]):
            best = (t, list(rec.acts), dict(rr), dict(es), list(rec.qs))

    if best is None:
        raise SystemExit("no finished lap to save")

    t, acts, rr, es, qs = best
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sf = rr["state_float"]
    n = min(len(acts), len(sf), len(rr["frames"]), len(rr["actions"]))
    frames = np.stack([np.asarray(f, dtype=np.uint8) for f in rr["frames"][:n]])
    speeds = np.array([float(np.linalg.norm(sf[i][56:59])) * 3.6 for i in range(n)], dtype=np.float32)
    acts_arr = np.stack(acts[:n])
    np.savez_compressed(
        out / "lap.npz",
        activity=acts_arr,
        frames=frames,
        speed=speeds,
        actions=np.array([int(a) if isinstance(a, (int, np.integer)) else 0 for a in rr["actions"][:n]], dtype=np.int64),
        zones=np.array([int(z) if isinstance(z, (int, np.integer)) else 0 for z in rr["current_zone_idx"][:n]], dtype=np.int64),
        race_time=np.float32(t),
        ms_per_action=np.int32(config_copy.ms_per_action),
    )
    print(f"\nsaved {n} steps of a {t:.3f}s lap -> {out/'lap.npz'}")
    print(f"  activity {acts_arr.shape} {acts_arr.dtype}, frames {frames.shape}")
    try:
        tmi.close_game()
    except Exception:
        pass


if __name__ == "__main__":
    main()
