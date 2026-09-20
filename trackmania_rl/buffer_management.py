"""
This file's main entry point is the function fill_buffer_from_rollout_with_n_steps_rule().
Its main inputs are a rollout_results object (obtained from a GameInstanceManager object), and a buffer to be filled.
It reassembles the rollout_results object into transitions, as defined in /trackmania_rl/experience_replay/experience_replay_interface.py
"""

import math
import random
from pathlib import Path

import numpy as np
from trackmania_rl.numba_compat import jit
from torchrl.data import ReplayBuffer

from config_files import config_copy
from trackmania_rl.experience_replay.experience_replay_interface import Experience
from trackmania_rl.reward_shaping import speedslide_quality_tarmac


@jit(nopython=True)
def get_potential(state_float):
    # https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
    vector_vcp_to_vcp_further_ahead = state_float[65:68] - state_float[62:65]
    vector_vcp_to_vcp_further_ahead_normalized = vector_vcp_to_vcp_further_ahead / np.linalg.norm(vector_vcp_to_vcp_further_ahead)

    return (
        config_copy.shaped_reward_dist_to_cur_vcp
        * max(
            config_copy.shaped_reward_min_dist_to_cur_vcp,
            min(config_copy.shaped_reward_max_dist_to_cur_vcp, np.linalg.norm(state_float[62:65])),
        )
    ) + (config_copy.shaped_reward_point_to_vcp_ahead * (vector_vcp_to_vcp_further_ahead_normalized[2] - 1))




# ---------------------------------------------------------------------- pace reference
# Cached so every rollout does not re-read the file; refreshed when it changes on disk.
_pace_cache = {"path": None, "mtime": 0.0, "table": None}


def _pace_reference_path() -> Path:
    return Path(__file__).resolve().parents[1] / "save" / config_copy.run_name / "reference_pace.npy"


def get_pace_reference():
    """Race time in ms at which the best lap so far reached each virtual checkpoint."""
    path = _pace_reference_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    if _pace_cache["table"] is None or _pace_cache["path"] != path or mtime > _pace_cache["mtime"]:
        try:
            _pace_cache["table"] = np.load(path)
            _pace_cache["path"] = path
            _pace_cache["mtime"] = mtime
        except Exception:
            return None
    return _pace_cache["table"]


def update_pace_reference(rollout_results) -> bool:
    """If this lap finished and beat the reference, it becomes the new reference."""
    if "race_time" not in rollout_results:
        return False
    if "exploring_start_zone" in rollout_results:
        # This lap was rewound to the middle of the track, so its race_time covers a stretch the
        # policy never drove. Taking it as the reference would invent a pace nothing can match.
        return False
    zones = rollout_results["current_zone_idx"]
    n = len(rollout_results["frames"])
    if n < 2:
        return False
    try:
        max_zone = int(max(z for z in zones[:n] if isinstance(z, (int, np.integer))))
    except ValueError:
        return False

    current = get_pace_reference()
    race_time = rollout_results["race_time"]
    if current is not None and max_zone < len(current) - 1:
        return False  # did not get as far as the reference
    if current is not None:
        prev_total = float(current[min(max_zone, len(current) - 1)])
        if race_time > prev_total - config_copy.pace_reference_min_improvement_ms:
            return False

    # First arrival time at each zone, forward-filled so every index is defined.
    table = np.full(max_zone + 1, np.inf, dtype=np.float64)
    for i in range(n):
        z = zones[i]
        if not isinstance(z, (int, np.integer)) or z < 0 or z > max_zone:
            continue
        t = i * config_copy.ms_per_action
        if t < table[z]:
            table[z] = t
    last = 0.0
    for z in range(len(table)):
        if not np.isfinite(table[z]):
            table[z] = last
        else:
            last = table[z]
    path = _pace_reference_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # np.save appends .npy to any name that lacks it, so the temp file must already
        # end in .npy or the atomic replace below silently has nothing to rename.
        tmp = path.with_name(path.stem + ".tmp.npy")
        np.save(tmp, table)
        tmp.replace(path)
        print(f"pace reference advanced: {race_time / 1000:.3f}s over {len(table):,} checkpoints", flush=True)
        return True
    except OSError as e:
        print(f"could not write pace reference: {e}", flush=True)
        return False


def fill_buffer_from_rollout_with_n_steps_rule(
    buffer: ReplayBuffer,
    buffer_test: ReplayBuffer,
    rollout_results: dict,
    n_steps_max: int,
    gamma: float,
    discard_non_greedy_actions_in_nsteps: bool,
    engineered_speedslide_reward: float,
    engineered_neoslide_reward: float,
    engineered_kamikaze_reward: float,
    engineered_close_to_vcp_reward: float,
):
    assert len(rollout_results["frames"]) == len(rollout_results["current_zone_idx"])
    n_frames = len(rollout_results["frames"])

    number_memories_added_train = 0
    number_memories_added_test = 0
    Experiences_For_Buffer = []
    Experiences_For_Buffer_Test = []
    list_to_fill = Experiences_For_Buffer_Test if random.random() < config_copy.buffer_test_ratio else Experiences_For_Buffer

    gammas = (gamma ** np.linspace(1, n_steps_max, n_steps_max)).astype(
        np.float32
    )  # Discount factor that will be placed in front of next_step in Bellman equation, depending on n_steps chosen

    pace_ref = get_pace_reference()
    update_pace_reference(rollout_results)

    reward_into = np.zeros(n_frames)
    for i in range(1, n_frames):
        reward_into[i] += config_copy.constant_reward_per_ms * (
            config_copy.ms_per_action
            if (i < n_frames - 1 or ("race_time" not in rollout_results))
            else rollout_results["race_time"] - (n_frames - 2) * config_copy.ms_per_action
        )
        reward_into[i] += (
            rollout_results["meters_advanced_along_centerline"][i] - rollout_results["meters_advanced_along_centerline"][i - 1]
        ) * config_copy.reward_per_m_advanced_along_centerline

        # Crash penalty: a single-step speed loss larger than hard braking can produce.
        # The implicit cost of a wall hit (fewer metres advanced over the following seconds)
        # is real but diffuse, and has to travel back through bootstrapping to reach the turn
        # that caused it. This puts an unambiguous cost on the action itself.
        # Pace shaping: time gained on the best lap so far over this step. A potential
        # difference, so it cannot change the optimal policy -- only how quickly the agent
        # discovers where it is losing time.
        pace_coef = getattr(config_copy, "pace_shaping_coef", 0.0)
        if pace_coef != 0.0 and pace_ref is not None:
            z_now, z_prev = rollout_results["current_zone_idx"][i], rollout_results["current_zone_idx"][i - 1]
            if isinstance(z_now, (int, np.integer)) and isinstance(z_prev, (int, np.integer)):
                zn = min(max(int(z_now), 0), len(pace_ref) - 1)
                zp = min(max(int(z_prev), 0), len(pace_ref) - 1)
                reward_into[i] += pace_coef * ((pace_ref[zn] - pace_ref[zp]) - config_copy.ms_per_action)

        crash_coef = getattr(config_copy, "crash_penalty_per_m_per_s", 0.0)
        # On a lap that FINISHES, rollout() appends one extra entry to frames and
        # current_zone_idx at the finish line but not to state_float, so state_float is one
        # short; on a lap that does not finish they are equal. That is why every state_float
        # access here is guarded, and skipping the final step costs nothing either way.
        if crash_coef != 0.0 and i < n_frames - 1:
            # Only speed the agent did not ASK to lose. Actions 6-11 all press brake, and
            # hard braking sheds up to ~5 m/s per step legitimately; counting that would teach
            # the car not to brake into corners. Even with braking exempt, the 54.370 teacher
            # still trips a 2.5 m/s threshold ~5 times a lap -- brushing walls is part of the
            # fast line here -- so the threshold is set above the teacher's p99.9 (4.85 m/s)
            # to catch only impacts that actually end runs.
            prev_action = rollout_results["actions"][i - 1] if i - 1 < len(rollout_results["actions"]) else 0
            was_braking = isinstance(prev_action, (int, np.integer)) and prev_action >= 6
            if not was_braking:
                speed_now = np.linalg.norm(rollout_results["state_float"][i][56:59])
                speed_prev = np.linalg.norm(rollout_results["state_float"][i - 1][56:59])
                lost = speed_prev - speed_now
                threshold = getattr(config_copy, "crash_penalty_threshold_m_per_s", 5.0)
                if lost > threshold:
                    reward_into[i] -= crash_coef * (lost - threshold)
        if i < n_frames - 1:
            if config_copy.final_speed_reward_per_m_per_s != 0 and rollout_results["state_float"][i][58] > 0:
                # car has velocity *forward*
                reward_into[i] += config_copy.final_speed_reward_per_m_per_s * (
                    np.linalg.norm(rollout_results["state_float"][i][56:59]) - np.linalg.norm(rollout_results["state_float"][i - 1][56:59])
                )
            if engineered_speedslide_reward != 0 and np.all(rollout_results["state_float"][i][25:29]):
                # all wheels touch the ground
                reward_into[i] += engineered_speedslide_reward * max(
                    0.0,
                    1 - abs(speedslide_quality_tarmac(rollout_results["state_float"][i][56], rollout_results["state_float"][i][58]) - 1),
                )  # TODO : indices 25:29, 56 and 58 are hardcoded, this is bad....

            # lateral speed is higher than 2 meters per second
            reward_into[i] += (
                engineered_neoslide_reward if abs(rollout_results["state_float"][i][56]) >= 2.0 else 0
            )  # TODO : 56 is hardcoded, this is bad....
            # kamikaze reward
            if (
                engineered_kamikaze_reward != 0
                and rollout_results["actions"][i] <= 2
                or np.sum(rollout_results["state_float"][i][25:29]) <= 1
            ):
                reward_into[i] += engineered_kamikaze_reward
            if engineered_close_to_vcp_reward != 0:
                reward_into[i] += engineered_close_to_vcp_reward * max(
                    config_copy.engineered_reward_min_dist_to_cur_vcp,
                    min(config_copy.engineered_reward_max_dist_to_cur_vcp, np.linalg.norm(rollout_results["state_float"][i][62:65])),
                )
    for i in range(n_frames - 1):  # Loop over all frames that were generated
        # Switch memory buffer sometimes
        if random.random() < 0.1:
            list_to_fill = Experiences_For_Buffer_Test if random.random() < config_copy.buffer_test_ratio else Experiences_For_Buffer

        n_steps = min(n_steps_max, n_frames - 1 - i)
        if discard_non_greedy_actions_in_nsteps:
            try:
                first_non_greedy = rollout_results["action_was_greedy"][i + 1 : i + n_steps].index(False) + 1
                n_steps = min(n_steps, first_non_greedy)
            except ValueError:
                pass

        rewards = np.empty(n_steps_max).astype(np.float32)
        for j in range(n_steps):
            rewards[j] = (gamma**j) * reward_into[i + j + 1] + (rewards[j - 1] if j >= 1 else 0)

        state_img = rollout_results["frames"][i]
        state_float = rollout_results["state_float"][i]
        state_potential = get_potential(rollout_results["state_float"][i])

        # Get action that was played
        action = rollout_results["actions"][i]
        terminal_actions = float((n_frames - 1) - i) if "race_time" in rollout_results else math.inf
        next_state_has_passed_finish = ((i + n_steps) == (n_frames - 1)) and ("race_time" in rollout_results)

        if not next_state_has_passed_finish:
            next_state_img = rollout_results["frames"][i + n_steps]
            next_state_float = rollout_results["state_float"][i + n_steps]
            next_state_potential = get_potential(rollout_results["state_float"][i + n_steps])
        else:
            # It doesn't matter what next_state_img and next_state_float contain, as the transition will be forced to be final
            next_state_img = state_img
            next_state_float = state_float
            next_state_potential = 0

        list_to_fill.append(
            Experience(
                state_img,
                state_float,
                state_potential,
                action,
                n_steps,
                rewards,
                next_state_img,
                next_state_float,
                next_state_potential,
                gammas,
                terminal_actions,
            )
        )
    number_memories_added_train += len(Experiences_For_Buffer)
    if len(Experiences_For_Buffer) > 1:
        buffer.extend(Experiences_For_Buffer)
    elif len(Experiences_For_Buffer) == 1:
        buffer.add(Experiences_For_Buffer[0])
    number_memories_added_test += len(Experiences_For_Buffer_Test)
    if len(Experiences_For_Buffer_Test) > 1:
        buffer_test.extend(Experiences_For_Buffer_Test)
    elif len(Experiences_For_Buffer_Test) == 1:
        buffer_test.add(Experiences_For_Buffer_Test[0])

    return buffer, buffer_test, number_memories_added_train, number_memories_added_test
