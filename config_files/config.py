"""
This file contains a run's configuration.
It is expected that this file contains all relevant information about a run.

Two files named "config.py" and "config_copy.py" coexist in the same folder.

At the beginning of training, parameters are copied from config.py to config_copy.py
During training, config_copy.py will be reloaded at regular time intervals.
config_copy.py is NOT tracked in git, as it is essentially a temporary file.

Training parameters modifications made during training in config_copy.py will be applied on the fly
without losing the existing content of the replay buffer.

The content of config.py may be modified after starting a run: it will have no effect on the ongoing run.
This setup provides the possibility to:
1) Modify training parameters on the fly
2) Continue to code, use git, and modify config.py without impacting an ongoing run.

This file is preconfigured with sensible hyperparameters for the map ESL-Hockolicious, assuming the user
has a computer with 16GB RAM.
"""
from itertools import repeat

from config_files.inputs_list import *
from config_files.state_normalization import *
from config_files.user_config import *

W_downsized = 160
H_downsized = 120

run_name = "a09_rl"
running_speed = 80

tm_engine_step_per_action = 5
ms_per_tm_engine_step = 10
ms_per_action = ms_per_tm_engine_step * tm_engine_step_per_action
n_zone_centers_in_inputs = 40
one_every_n_zone_centers_in_inputs = 20
n_zone_centers_extrapolate_after_end_of_map = 1000
n_zone_centers_extrapolate_before_start_of_map = 20
n_prev_actions_in_inputs = 5
n_contact_material_physics_behavior_types = 4  # See contact_materials.py

# ----------------------------------------------------------------------------------------
# Exploring starts.
#
# Starting every rollout at the start line means the agent visits the last third of the track
# only on the rare laps it survives that far, so the sections it is worst at are the ones it
# practises least. Exploring starts (Sutton & Barto, 5.3) break that: a fraction of exploratory
# rollouts are rewound to a state banked earlier in a real lap, partway round the track.
#
# States are banked only from laps that began at the start line and only at moments of forward
# progress, so they sit on a clean racing line rather than against a wall. Sampling is weighted
# by where recent laps died, which makes this a curriculum that follows the agent's own
# weaknesses instead of a fixed section-by-section order.
#
# Only exploratory rollouts are affected, so eval lap times stay end-to-end and comparable.
exploring_starts_prob = 0.0          # share of exploratory rollouts that start mid-lap
exploring_starts_n_buckets = 20      # track is split into this many sections
exploring_starts_failure_decay = 0.98  # how fast an old failure stops attracting starts


# ----------------------------------------------------------------------------------------
# Checkpoint promotion.
#
# Lap times here carry roughly 1.7 s of noise: identical distilled weights drove 57.110 one day
# and 58.770 the next. A single fast lap is therefore not evidence that the policy improved, and
# promoting on one is how three checkpoints that crash at 4.5% of the track came to be recorded
# as bests while the genuinely good policy was overwritten.
#
# So promotion needs two things a single best lap does not give:
#   - a MEDIAN over a window of eval laps, with laps that did not finish counted at the cutoff.
#     Scoring only finishes is what let a policy that crashes almost immediately be promoted on
#     its one fluke lap.
#   - a snapshot taken in-process, at the moment the window closes. The previous external
#     watcher copied whatever weights were on disk minutes after the lap it was reacting to,
#     so the file it saved was never the network that drove the time.
checkpoint_eval_window = 7           # eval laps per promotion decision
checkpoint_promote_margin_ms = 100   # a candidate is only written if the median improves by this much

cutoff_rollout_if_race_not_finished_within_duration_ms = 300_000
cutoff_rollout_if_no_vcp_passed_within_duration_ms = 2_000

temporal_mini_race_duration_ms = 7000
temporal_mini_race_duration_actions = temporal_mini_race_duration_ms // ms_per_action
oversample_long_term_steps = 40
oversample_maximum_term_steps = 5
min_horizon_to_update_priority_actions = temporal_mini_race_duration_actions - 40
# If mini_race_time == mini_race_duration this is the end of the minirace
margin_to_announce_finish_meters = 700

global_schedule_speed = 1

# FINE-TUNING FROM A DISTILLED POLICY.
# The stock schedule starts at 1.0 -- 100% random actions for the first 50k frames -- which
# would erase the distilled policy within minutes of starting. This run begins from a network
# that already completes laps, so exploration starts where the stock schedule *ends*.
epsilon_schedule = [
    (0, 0.04),
    (15000000, 0.02),
]
epsilon_boltzmann_schedule = [
    (0, 0.15),
    (15000000, 0.06),
]
tau_epsilon_boltzmann = 0.01
discard_non_greedy_actions_in_nsteps = True
buffer_test_ratio = 0.05

engineered_speedslide_reward_schedule = [
    (0, 0),
]
engineered_neoslide_reward_schedule = [
    (0, 0),
]
engineered_kamikaze_reward_schedule = [
    (0, 0),
]
engineered_close_to_vcp_reward_schedule = [
    (0, 0),
]


# ----------------------------------------------------------------------------------------
# Crash penalty.
#
# TMInterface exposes no collision flag: state_float carries per-WHEEL contact (is_sliding,
# has_ground_contact, contact material under each wheel) but nothing that says the car's body
# hit a wall. The only reliable signature of an impact is a sudden loss of speed that braking
# cannot account for -- maximum braking sheds roughly 1-1.5 m/s per 50 ms action, so a drop
# beyond crash_penalty_threshold_m_per_s in one step is an impact.
#
# Note this is NOT potential-based shaping, so unlike shaped_reward_dist_to_cur_vcp it CAN
# change which policy is optimal. In Trackmania deliberate wall contact (wallbangs, wall
# riding) is sometimes the fast line, and a penalty discourages it. It is enabled here because
# the agent currently fails ~60% of laps, so reliability is worth more than exotic technique.
# Set crash_penalty_per_m_per_s to 0 to remove it.
crash_penalty_per_m_per_s = 0.0
crash_penalty_threshold_m_per_s = 5.0


# ----------------------------------------------------------------------------------------
# Pace shaping against the best lap so far.
#
# reference_pace[z] is the race time, in ms, at which the best lap so far reached virtual
# checkpoint z. The per-step shaping reward is
#
#     pace_shaping_coef * ( (ref[z_now] - ref[z_prev]) - ms_per_action )
#
# i.e. the time gained on the reference over that step: positive when the car covers a stretch
# faster than the reference did, negative when it loses ground. Summed over a lap it
# telescopes to the total time gained, so it is the difference of a potential
# Phi(s) = ref_time_at_position - race_time, and therefore policy-invariant (Ng, Harada &
# Russell 1999) -- it can change how fast learning happens, never which policy is optimal.
#
# The reference ratchets: whenever a finished lap beats it, that lap becomes the new
# reference and the bar moves. Delete save/<run>/reference_pace.npy to reset it.
pace_shaping_coef = 0.0      # same units as constant_reward_per_ms: 1 s gained ~ +1.2
pace_reference_min_improvement_ms = 50

n_steps = 15
constant_reward_per_ms = -6 / 5000
reward_per_m_advanced_along_centerline = 5 / 500

float_input_dim = 27 + 3 * n_zone_centers_in_inputs + 4 * n_prev_actions_in_inputs + 4 * n_contact_material_physics_behavior_types + 1
float_hidden_dim = 256
conv_head_output_dim = 5632
dense_hidden_dimension = 1024
iqn_embedding_dimension = 64

# ======================================================================================================================
#                                          FLY BRAIN (FlyWire FAFB v783)
# ======================================================================================================================
# When use_fly_brain is True, the dense trunk between the input heads and the IQN dueling heads
# is replaced by a recurrent network wired from the Drosophila whole-brain connectome.
# The topology is fixed by the connectome; the weights train.
# Build the artifact first:  python scripts/tools/flywire/build_connectome.py
use_fly_brain = True
fly_connectome_path = Path(__file__).resolve().parents[1] / "data" / "flywire" / "connectome_783.npz"
# Shape of the conv head's feature map before flattening, used for the retinotopic map onto
# the optic lobe. Must match img_head's output: (32, 11, 16) for a 120x160 input.
fly_visual_channels = 32
fly_visual_h = 11
fly_visual_w = 16
fly_readout_dim = 512  # width of the embedding decoded from descending-neuron activity
# Each settle step is one step of a leaky integrator, so the step size is a physical quantity,
# not a free knob: dt = ms_per_action / fly_n_settle_steps. Drosophila central neurons have a
# membrane time constant near 16 ms (MBON-alpha3: 16.06 +/- 4.92 ms, Pribbenow et al. 2022,
# eLife 77578). 4 steps gives dt = 12.5 ms, the coarsest step still under one time constant,
# which is what the discretisation needs. The per-neuron leak is initialised to exp(-dt/tau)
# and trains from there.
# 4 settle steps: dt = 12.5 ms against a 16 ms membrane, so dt/tau = 0.78 < 1. That is
# both the biological condition and the numerical stability condition for the Euler step --
# at 2 steps dt/tau = 1.56 and the integrator is unstable.
fly_n_settle_steps = 4
fly_membrane_tau_ms = 16.0
fly_sensory_rank = 64  # rank of the float-features -> sensory-neuron projection
fly_init_gain = 0.25  # initial recurrent gain; > 1 risks a diverging settle loop
fly_dale = False  # True pins each neuron's excitatory/inhibitory sign for the whole of training
fly_step_norm = True  # normalise activity between settle steps
fly_dt_ms = ms_per_action / fly_n_settle_steps
# Live activity tap: publishes the driving network's settled activity to a memory-mapped
# file that scripts/tools/flywire/live_viewer.py renders. Worker 0 only, every Nth inference.
fly_live_tap = True
fly_live_tap_path = Path(__file__).resolve().parents[1] / "data" / "flywire" / "live_activity.mmap"
fly_live_tap_every = 4
# Doubled from 8. This is the one variance reduction that is nearly free here: the
# connectome trunk runs BEFORE the quantile expansion, at batch_size, so more quantiles cost
# only the small dueling heads -- unlike the baseline, where the whole 5888-wide trunk is
# replicated per quantile. The IQN loss is a sample-mean over quantile pairs, so 8 -> 16
# halves that component of the gradient variance.
iqn_n = 16

# ======================================================================================================================
#                              KL ANCHOR TO THE DISTILLED POLICY
# ======================================================================================================================
# Fine-tuning a distilled policy drifts. Measured on 4,096 held-out states: agreement with the
# teacher fell from 86.9% (right after distillation) to 32.7% after 940k frames of RL, while the
# best lap did not improve at all -- 54.550s at 193k frames and still 54.550s at 940k. Two thirds
# of the policy's decisions changed and bought nothing.
#
# The standard remedy is to anchor the policy to its reference: RL may still improve it, but every
# departure has to be worth the KL cost. Set kl_anchor_weight to 0 to disable.
kl_anchor_weight = 0.0
kl_anchor_temp = 0.05          # same temperature the distillation used
kl_anchor_run = "_anchor"   # PINNED to the verified-good policy; no code moves it, only a human
kl_anchor_reload_every = 400   # batches between checks for a newer reference
kl_anchor_quantiles = 8        # quantiles for the reference forward; fewer = cheaper, it only needs a policy
  # must be an even number because we sample tau symmetrically around 0.5
iqn_k = 32  # must be an even number because we sample tau symmetrically around 0.5
iqn_kappa = 5e-3
use_ddqn = False

prio_alpha = np.float32(0.3)  # Rainbow-IQN paper: 0.2, Rainbow paper: 0.5, PER paper 0.6
prio_epsilon = np.float32(2e-3)  # Defaults to 10^-6 in stable-baselines
prio_beta = np.float32(1)

# Replay ratio: the fly trunk trains at ~322 ms/batch vs ~40 ms for the dense trunk, so
# reusing each memory 32 times costs ~20 ms of GPU per collected frame and oversubscribes
# the card, starving the collectors (race_time_ratio fell 3.3x -> 0.85x). 8 keeps the
# learner at roughly the same GPU share the baseline had.
number_times_single_memory_is_used_before_discard = 16

memory_size_schedule = [
    (0, (50000, 20000)),
    (25000000, (100000, 75000)),
    (35000000, (200000, 150000)),
]
# Fine-tuning step: lr * clip = 0.0075, a quarter of the from-scratch rate. Large updates
# here would undo distillation faster than RL could rebuild it.
lr_schedule = [
    (0, 0.00015),
    (15000000, 5e-05),
    (60000000, 5e-05),
    (75000000, 1e-05),
]
tensorboard_suffix_schedule = [
    (0, ""),
    (6_000_000 * global_schedule_speed, "_2"),
    (15_000_000 * global_schedule_speed, "_3"),
    (30_000_000 * global_schedule_speed, "_4"),
    (45_000_000 * global_schedule_speed, "_5"),
    (80_000_000 * global_schedule_speed, "_6"),
    (150_000_000 * global_schedule_speed, "_7"),
]
gamma_schedule = [
    (0, 0.99979983),
    (7500000, 0.99979983),
    (12500000, 1),
]

batch_size = 512
weight_decay_lr_ratio = 1 / 50
adam_epsilon = 1e-4
adam_beta1 = 0.9
adam_beta2 = 0.999

single_reset_flag = 0
reset_every_n_frames_generated = 400_000_00000000
additional_transition_after_reset = 1_600_000
last_layer_reset_factor = 0.8  # 0 : full reset, 1 : nothing happens
overall_reset_mul_factor = 0.01  # 0 : nothing happens ; 1 : full reset

clip_grad_value = 1000
# Effective update magnitude is lr * clip_grad_norm whenever the raw gradient norm far
# exceeds the clip, which it does here: backprop through 4 recurrent settle steps produces
# norms of 3e3-6e3, so every step is normalised. Measured: lr*clip = 3.0e-2 diverged,
# 6.0e-3 barely learned. 1.5e-2 sits between them.
clip_grad_norm = 50

number_memories_trained_on_between_target_network_updates = 2048
soft_update_tau = 0.02

distance_between_checkpoints = 0.5
road_width = 90  ## a little bit of margin, could be closer to 24 probably ? Don't take risks there are curvy roads
max_allowable_distance_to_virtual_checkpoint = np.sqrt((distance_between_checkpoints / 2) ** 2 + (road_width / 2) ** 2)

timeout_during_run_ms = 10_100
timeout_between_runs_ms = 600_000_000
tmi_protection_timeout_s = 500
game_reboot_interval = 3600 * 12  # In seconds

frames_before_save_best_runs = 7_500_000

plot_race_time_left_curves = False
n_transitions_to_plot_in_distribution_curves = 1000
make_highest_prio_figures = False
apply_randomcrop_augmentation = False
n_pixels_to_crop_on_each_side = 2

max_rollout_queue_size = 1

use_jit = True

# gpu_collectors_count is the number of Trackmania instances that will be launched in parallel.
# It is recommended that users adjust this number depending on the performance of their machine.
# We recommend trying different values and finding the one that maximises the number of batches done per unit of time.
# One collector, deliberately. Two fullscreen TmForever instances cannot both stay
# un-minimized: whichever loses focus minimizes, stops rendering, stops answering
# TMInterface, and every rollout on it dies with a WinError 10060 timeout. Raise this back
# to 2 only after confirming the game launches windowed.
gpu_collectors_count = 1

send_shared_network_every_n_batches = 10
update_inference_network_every_n_actions = 20

target_self_loss_clamp_ratio = 4

final_speed_reward_as_if_duration_s = 0
final_speed_reward_per_m_per_s = reward_per_m_advanced_along_centerline * final_speed_reward_as_if_duration_s

shaped_reward_dist_to_cur_vcp = -0.1
shaped_reward_min_dist_to_cur_vcp = 2
shaped_reward_max_dist_to_cur_vcp = 25
engineered_reward_min_dist_to_cur_vcp = 5
engineered_reward_max_dist_to_cur_vcp = 25
shaped_reward_point_to_vcp_ahead = 0

threshold_to_save_all_runs_ms = -1

deck_height = -np.inf
game_camera_number = 2

sync_virtual_and_real_checkpoints = True

""" 
============================================      MAP CYCLE     =======================================================

In this section we define the map cycle.

It is a list of iterators, each iterator must return tuples with the following information:
    - short map name        (string):     for logging purposes
    - map path              (string):     to automatically load the map in game. 
                                          This is the same map name as the "map" command in the TMInterface console.
    - reference line path   (string):     where to find the reference line for this map
    - is_explo              (boolean):    whether the policy when running on this map should be exploratory
    - fill_buffer           (boolean):    whether the memories generated during this run should be placed in the buffer 

The map cycle may seem complex at first glance, but it provides a large amount of flexibility:
    - can train on some maps, test blindly on others
    - can train more on some maps, less on others
    - can define multiple reference lines for a given map
    - etc...

The example below defines a simple cycle where the agent alternates between four exploratory runs on map5, and one 
evaluation run on the same map.

map_cycle = [
    repeat(("map5", '"My Challenges/Map5.Challenge.Gbx"', "map5_0.5m_cl.npy", True, True), 4),
    repeat(("map5", '"My Challenges/Map5.Challenge.Gbx"', "map5_0.5m_cl.npy", False, True), 1),
]
"""

nadeo_maps_to_train_and_test = [
    "A01-Race",
    # "A02-Race",
    "A03-Race",
    # "A04-Acrobatic",
    "A05-Race",
    # "A06-Obstacle",
    "A07-Race",
    # "A08-Endurance",
    # "A09-Race",
    # "A10-Acrobatic",
    "A11-Race",
    # "A12-Speed",
    # "A13-Race",
    "A14-Race",
    "A15-Speed",
    "B01-Race",
    "B02-Race",
    "B03-Race",
    # "B04-Acrobatic",
    "B05-Race",
    # "B06-Obstacle",
    # "B07-Race",
    # "B08-Endurance",
    # "B09-Acrobatic",
    "B10-Speed",
    # "B11-Race",
    # "B12-Race",
    # "B13-Obstacle",
    "B14-Speed",
    # "B15-Race",
]

map_cycle = []
# for map_name in nadeo_maps_to_train_and_test:
# short_map_name = map_name[0:3]
# map_cycle.append(repeat((short_map_name, f'"Official Maps\{map_name}.Challenge.Gbx"', f"{map_name}_0.5m_cl2.npy", True, True), 4))
# map_cycle.append(repeat((short_map_name, f'"Official Maps\{map_name}.Challenge.Gbx"', f"{map_name}_0.5m_cl2.npy", False, True), 1))


map_cycle += [
    # repeat(("map5", '"My Challenges/Map5.Challenge.Gbx"', "map5_0.5m_cl.npy", True, True), 4),
    # repeat(("map5", '"My Challenges/Map5.Challenge.Gbx"', "map5_0.5m_cl.npy", False, True), 1),
    # repeat(("map8", '"My Challenges/Map8.Challenge.Gbx"', "map8_0.5m_cl.npy", True, True), 4),
    # repeat(("map8", '"My Challenges/Map8.Challenge.Gbx"', "map8_0.5m_cl.npy", False, True), 1),
    # repeat(("yosh1", '"My Challenges\Yosh1.Challenge.Gbx"', "yosh1_0.5m_clprog.npy", True, True), 4),
    # repeat(("yosh1", '"My Challenges\Yosh1.Challenge.Gbx"', "yosh1_0.5m_clprog.npy", False, True), 1),
    # repeat(("wallb1", "Wallbang_full.Challenge.Gbx", "Wallbang_full_0.5m_cl.npy", True, True), 4),
    # repeat(("wallb1", "Wallbang_full.Challenge.Gbx", "Wallbang_full_0.5m_cl.npy", False, True), 1),
    # repeat(("yosh3", '"My Challenges\Yosh3.Challenge.Gbx"', "yosh3_0.5m_clprog_cut1.npy", True, True), 4),
    # repeat(("yosh3", '"My Challenges\Yosh3.Challenge.Gbx"', "yosh3_0.5m_clprog_cut1.npy", False, True), 1),
    # repeat(("A06", '"Official Maps\White\A06-Obstacle.Challenge.Gbx"', "A06-Obstacle_10m_cl.npy", True, True), 4),
    # repeat(("A06", '"Official Maps\White\A06-Obstacle.Challenge.Gbx"', "A06-Obstacle_10m_cl.npy", False, True), 1),
    # repeat(("A07", '"Official Maps\White\A07-Race.Challenge.Gbx"', "A07-Race_10m_cl.npy", True, True), 4),
    # repeat(("A07", '"Official Maps\White\A07-Race.Challenge.Gbx"', "A07-Race_10m_cl.npy", False, True), 1),
    # repeat(("B01", '"Official Maps\Green\B01-Race.Challenge.Gbx"', "B01-Race_10m_cl.npy", True, True), 4),
    # repeat(("B01", '"Official Maps\Green\B01-Race.Challenge.Gbx"', "B01-Race_10m_cl.npy", False, True), 1),
    # repeat(("B02", '"Official Maps\Green\B02-Race.Challenge.Gbx"', "B02-Race_10m_cl.npy", True, True), 4),
    # repeat(("B02", '"Official Maps\Green\B02-Race.Challenge.Gbx"', "B02-Race_10m_cl.npy", False, True), 1),
    # repeat(("B03", '"Official Maps\Green\B03-Race.Challenge.Gbx"', "B03-Race_10m_cl.npy", True, True), 4),
    # repeat(("B03", '"Official Maps\Green\B03-Race.Challenge.Gbx"', "B03-Race_10m_cl.npy", False, True), 1),
    # repeat(("B05", '"Official Maps\Green\B05-Race.Challenge.Gbx"', "B05-Race_10m_cl.npy", True, True), 4),
    # repeat(("B05", '"Official Maps\Green\B05-Race.Challenge.Gbx"', "B05-Race_10m_cl.npy", False, True), 1),
    repeat(("a09", "A09-Race.Challenge.Gbx", "A09-Race_0.5m.npy", True, True), 4),
    repeat(("a09", "A09-Race.Challenge.Gbx", "A09-Race_0.5m.npy", False, True), 1),
    # repeat(("hock", "ESL-Hockolicious.Challenge.Gbx", "ESL-Hockolicious_0.5m_cl2.npy", True, True), 4),
    # repeat(("hock", "ESL-Hockolicious.Challenge.Gbx", "ESL-Hockolicious_0.5m_cl2.npy", False, True), 1),
    # repeat(("A02", f'"Official Maps\A02-Race.Challenge.Gbx"', "A02-Race_0.5m_cl2.npy", False, False), 1),
    # repeat(("yellowmile", f'"The Yellow Mile_.Challenge.Gbx"', "YellowMile_0.5m_cl.npy", False, False), 1),
    # repeat(("te86", f'"te 86.Challenge.Gbx"', "te86_0.5m_cl.npy", False, False), 1),
    # repeat(("minishort037", f'"Mini-Short.037.Challenge.Gbx"', "minishort037_0.5m_cl.npy", False, False), 1),
    # repeat(("map3", '"My Challenges\Map3_nowalls.Challenge.Gbx"', "map3_0.5m_cl.npy", False, False), 1),
    # repeat(("wallb1", "Wallbang_full.Challenge.Gbx", "Wallbang_full_0.5m_cl.npy", False, False), 1),
    # repeat(("hock", "ESL-Hockolicious.Challenge.Gbx", "ESL-Hockolicious_0.5m_cl2.npy", False, False), 1),
    # repeat(("A01", f'"Official Maps\A01-Race.Challenge.Gbx"', f"A01-Race_0.5m_cl2.npy", True, True), 4),
    # repeat(("A01", f'"Official Maps\A01-Race.Challenge.Gbx"', f"A01-Race_0.5m_cl2.npy", False, True), 1),
    # repeat(("A02", f'"Official Maps\A02-Race.Challenge.Gbx"', f"A02-Race_0.5m_alyen.npy", True, True), 4),
    # repeat(("A02", f'"Official Maps\A02-Race.Challenge.Gbx"', f"A02-Race_0.5m_alyen.npy", False, True), 1),
    # repeat(("A01", f'"Official Maps\A01-Race.Challenge.Gbx"', f"A01-Race_0.5m_rollin.npy", True, True), 4),
    # repeat(("A01", f'"Official Maps\A01-Race.Challenge.Gbx"', f"A01-Race_0.5m_rollin.npy", False, True), 1),
    # repeat(("A11", f'"Official Maps\A11-Race.Challenge.Gbx"', f"A11-Race_0.5m_cl2.npy", True, True), 4),
    # repeat(("A11", f'"Official Maps\A11-Race.Challenge.Gbx"', f"A11-Race_0.5m_cl2.npy", False, True), 1),
    # repeat(("A15", f'"Official Maps\A15-Speed.Challenge.Gbx"', f"A15-Speed_0.5m_hefest.npy", True, True), 4),
    # repeat(("A15", f'"Official Maps\A15-Speed.Challenge.Gbx"', f"A15-Speed_0.5m_hefest.npy", False, True), 1),
    # repeat(("E02", f'"Official Maps\E02-Endurance.Challenge.Gbx"', f"E02-Endurance_0.5m_karjen.npy", True, True), 4),
    # repeat(("E02", f'"Official Maps\E02-Endurance.Challenge.Gbx"', f"E02-Endurance_0.5m_karjen.npy", False, True), 1),
    # repeat(("minitrial1", f'"Minitrial 1.Challenge.Gbx"', f"minitrial1_0.5m_gizmo-levon.npy", True, True), 4),
    # repeat(("minitrial1", f'"Minitrial 1.Challenge.Gbx"', f"minitrial1_0.5m_gizmo-levon.npy", False, True), 1),
    # repeat(("minitrial1", f'"Minitrial 1.Challenge.Gbx"', f"minitrial1_0.5m_gizmo.npy", True, True), 4),
    # repeat(("minitrial1", f'"Minitrial 1.Challenge.Gbx"', f"minitrial1_0.5m_gizmo.npy", False, True), 1),
    # repeat(("D06", '"Official Maps/D06-Obstacle.Challenge.Gbx"', f"D06-Obstacle_0.5m_darkbringer.npy", True, True), 4),
    # repeat(("D06", '"Official Maps/D06-Obstacle.Challenge.Gbx"', f"D06-Obstacle_0.5m_darkbringer.npy", False, True), 1),
    # repeat(("D06", '"Official Maps/D06-Obstacle.Challenge.Gbx"', f"D06-Obstacle_0.5m_linesight2rollin3.npy", True, True), 4),
    # repeat(("D06", '"Official Maps/D06-Obstacle.Challenge.Gbx"', f"D06-Obstacle_0.5m_linesight2rollin3.npy", False, True), 1),
    # repeat(("D15", '"Official Maps\D15-Endurance.Challenge.Gbx"', f"D15-Endurance_0.5m_gwenlap3.npy", True, True), 4),
    # repeat(("D15", '"Official Maps\D15-Endurance.Challenge.Gbx"', f"D15-Endurance_0.5m_gwenlap3.npy", False, True), 1),
    # repeat(("C12", '"Official Maps\C12-Obstacle.Challenge.Gbx"', f"C12-Obstacle_0.5m_weapon.npy", True, True), 4),
    # repeat(("C12", '"Official Maps\C12-Obstacle.Challenge.Gbx"', f"C12-Obstacle_0.5m_weapon.npy", False, True), 1),
    # repeat(("D15olnc", '"D15-Endurance True One Lap No Cut.Challenge.Gbx"', f"D15-OnelapNocut_0.5m_wirtual.npy", True, True), 4),
    # repeat(("D15olnc", '"D15-Endurance True One Lap No Cut.Challenge.Gbx"', f"D15-OnelapNocut_0.5m_wirtual.npy", False, True), 1),
    # repeat(("E03", '"E03-Endurance No Cut.Challenge.Gbx"', f"E03-Endurance_0.5m_racehansnocutlap3.npy", True, True), 4),
    # repeat(("E03", '"E03-Endurance No Cut.Challenge.Gbx"', f"E03-Endurance_0.5m_racehansnocutlap3.npy", False, True), 1),
    # repeat(("E03", '"E03-Endurance No Cut.Challenge.Gbx"', f"E03-Endurance_0.5m_linesight2racehans3.npy", True, True), 4),
    # repeat(("E03", '"E03-Endurance No Cut.Challenge.Gbx"', f"E03-Endurance_0.5m_linesight2racehans3.npy", False, True), 1),
    # repeat(("A07", f'"Official Maps/A07-Race.Challenge.Gbx"', f"A07-Race_0.5m_raceta.npy", True, True), 4),
    # repeat(("A07", f'"Official Maps/A07-Race.Challenge.Gbx"', f"A07-Race_0.5m_raceta.npy", False, True), 1),
]
