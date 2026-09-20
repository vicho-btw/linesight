<div align="center">

  <h3>Linesight</h3>

  Trackmania AI
  <br>
  <strong>[Linesight documentation][doc-link]</strong>
  <br>
  <br>
  [![Discord][doc-badge]][doc-link]
  [![Discord][discord-badge]][discord-link]

</div>

## Linesight

Linesight is a reinforcement learning project seeking to push what can be done with AI in Trackmania as far as possible. 

## Trackmania

Trackmania is a racing game that sacrifices some of the realism of sim-racers for a wide variety of track types with all kinds of tricks like wall riding, stunt jumps and wallbangs. Furthermore, Trackmania was designed for equality of input devices which means that keyboard inputs are a viable way to play and therefore that discrete input algorithms like DQN can be applied. In other words, Trackmania is a deep game which can serve as a benchmark to work on any RL algorithm.

## Trackmania Interface

Our work, combined with the efforts of [donadigo](https://github.com/donadigo) and [Kim](https://github.com/koyaanis) of the [Trackmania Interface team](https://donadigo.com/tminterface/) allow interfacing to [Trackmania Nations Forever](https://en.wikipedia.org/wiki/TrackMania#TrackMania_United). Allowing you to programmatically send inputs, get car states, get screenshots, etc... This part of our codebase could be useful to other RL projects.


## This fork: driving with a fruit fly connectome

This fork replaces Linesight's dense trunk with the **whole-brain connectome of an adult
*Drosophila melanogaster*** (FlyWire FAFB v783) and trains it to drive Trackmania.

The network is the fly's actual wiring diagram: **139,248 neurons** and **2,700,429 synaptic
connections**, with the *topology fixed* and only the synaptic weights trained. Connections
obey Dale's law — every neuron is purely excitatory or inhibitory according to its annotated
neurotransmitter (acetylcholine +1, GABA and glutamate −1; glutamate is inhibitory in
*Drosophila*, unlike vertebrates). The FlyWire `super_class` annotation is used as-is to wire
the brain into the agent: the 77,541 **optic** neurons receive the visual features, **sensory**
and **ascending** neurons receive the car-state floats, and the 1,303 **descending** neurons —
the ones that in a real fly carry commands to the motor centres — drive the Q-value readout.

It works. The connectome completes ESL-Hockolicious in **55.34 s**, and
`scripts/tools/flywire/render_lap_video.py` renders a lap with the agent's real 120×160
grayscale input beside all 139,248 neurons firing at their true FlyWire coordinates.

**It is not faster than the network it replaced.** The dense-trunk baseline in this repo
reaches **53.83 s** on the same track, and the human world record is **49.47 s**. Measured over
3,365 completed laps, the connectome's best is 54.40 s against the baseline's 53.83 s. The
interesting result is that a biological wiring diagram drives the track at all, not that it
drives it better.

### What is in here

| | |
|---|---|
| `trackmania_rl/agents/flybrain.py` | the connectome policy: custom sparse autograd, leaky-integrator settle loop |
| `scripts/tools/flywire/README_flywire.md` | how to build the connectome, and what each design choice costs |
| `scripts/tools/flywire/replay_to_actions.py` | extract a human replay's inputs; refuses analog (pad/wheel) replays |
| `scripts/tools/flywire/drive_replay.py` | reproduce a replay in-engine — bit-exact on three maps |
| `scripts/tools/flywire/ghost_to_pace.py` | build a per-checkpoint pace reference from any replay ghost |
| `scripts/tools/flywire/bc_train.py` | behaviour-clone a human lap (cross-entropy; a human has no Q-values) |
| `scripts/tools/flywire/render_lap_video.py` | the lap video, neurons at real FlyWire coordinates |
| `scripts/tools/flywire/live_viewer.py` | watch the connectome drive in a browser, live |

### Two things worth knowing if you build on this

**TMInterface latches a requested input on the *next* engine tick.** Replaying a recorded lap
one tick early is the difference between reproducing it exactly and crashing three seconds in.
`drive_replay.py --shift 1` compensates; without it, nothing reproduces.

**IQN samples its quantiles randomly at every decision**, so a greedy policy picks different
actions on identical states and lap times scatter by seconds. Evaluate with fixed quantiles
(`eval_agent.py --deterministic`) and explore with random ones — four laps went from a 31-point
spread to 0.2 points.

Connectome data is CC-BY-4.0; citations are in `scripts/tools/flywire/README_flywire.md`.

## Results

To our knowledge, Linesight is by far the most advanced AI in Trackmania. It was the first to demonstrate human-level driving around May 2023, with [Wirtual playing against it](https://www.youtube.com/watch?v=wjHW3ai47Og) in June. In May 2024, Linesight was the first to [showcase beating world records on official campaign tracks](https://www.youtube.com/watch?v=cUojVsCJ51I).

Now that the project is open-source, can you help make it even stronger?

[doc-link]: https://linesight-rl.github.io/linesight/build/html/
[discord-link]:       https://discord.gg/PvWYGkGKqd

[doc-badge]: https://img.shields.io/badge/Documentation-blue?style=for-the-badge&logoSize=small&logo=readthedocs
[discord-badge]: https://img.shields.io/discord/847108820479770686?style=for-the-badge&logo=discord&logoSize=auto&label=Discord
