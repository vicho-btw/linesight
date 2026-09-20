# Driving Trackmania with the fly connectome

This replaces Linesight's dense trunk with a recurrent network whose wiring diagram is the
*Drosophila melanogaster* whole-brain connectome (FlyWire FAFB v783). The connectome fixes
the **topology**; synapse counts and neurotransmitter predictions set the **initial weights**;
the weights then train by ordinary backprop through the IQN loss.

## Why this is affordable

The headline figure for FlyWire is "~50M synapses", which sounds fatal for a laptop GPU. It
is not the relevant number. Those synaptic contacts collapse into **2,700,429 neuron-to-neuron
connections** at the standard 5-synapse threshold, and a connection is one trainable weight.
That is *fewer* parameters than the dense trunk being replaced.

The second saving is placement. The brain sits **before** the IQN quantile expansion, so it
processes `batch_size` samples rather than `batch_size * iqn_n`. At Linesight's defaults that
is an 8x reduction. The quantile machinery is distributional-RL bookkeeping, not part of the
animal, so there is no reason for the brain to see it.

## Measured performance

Full `IQN_Network`, batch 512, RTX 5080 Laptop (16 GiB), real v783 connectome:

| | fwd+bwd | peak GPU |
|---|---|---|
| Baseline Linesight | 40.1 ms | 0.82 GiB |
| Fly brain, 4 settle steps | 322 ms | 6.24 GiB |

Rollout inference (batch 1, `iqn_k=32`): baseline 2.09 ms, fly brain 5.55 ms. Note the
baseline is *already* inference-limited at `running_speed = 80` (it wants 1600 actions/s per
instance and gets ~479), so the practical rollout slowdown is about 2.7x, not 8x.

Getting to 322 ms took three fixes, each necessary:

1. **COO `torch.sparse.mm` is unusable at this scale.** Its backward transposes the matrix on
   every call; at 139k x 139k it attempts a 72 GiB allocation. Replaced by
   `FixedSparsityMatmul`, which precomputes the transposed CSR layout once.
2. **`torch.sparse.sampled_addmm` is very slow here** -- the cuSPARSE SDDMM kernel measured
   ~0.016 TFLOPS, 178 ms per call, 67% of total GPU time. Replaced by an explicit
   gather-reduce.
3. **`torch.compile` fuses that gather-reduce into a single kernel**, which both triples its
   speed and removes a 5.5 GiB intermediate. 1077 ms -> 233 ms for the trunk alone.

Correctness is checked against a dense reference (`bench_flybrain.py`, plus a standalone
dense comparison): weight-gradient relative error 1.5e-7.

## How many settle steps?

Not a free parameter. Each settle step is one step of a leaky integrator, so the step size is
physical: `dt = ms_per_action / n_settle_steps`. *Drosophila* central neurons have a membrane
time constant near 16 ms (MBON-alpha3: 16.06 +/- 4.92 ms over 5 cells, Pribbenow et al. 2022,
eLife 77578). With `ms_per_action = 50`:

| steps | dt | dt / tau |
|---|---|---|
| 2 | 25.0 ms | 1.56 -- coarser than the membrane |
| 3 | 16.7 ms | 1.04 |
| **4** | **12.5 ms** | **0.78** |
| 5 | 10.0 ms | 0.63 |

4 is the coarsest step still under one time constant, which is what the discretisation needs.
The per-neuron leak is initialised to `exp(-dt/tau)` and trains from there.

## Getting the data

Two downloads, both open access, no login:

```bash
mkdir -p data/flywire && cd data/flywire

# Connectivity: neuron-pair x neuropil synapse counts (852 MB)
curl -L -o proofread_connections_783.feather \
  "https://zenodo.org/records/10676866/files/proofread_connections_783.feather?download=1"

# Annotations: super_class, neurotransmitter, side, soma position (32 MB)
curl -L -o neuron_annotations.tsv \
  "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv"
```

Then build the training artifact:

```bash
python scripts/tools/flywire/build_connectome.py
```

This aggregates edges over neuropils, drops connections below 5 synapses (the FlyWire/Codex
convention, below which automatic synapse detection has a high false-positive rate), assigns
each edge a sign from its presynaptic neuron's transmitter, and writes
`data/flywire/connectome_783.npz`.

Verify it before spending GPU-days on it:

```bash
python scripts/tools/flywire/bench_flybrain.py
```

## How the fly is wired into the agent

FlyWire's `super_class` annotation partitions the brain, and the partition maps onto an RL
agent almost directly:

| super_class | count | role |
|---|---|---|
| `optic` | 77,541 | optic lobe — receives the conv feature map retinotopically |
| `visual_projection` | 8,038 | carries vision into the central brain |
| `sensory`, `sensory_ascending`, `ascending` | 19,269 | body state, proprioception — receives the float features |
| `central` | 32,383 | the central brain; does the computing |
| `descending` | 1,303 | the brain's output channel to the body — **Q-values are read out here** |
| `visual_centrifugal`, `motor`, `endocrine` | 714 | |

Descending neurons are the fly's actual motor-output pathway, so decoding actions from them
is not a metaphor.

**Retinotopic injection.** The conv head produces a `(32, 11, 16)` feature map, which is a
retinotopic grid. Each optic neuron is assigned a cell in that grid from its soma position
(per-hemisphere SVD onto the two dominant spatial axes, then rank-normalised), and reads
only its own column. This is held as a sparse matrix with 32 nonzeros per neuron — a dense
gather would materialise a `(512, 77541, 32)` tensor, about 5 GB. The conv head therefore
acts as a photoreceptor layer, leaving the real optic-lobe circuitry to do the visual work.

**Dale's law.** In *Drosophila*, glutamate is predominantly *inhibitory* (via GluCl-α),
unlike vertebrate cortex. Signs are therefore acetylcholine `+1`, GABA `−1`, glutamate `−1`.
Aminergic neurons (dopamine, serotonin, octopamine) are neither fast-excitatory nor
fast-inhibitory; they get a small symmetric init and training resolves them. Setting
`fly_dale = True` reparameterises weights as `sign * softplus(raw)` so a neuron can never
flip its sign for the whole of training.

## Two deliberate departures from biology

1. **No state between actions.** Linesight's replay buffer stores independent transitions;
   carrying recurrent state would require sequence sampling and a buffer rewrite. The brain
   instead *settles* for `fly_n_settle_steps` from rest on each sensory snapshot.
2. **The conv head is kept.** It stands in for the photoreceptor layer rather than being
   replaced by the retina.

## Configuration

Set `use_fly_brain = True` in `config_files/config.py`. Related knobs are in the same block.
`jit` is disabled automatically when the brain is on: TorchScript cannot trace sparse COO
construction and the settle loop defeats `torch.compile`'s static shapes.

`fly_init_gain` is the setting to watch. Above ~1.0 the settle loop can diverge; the bench
script prints per-step activity statistics so you can see it happening before training does.

## Citation

The connectome data is CC-BY-4.0 and requires attribution:

- Dorkenwald, S. et al. **Neuronal wiring diagram of an adult brain.** *Nature* 634, 124–138 (2024).
- Schlegel, P. et al. **Whole-brain annotation and multi-connectome cell typing of *Drosophila*.** *Nature* 634, 139–152 (2024).
- Connectivity files: Zenodo, <https://doi.org/10.5281/zenodo.10676866>
- Annotations: <https://github.com/flyconnectome/flywire_annotations>
- Codex data explorer: <https://codex.flywire.ai/>

## Windows environment notes

Two things on this machine were blocked by **Smart App Control** (Windows 11), which refuses
to load unsigned native code:

- **numba** (`ImportError: DLL load failed while importing _dynfunc`). Worked around in code:
  `trackmania_rl/numba_compat.py` falls back to plain Python when numba will not import. The
  three decorated functions are small numeric helpers, so the cost is minor. No security
  setting needs changing for this one.
- **`TMLoader.exe`** ("Potentially unwanted application"). This one has no code workaround --
  TrackMania is installed inside TMLoader's own database and TMInterface injects through it,
  so training cannot launch the game while Smart App Control is enforcing. Check its state
  with:

  ```powershell
  (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy').VerifiedAndReputablePolicyState
  # 0 = off, 1 = enforcing, 2 = evaluation
  ```

  Turning it off is irreversible without reinstalling Windows.

**`torch.compile` does work on Windows here**, via `pip install triton-windows` (triton 3.8.0
against torch 2.11+cu128). The repo currently gates `torch.compile` behind `config.is_linux`;
that gate is worth revisiting for the baseline network too, independently of this work.
