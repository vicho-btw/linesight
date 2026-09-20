"""
A recurrent network whose wiring diagram is the *Drosophila melanogaster* whole-brain
connectome (FlyWire FAFB v783: 139,248 neurons, ~2.7M neuron-to-neuron connections).

The connectome fixes the topology. Synapse counts and neurotransmitter identities set the
initial weights, and the weights then train normally. Nothing about the sparsity pattern
ever changes.

How the fly is wired into the agent
-----------------------------------
The annotation file sorts every neuron into a super_class, and those classes map onto the
parts of a reinforcement-learning agent almost directly:

    conv feature map  ->  optic (77,541)             the optic lobe, injected retinotopically
    float features    ->  sensory/ascending (19,269) body state, proprioception
                          central (32,383)           the central brain does the computing
    Q-value readout   <-  descending (1,303)         the brain's output channel to the body

Descending neurons are the real motor-output pathway of the fly brain, so reading actions
out of them is not a metaphor.

Two deliberate departures from biology
--------------------------------------
1. The brain holds no state between actions. Linesight's replay buffer stores independent
   transitions; carrying recurrent state would require sequence sampling and a rewrite of
   the buffer. Instead the brain *settles* for `n_settle_steps` from rest on each sensory
   snapshot, which is closer to "what fixed point does this stimulus drive" than to a
   continuously-running brain.
2. The convolutional head is kept in front of the optic lobe, acting as a photoreceptor
   layer. It produces a retinotopic (C, H, W) feature map and each optic neuron reads the
   column at its own position, so the real optic-lobe circuitry still does the visual work.

A note on why the sparse matmul is hand-written
-----------------------------------------------
torch.sparse.mm on a COO tensor cannot be used at this scale: its backward transposes the
matrix on every call, and at 139k x 139k it tries to allocate 72 GiB. torch's
sparse_sampled_addmm avoids that but runs the cuSPARSE SDDMM kernel at roughly 0.016 TFLOPS
here, costing ~178 ms per call. FixedSparsityMatmul below precomputes the transposed CSR
layout once and computes the weight gradient as a chunked gather-reduce, whose cost is
bounded by memory bandwidth rather than by cuSPARSE.
"""

import struct
from pathlib import Path

import time

import numpy as np
import torch
import torch.nn.functional as F

OPTIC = "optic"
SENSORY_CLASSES = ("sensory", "sensory_ascending", "ascending")
DESCENDING = "descending"


def _rank_to_grid(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Map values onto 0..n_bins-1 so that each bin holds a roughly equal number of neurons."""
    if len(values) == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return np.minimum((ranks / max(len(values), 1) * n_bins).astype(np.int64), n_bins - 1)


def _retinotopic_grid_index(position: np.ndarray, side_id: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """
    Assign every optic-lobe neuron a cell in a (grid_h, grid_w) retinotopic map.

    The optic lobe is a columnar, retinotopically-ordered structure, so the two dominant
    axes of spatial variation within one lobe approximate the two visual-field axes. We take
    them per-hemisphere with an SVD and rank-normalise, which keeps the mapping data-driven
    rather than dependent on a hand-picked anatomical axis convention.
    """
    grid = np.zeros(len(position), dtype=np.int64)
    for side in np.unique(side_id):
        mask = side_id == side
        if mask.sum() < 3:
            continue
        pts = position[mask]
        pts = pts - pts.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(pts, full_matrices=False)
        proj = pts @ vt[:2].T
        rows = _rank_to_grid(proj[:, 0], grid_h)
        cols = _rank_to_grid(proj[:, 1], grid_w)
        grid[mask] = rows * grid_w + cols
    return grid


def _gather_reduce(grad_y: torch.Tensor, x: torch.Tensor, row_idx: torch.Tensor, col_idx: torch.Tensor) -> torch.Tensor:
    """grad_values[e] = sum_b grad_y[row_e, b] * x[col_e, b]."""
    return (grad_y[row_idx] * x[col_idx]).sum(dim=1, dtype=torch.float32)


_gather_reduce_impl = None


def _get_gather_reduce():
    """
    Return the fastest available implementation of the weight-gradient reduction.

    When torch.compile is usable, inductor fuses the two gathers, the product and the
    reduction into a single kernel. That matters for more than speed: unfused, the
    (n_edges, chunk) intermediate is materialised, and at batch 512 it would be 5.5 GiB.
    Fused, peak memory is just the inputs. Measured here at 2.8M edges: 20 ms fused versus
    79 ms eager, at identical fp32 precision.

    Falls back to eager if triton is unavailable (torch.compile needs it, and it is not
    installed by default on Windows).
    """
    global _gather_reduce_impl
    if _gather_reduce_impl is None:
        try:
            import triton  # noqa: F401

            _gather_reduce_impl = torch.compile(_gather_reduce, dynamic=False)
        except Exception:
            _gather_reduce_impl = _gather_reduce
    return _gather_reduce_impl



# ---------------------------------------------------------------------------- live tap
# Layout of the shared file the live viewer reads. Header is fixed-width so the
# viewer can parse it without knowing anything about this module.
TAP_MAGIC = b"FLYTAP01"
TAP_HEADER = 256
TAP_OFF_FRAME = 8     # <Q  incremented last, so a reader can tell a frame changed
TAP_OFF_N = 16        # <I  neuron count
TAP_OFF_NACT = 20     # <I  action count
TAP_OFF_RMS = 24      # <f
TAP_OFF_MAX = 28      # <f
TAP_OFF_SCALE = 32    # <f  divisor used to map activity into 0..255
TAP_OFF_Q = 36        # 12 x <f
TAP_OFF_ACTION = 84   # <i  argmax action
TAP_OFF_TIME = 88     # <d  wall clock


class _LiveTap:
    """
    Publishes the brain's settled activity into a memory-mapped file so a local viewer can
    watch the network drive in real time.

    Kept deliberately cheap: quantisation to uint8 happens on the GPU, so each published
    frame is one 139 KB transfer instead of 557 KB, and frames are published every `every`
    inferences rather than all of them. Any failure disables the tap rather than disturbing
    a training run.
    """

    def __init__(self, path, n_neurons: int, n_actions: int = 12, every: int = 6, scale: float = 3.0):
        self.path = Path(path)
        self.n = n_neurons
        self.every = max(1, int(every))
        self.scale = float(scale)
        self.counter = 0
        self.frame = 0
        self.ok = False
        # The viewer touches this file every poll. When nobody is watching we skip the
        # device-to-host copy entirely, so an unobserved training run pays nothing.
        self.watch_path = Path(str(path) + ".watch")
        self.watching = False
        self._watch_checked = -1
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            size = TAP_HEADER + n_neurons
            # Reuse an existing file of the right size rather than recreating it. On Windows,
            # "w+" fails with EINVAL if the viewer already has the file mapped, which silently
            # cost us the live view on every restart.
            mode = "r+" if (self.path.exists() and self.path.stat().st_size == size) else "w+"
            self.raw = np.memmap(self.path, dtype=np.uint8, mode=mode, shape=(size,))
            self.raw[:] = 0
            self._put(0, "<8s", TAP_MAGIC)
            self._put(TAP_OFF_N, "<I", n_neurons)
            self._put(TAP_OFF_NACT, "<I", n_actions)
            self._put(TAP_OFF_SCALE, "<f", self.scale)
            self.raw.flush()
            self.ok = True
        except Exception as e:  # pragma: no cover
            print(f"live tap disabled ({e})", flush=True)

    def _put(self, off, fmt, *vals):
        b = struct.pack(fmt, *vals)
        self.raw[off : off + len(b)] = np.frombuffer(b, dtype=np.uint8)

    def publish(self, x_col: "torch.Tensor"):
        """x_col: (n_neurons,) settled activity for a single sample."""
        if not self.ok:
            return
        self.counter += 1

        # Re-check for a viewer a few times a second, not on every inference.
        if self.counter - self._watch_checked >= 32:
            self._watch_checked = self.counter
            try:
                self.watching = (time.time() - self.watch_path.stat().st_mtime) < 4.0
            except OSError:
                self.watching = False

        if not self.watching or self.counter % self.every:
            return
        try:
            # One device-to-host transfer and one sync, nothing else. Reading rms/max off the
            # GPU tensor directly would add two more syncs and cost more than the copy itself
            # (measured 5.6 -> 8.9 ms per inference); they are recovered from the quantised
            # bytes instead, which is ample for a display.
            q = (x_col.abs() / self.scale).clamp_(0.0, 1.0).mul_(255.0).to(torch.uint8)
            arr = q.cpu().numpy()
            self.raw[TAP_HEADER:] = arr
            a = arr.astype(np.float32) * (self.scale / 255.0)
            self._put(TAP_OFF_RMS, "<f", float(np.sqrt((a * a).mean())))
            self._put(TAP_OFF_MAX, "<f", float(a.max()))
            self._put(TAP_OFF_TIME, "<d", time.time())
            self.frame += 1
            self._put(TAP_OFF_FRAME, "<Q", self.frame)  # last: signals a complete frame
        except Exception as e:  # pragma: no cover
            print(f"live tap write failed, disabling ({e})", flush=True)
            self.ok = False

    def publish_q(self, q_values):
        """Q values for the frame just published, written into the header."""
        if not self.ok or not self.watching:
            return
        try:
            vals = [float(v) for v in q_values[:12]]
            vals += [0.0] * (12 - len(vals))
            self._put(TAP_OFF_Q, "<12f", *vals)
            self._put(TAP_OFF_ACTION, "<i", int(max(range(len(q_values)), key=lambda i: q_values[i])))
        except Exception:
            pass


class _FixedSparsityMatmulFn(torch.autograd.Function):
    """y = W @ x, with W's sparsity pattern fixed and its values trainable."""

    @staticmethod
    def forward(ctx, values, x, crow, col, crow_t, col_t, perm_t, row_idx, col_idx, shape, grad_chunk, grad_dtype):
        n_rows, n_cols = shape
        w = torch.sparse_csr_tensor(crow, col, values, (n_rows, n_cols))
        y = w @ x
        ctx.save_for_backward(values, x, crow_t, col_t, perm_t, row_idx, col_idx)
        ctx.shape = shape
        ctx.grad_chunk = grad_chunk
        ctx.grad_dtype = grad_dtype
        return y

    @staticmethod
    def backward(ctx, grad_y):
        values, x, crow_t, col_t, perm_t, row_idx, col_idx = ctx.saved_tensors
        n_rows, n_cols = ctx.shape
        grad_y = grad_y.contiguous()

        grad_values = grad_x = None

        if ctx.needs_input_grad[1]:
            # The transposed layout was precomputed, so this is a plain SpMM with no re-sorting.
            w_t = torch.sparse_csr_tensor(crow_t, col_t, values[perm_t], (n_cols, n_rows))
            grad_x = w_t @ grad_y

        if ctx.needs_input_grad[0]:
            # grad_values[e] = sum_b grad_y[row_e, b] * x[col_e, b]
            #
            # Done in batch-column chunks: the full (n_edges, batch) intermediate would be
            # about 5.5 GiB at 2.8M edges and batch 512. Chunking bounds that while moving the
            # same total traffic, and it beats cuSPARSE's SDDMM substantially here.
            #
            # Chunking the batch is what bounds memory when the kernel is not fused, and it
            # turns out to be faster even when it is: smaller column blocks keep the gathered
            # rows resident in cache. 128 measured fastest at batch 512.
            batch = x.shape[1]
            grad_values = torch.zeros_like(values)
            chunk = max(1, min(ctx.grad_chunk, batch))
            kernel = _get_gather_reduce()
            gy_lo = grad_y if ctx.grad_dtype == torch.float32 else grad_y.to(ctx.grad_dtype)
            x_lo = x if ctx.grad_dtype == torch.float32 else x.to(ctx.grad_dtype)
            for start in range(0, batch, chunk):
                stop = min(start + chunk, batch)
                grad_values += kernel(gy_lo[:, start:stop], x_lo[:, start:stop], row_idx, col_idx)

        return grad_values, grad_x, None, None, None, None, None, None, None, None, None, None


class FixedSparsityMatmul(torch.nn.Module):
    """
    A linear map whose sparsity pattern is fixed at construction and whose values train.

    Args:
        row: postsynaptic / output index of each nonzero
        col: presynaptic / input index of each nonzero
        shape: (n_rows, n_cols)
        init_values: initial value of each nonzero
        grad_chunk: batch-columns processed at a time in the backward gather-reduce
        grad_dtype: precision of the backward gather; the reduction always accumulates in fp32.
            fp32 is the default because the fused kernel is faster than an eager fp16 gather.

    `row` and `col` must already be sorted lexicographically by (row, col), which is what CSR
    requires and what build_connectome.py emits.
    """

    def __init__(
        self,
        row: np.ndarray,
        col: np.ndarray,
        shape,
        init_values: np.ndarray,
        grad_chunk: int = 128,
        grad_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        n_rows, n_cols = int(shape[0]), int(shape[1])
        self.shape = (n_rows, n_cols)
        self.n_edges = len(row)
        self.grad_chunk = grad_chunk
        self.grad_dtype = grad_dtype

        row = row.astype(np.int64)
        col = col.astype(np.int64)

        crow = np.zeros(n_rows + 1, dtype=np.int64)
        np.cumsum(np.bincount(row, minlength=n_rows), out=crow[1:])

        order_t = np.lexsort((row, col))  # sort by (col, row) -> row-major CSR of the transpose
        crow_t = np.zeros(n_cols + 1, dtype=np.int64)
        np.cumsum(np.bincount(col, minlength=n_cols), out=crow_t[1:])

        self.register_buffer("crow", torch.from_numpy(crow), persistent=False)
        self.register_buffer("col", torch.from_numpy(col.copy()), persistent=False)
        self.register_buffer("crow_t", torch.from_numpy(crow_t), persistent=False)
        self.register_buffer("col_t", torch.from_numpy(row[order_t].copy()), persistent=False)
        self.register_buffer("perm_t", torch.from_numpy(order_t.copy()), persistent=False)
        self.register_buffer("row_idx", torch.from_numpy(row.copy()), persistent=False)
        self.register_buffer("col_idx", torch.from_numpy(col.copy()), persistent=False)

        self.values = torch.nn.Parameter(torch.from_numpy(init_values.astype(np.float32)))

    def forward(self, x: torch.Tensor, values: torch.Tensor = None) -> torch.Tensor:
        """x: (n_cols, batch) -> (n_rows, batch). `values` overrides the stored parameter."""
        return _FixedSparsityMatmulFn.apply(
            self.values if values is None else values,
            x,
            self.crow,
            self.col,
            self.crow_t,
            self.col_t,
            self.perm_t,
            self.row_idx,
            self.col_idx,
            self.shape,
            self.grad_chunk,
            self.grad_dtype,
        )

    def extra_repr(self) -> str:
        return f"shape={self.shape}, n_edges={self.n_edges}"


class FlyBrain(torch.nn.Module):
    """
    Sparse recurrent module wired from the FlyWire connectome.

    Args:
        connectome_path: path to the .npz produced by scripts/tools/flywire/build_connectome.py
        visual_channels/visual_h/visual_w: shape of the conv head's feature map before flattening
        float_dim: width of the float feature extractor's output
        readout_dim: width of the embedding handed back to the IQN heads
        n_settle_steps: number of recurrent updates per action
        sensory_rank: rank of the low-rank projection from float features onto sensory neurons
        init_gain: scales the initial recurrent weights; controls how strongly the brain recurs
        dale: if True, weights are reparameterised so a neuron can never flip its sign
        step_norm: normalise activity after each settle step to keep the loop from diverging
        grad_chunk: batch-columns per chunk in the sparse backward; trades memory for speed
        dt_ms: simulated time per settle step, i.e. ms_per_action / n_settle_steps
        membrane_tau_ms: membrane time constant of a fly neuron, used to initialise the leak
    """

    def __init__(
        self,
        connectome_path: Path,
        visual_channels: int,
        visual_h: int,
        visual_w: int,
        float_dim: int,
        readout_dim: int,
        n_settle_steps: int = 4,
        sensory_rank: int = 64,
        init_gain: float = 0.9,
        dale: bool = False,
        step_norm: bool = True,
        grad_chunk: int = 128,
        dt_ms: float = 12.5,
        membrane_tau_ms: float = 16.0,
    ):
        super().__init__()
        data = np.load(connectome_path, allow_pickle=False)

        super_classes = [str(s) for s in data["super_classes"]]
        super_class_id = data["super_class_id"]
        n_neurons = len(super_class_id)

        self.n_neurons = n_neurons
        self.n_settle_steps = n_settle_steps
        self.readout_dim = readout_dim
        self.dale = dale
        self.step_norm = step_norm
        self.visual_channels = visual_channels
        self.visual_h = visual_h
        self.visual_w = visual_w

        def class_mask(*names):
            ids = [super_classes.index(n) for n in names if n in super_classes]
            return np.isin(super_class_id, ids)

        optic_idx = np.flatnonzero(class_mask(OPTIC)).astype(np.int64)
        sensory_idx = np.flatnonzero(class_mask(*SENSORY_CLASSES)).astype(np.int64)
        descending_idx = np.flatnonzero(class_mask(DESCENDING)).astype(np.int64)

        if len(descending_idx) == 0:
            raise ValueError("connectome has no descending neurons; cannot read out actions")

        self.register_buffer("optic_idx", torch.from_numpy(optic_idx), persistent=False)
        self.register_buffer("sensory_idx", torch.from_numpy(sensory_idx), persistent=False)
        self.register_buffer("descending_idx", torch.from_numpy(descending_idx), persistent=False)

        # ------------------------------------------------------------ recurrent connectivity
        src = data["edge_src"].astype(np.int64)
        dst = data["edge_dst"].astype(np.int64)
        syn = data["edge_syn"].astype(np.float64)
        sign = data["edge_sign"].astype(np.float64)
        self.n_edges = len(src)

        # Initial magnitude follows synapse count, normalised per postsynaptic neuron so every
        # neuron receives input of comparable scale regardless of how many partners it has.
        w0 = sign * syn
        row_sq = np.zeros(n_neurons, dtype=np.float64)
        np.add.at(row_sq, dst, syn**2)
        row_norm = np.sqrt(np.maximum(row_sq, 1e-12))
        w0 = w0 / row_norm[dst] * init_gain

        # Aminergic neurons (sign 0) are neither fast-excitatory nor fast-inhibitory. Give them a
        # small symmetric init and let training decide.
        modulatory = sign == 0
        if modulatory.any():
            rng = np.random.default_rng(0)
            w0[modulatory] = (
                rng.normal(0.0, init_gain * 0.1, size=int(modulatory.sum()))
                * np.sqrt(syn[modulatory])
                / row_norm[dst[modulatory]]
            )

        if dale:
            # Reparameterise as sign * softplus(raw): a neuron's sign becomes structural and
            # cannot flip for the whole of training.
            w0 = np.log(np.expm1(np.clip(np.abs(w0), 1e-6, None)))

        # row = postsynaptic, col = presynaptic, for x_post = W @ x_pre
        self.recurrent = FixedSparsityMatmul(dst, src, (n_neurons, n_neurons), w0, grad_chunk=grad_chunk)
        self.register_buffer("edge_sign", torch.from_numpy(sign.astype(np.float32)), persistent=False)

        self.neuron_bias = torch.nn.Parameter(torch.zeros(n_neurons))

        # Each settle step is a discrete step of a leaky integrator, dv/dt = -v/tau + input, so
        # the leak is exp(-dt/tau) rather than an arbitrary constant. Drosophila central neurons
        # have a membrane time constant around 16 ms (MBON-alpha3, 16.06 +/- 4.92 ms over 5 cells,
        # Pribbenow et al. 2022, eLife 77578). With ms_per_action = 50 and 4 settle steps,
        # dt = 12.5 ms sits just under one time constant, which is what a leaky integrator needs
        # to be a faithful discretisation. The value is learnable from there.
        self.dt_ms = dt_ms
        self.membrane_tau_ms = membrane_tau_ms
        leak0 = float(np.exp(-dt_ms / max(membrane_tau_ms, 1e-6)))
        leak0 = min(max(leak0, 1e-4), 1 - 1e-4)
        self.neuron_leak = torch.nn.Parameter(torch.full((n_neurons,), float(np.log(leak0 / (1 - leak0)))))
        self.step_gain = torch.nn.Parameter(torch.ones(1))
        # Scale of the sensory drive relative to the recurrent state (see forward()).
        self.drive_gain = torch.nn.Parameter(torch.ones(1))

        # ------------------------------------------------------------ optic lobe injection
        # Each optic neuron reads the conv feature column at its own retinotopic position.
        # Held as a sparse (n_optic x visual_dim) matrix with `visual_channels` nonzeros per row,
        # so we never materialise a (batch, n_optic, channels) tensor.
        position = data["position"]
        side_id = data["side_id"]
        grid = _retinotopic_grid_index(position[optic_idx], side_id[optic_idx], visual_h, visual_w)
        n_optic = len(optic_idx)
        visual_dim = visual_channels * visual_h * visual_w

        rows = np.repeat(np.arange(n_optic, dtype=np.int64), visual_channels)
        channels = np.tile(np.arange(visual_channels, dtype=np.int64), n_optic)
        cols = channels * (visual_h * visual_w) + np.repeat(grid, visual_channels)
        # CSR requires column indices ascending within each row.
        order = np.lexsort((cols, rows))
        rows, cols = rows[order], cols[order]

        optic_rng = np.random.default_rng(1)
        optic_init = optic_rng.normal(0.0, 1.0 / np.sqrt(visual_channels), size=len(rows))
        self.optic_in = FixedSparsityMatmul(rows, cols, (n_optic, visual_dim), optic_init, grad_chunk=grad_chunk)
        self.n_optic = n_optic
        self.visual_dim = visual_dim

        # ------------------------------------------------------------ body-state injection
        self.sensory_proj = torch.nn.Sequential(
            torch.nn.Linear(float_dim, sensory_rank),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(sensory_rank, len(sensory_idx)),
        )

        # ------------------------------------------------------------ motor readout
        # Normalise descending activity before decoding. The settle loop erodes the
        # input-dependent part of the signal (varying fraction 0.48 -> 0.19 over four steps)
        # and what survives arrives at a very small absolute scale, so the dueling head
        # produced action advantages ~7x narrower than the dense trunk's. Standardising here
        # restores the surviving signal to unit scale without inventing information.
        self.descending_norm = torch.nn.LayerNorm(len(descending_idx))
        self.readout = torch.nn.Linear(len(descending_idx), readout_dim)

        self.live_tap = None  # set via enable_live_tap(); see scripts/tools/flywire/live_viewer.py

        self._init_interface_weights()

    def _init_interface_weights(self):
        for m in [self.sensory_proj[0], self.sensory_proj[2], self.readout]:
            torch.nn.init.orthogonal_(m.weight, gain=1.0)
            torch.nn.init.zeros_(m.bias)

    def enable_live_tap(self, path, n_actions: int = 12, every: int = 6, scale: float = 3.0):
        """Publish settled activity to `path` so the live viewer can render it."""
        self.live_tap = _LiveTap(path, self.n_neurons, n_actions=n_actions, every=every, scale=scale)
        if self.live_tap.ok:
            print(f"live tap publishing to {path} (every {every} inferences)", flush=True)

    def recurrent_values(self) -> torch.Tensor:
        if self.dale:
            return self.edge_sign * F.softplus(self.recurrent.values)
        return self.recurrent.values

    def forward(self, visual_features: torch.Tensor, float_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: (batch, visual_channels * visual_h * visual_w) flattened conv feature map
            float_features:  (batch, float_dim)

        Returns:
            (batch, readout_dim) embedding decoded from descending-neuron activity
        """
        batch_size = visual_features.shape[0]

        # The connectome runs in fp32, outside autocast. Sparse CSR kernels will not mix an fp32
        # value buffer with an fp16 dense operand, the submodules here would otherwise return
        # fp16 into fp32 accumulators, and a recurrent loop is where reduced precision hurts most.
        # The caller's autocast region resumes on the returned tensor.
        with torch.amp.autocast(device_type=visual_features.device.type, enabled=False):
            return self._forward_fp32(visual_features.float(), float_features.float(), batch_size)

    def _forward_fp32(self, visual_features: torch.Tensor, float_features: torch.Tensor, batch_size: int) -> torch.Tensor:
        # Sensory drive is constant across settle steps: the stimulus is held fixed while the
        # brain relaxes towards its response.
        optic_drive = self.optic_in(visual_features.t().contiguous())  # (n_optic, batch)
        sensory_drive = self.sensory_proj(float_features).t()  # (n_sensory, batch)

        drive = visual_features.new_zeros((self.n_neurons, batch_size))
        drive.index_copy_(0, self.optic_idx, optic_drive)
        drive.index_copy_(0, self.sensory_idx, sensory_drive)

        # Normalise the sensory drive to the same scale as the activity it is injected into.
        # step_norm pins the recurrent state at rms 1 every step, but the drive arrives at
        # whatever scale the input heads happen to produce -- measured rms 0.013 against a
        # recurrent term of 0.9-3.0, i.e. 70-230x weaker. The network then settles into its
        # own attractor and ignores the camera: the readout's variation across different
        # inputs decayed 0.014 -> 0.0045 over four steps, and the policy chose one action
        # regardless of state. Scaling the drive lets the stimulus actually steer the brain.
        drive = drive * torch.rsqrt(drive.pow(2).mean(dim=0, keepdim=True) + 1e-8) * self.drive_gain
        drive = drive + self.neuron_bias.unsqueeze(1)

        values = self.recurrent_values()

        # Proper Euler step of the leaky-integrator ODE  tau dx/dt = -x + f(Wx + I):
        #
        #     x <- (1 - dt/tau) x + (dt/tau) f(Wx + I)
        #
        # The previous form, x <- act(leak*x + Wx + drive) followed by renormalisation, let
        # the recurrent term *replace* the state every step instead of nudging it, and the
        # renormalisation stopped it ever contracting. The result was chaotic rather than
        # settling: successive steps still differed by ~0.12 after eight iterations, and a 2%
        # input perturbation flipped the chosen action 62% of the time (feedforward baseline:
        # 0%). That is why a good lap was followed by a garbage one.
        #
        # This form is a contraction whenever dt/tau < 1, which is exactly the condition that
        # made 4 settle steps the biologically right choice in the first place.
        alpha = min(self.dt_ms / max(self.membrane_tau_ms, 1e-6), 0.95)
        x = torch.zeros_like(drive)
        for _ in range(self.n_settle_steps):
            recurrent = self.recurrent(x, values=values)
            target = F.leaky_relu(recurrent + drive, negative_slope=0.01)
            if self.step_norm:
                # Normalise the *target*, not the state, so the update stays a contraction.
                target = target * torch.rsqrt(target.pow(2).mean(dim=0, keepdim=True) + 1e-6) * self.step_gain
            x = (1.0 - alpha) * x + alpha * target

        # Only tapped for single-sample inference: that is the network actually driving the
        # car, whereas a training batch is 512 unrelated replayed states.
        if self.live_tap is not None and batch_size == 1:
            self.live_tap.publish(x[:, 0].detach())

        descending = x.index_select(0, self.descending_idx).t()  # (batch, n_descending)
        return self.readout(self.descending_norm(descending))

    def extra_repr(self) -> str:
        return (
            f"n_neurons={self.n_neurons}, n_edges={self.n_edges}, settle_steps={self.n_settle_steps}, "
            f"optic={self.n_optic}, sensory={len(self.sensory_idx)}, "
            f"descending={len(self.descending_idx)}, dale={self.dale}, "
            f"dt={self.dt_ms}ms, tau={self.membrane_tau_ms}ms"
        )
