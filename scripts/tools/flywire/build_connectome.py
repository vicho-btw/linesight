"""
Build a compact, training-ready artifact from the FlyWire FAFB v783 whole-brain connectome.

Inputs (downloaded once into data/flywire/, see README_flywire.md):
    - proofread_connections_783.feather : neuron-pair x neuropil synapse counts (Zenodo 10676866, CC-BY-4.0)
    - neuron_annotations.tsv            : v783 annotations (flyconnectome/flywire_annotations)

Output:
    - data/flywire/connectome_783.npz

The output holds the connectivity *topology* only. Synapse counts and neurotransmitter
identities are carried through so the network can be initialised from them, but the
weights themselves are trained (see trackmania_rl/agents/flybrain.py).

Edges are aggregated over neuropils, so a pair of neurons that touch in several brain
regions becomes a single connection carrying the summed synapse count.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather

# Fast (ionotropic) transmitters determine a neuron's sign under Dale's law.
# In Drosophila, glutamate is predominantly *inhibitory* (GluCl-alpha), unlike vertebrate cortex.
NT_SIGN = {
    "acetylcholine": +1,
    "glutamate": -1,
    "gaba": -1,
    # Aminergic / modulatory transmitters are not fast excitatory or inhibitory drivers.
    # They are given a zero-mean init and left for training to resolve.
    "dopamine": 0,
    "serotonin": 0,
    "octopamine": 0,
}

# Ordering is stable and is what the model indexes groups by.
SUPER_CLASSES = [
    "optic",
    "visual_projection",
    "visual_centrifugal",
    "sensory",
    "sensory_ascending",
    "ascending",
    "central",
    "descending",
    "motor",
    "endocrine",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[3] / "data" / "flywire")
    parser.add_argument(
        "--min-syn-count",
        type=int,
        default=5,
        help="Drop connections with fewer synapses than this. 5 is the FlyWire/Codex convention: "
        "below it the false-positive rate on automatic synapse detection is high.",
    )
    args = parser.parse_args()

    conn_path = args.data_dir / "proofread_connections_783.feather"
    anno_path = args.data_dir / "neuron_annotations.tsv"
    out_path = args.data_dir / "connectome_783.npz"

    for p in (conn_path, anno_path):
        if not p.exists():
            raise SystemExit(f"missing input: {p}\nSee scripts/tools/flywire/README_flywire.md for download instructions.")

    # ------------------------------------------------------------------ annotations
    print(f"reading {anno_path.name} ...")
    anno = pd.read_csv(anno_path, sep="\t", low_memory=False, usecols=["root_id", "super_class", "top_nt", "side", "cell_class", "pos_x", "pos_y", "pos_z", "soma_x", "soma_y", "soma_z"],
    )
    anno = anno.dropna(subset=["root_id"]).drop_duplicates(subset=["root_id"])
    anno["root_id"] = anno["root_id"].astype(np.int64)
    print(f"  {len(anno):,} annotated neurons")

    # Dense index 0..N-1, ordered by root_id so the mapping is reproducible.
    anno = anno.sort_values("root_id").reset_index(drop=True)
    root_ids = anno["root_id"].to_numpy(dtype=np.int64)
    index_of = pd.Series(np.arange(len(root_ids), dtype=np.int64), index=root_ids)
    n_neurons = len(root_ids)

    super_class_id = anno["super_class"].map({name: i for i, name in enumerate(SUPER_CLASSES)}).fillna(-1).to_numpy(np.int8)
    neuron_sign = anno["top_nt"].map(NT_SIGN).fillna(0).to_numpy(np.int8)
    side_id = anno["side"].map({"left": 0, "right": 1, "center": 2}).fillna(-1).to_numpy(np.int8)

    # Spatial position, used to give optic-lobe neurons a retinotopic coordinate.
    # Prefer the soma; fall back to the generic annotation point for neurons whose soma was not located.
    pos = anno[["soma_x", "soma_y", "soma_z"]].to_numpy(np.float64)
    fallback = anno[["pos_x", "pos_y", "pos_z"]].to_numpy(np.float64)
    missing_soma = ~np.isfinite(pos).all(axis=1)
    pos[missing_soma] = fallback[missing_soma]
    print(f"  {int(missing_soma.sum()):,} neurons fell back to pos_* (no soma located)")
    print(f"  {int((~np.isfinite(pos).all(axis=1)).sum()):,} neurons still have no position")
    position = np.nan_to_num(pos, nan=0.0).astype(np.float32)

    unmapped = int((super_class_id < 0).sum())
    if unmapped:
        print(f"  warning: {unmapped:,} neurons have an unrecognised super_class -> id -1")

    # ------------------------------------------------------------------ connections
    print(f"reading {conn_path.name} (this is the big one) ...")
    table = feather.read_table(conn_path)
    print(f"  columns: {table.column_names}")

    def pick(*candidates):
        for c in candidates:
            if c in table.column_names:
                return c
        raise SystemExit(f"none of {candidates} found in {table.column_names}")

    pre_col = pick("pre_pt_root_id", "pre_root_id", "pre_id")
    post_col = pick("post_pt_root_id", "post_root_id", "post_id")
    syn_col = pick("syn_count", "n_syn", "count")

    conn = table.select([pre_col, post_col, syn_col]).to_pandas()
    del table
    conn.columns = ["pre", "post", "syn"]
    print(f"  {len(conn):,} (pair x neuropil) rows")

    # Collapse neuropils: one edge per ordered neuron pair.
    conn = conn.groupby(["pre", "post"], sort=False, as_index=False)["syn"].sum()
    print(f"  {len(conn):,} unique neuron->neuron pairs")

    conn = conn[conn["syn"] >= args.min_syn_count]
    print(f"  {len(conn):,} pairs with >= {args.min_syn_count} synapses")

    # Keep only edges whose both endpoints are annotated neurons.
    known = index_of.index
    conn = conn[conn["pre"].isin(known) & conn["post"].isin(known)]
    print(f"  {len(conn):,} pairs with both endpoints annotated")

    src = index_of.loc[conn["pre"].to_numpy()].to_numpy(np.int32)
    dst = index_of.loc[conn["post"].to_numpy()].to_numpy(np.int32)
    syn = conn["syn"].to_numpy(np.int32)

    # Sort by (dst, src): gives CSR-by-row-of-postsynaptic-neuron, which is the layout
    # the sparse matmul wants for  x_post = W @ x_pre.
    order = np.lexsort((src, dst))
    src, dst, syn = src[order], dst[order], syn[order]

    edge_sign = neuron_sign[src].astype(np.int8)  # Dale's law: sign is a property of the *presynaptic* neuron

    print()
    print(f"  neurons : {n_neurons:,}")
    print(f"  edges   : {len(src):,}")
    print(f"  synapses: {syn.sum():,}")
    print(f"  density : {len(src) / n_neurons**2:.2e}")
    print(f"  mean in-degree : {len(src) / n_neurons:.1f}")
    print(f"  excitatory edges: {(edge_sign > 0).sum():,}  inhibitory: {(edge_sign < 0).sum():,}  modulatory: {(edge_sign == 0).sum():,}")

    np.savez_compressed(
        out_path,
        root_ids=root_ids,
        edge_src=src,
        edge_dst=dst,
        edge_syn=syn,
        edge_sign=edge_sign,
        super_class_id=super_class_id,
        neuron_sign=neuron_sign,
        side_id=side_id,
        position=position,
        super_classes=np.array(SUPER_CLASSES),
        min_syn_count=np.int32(args.min_syn_count),
    )
    print(f"\nwrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
