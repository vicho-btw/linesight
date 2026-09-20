"""
Render a captured lap as video: what the agent sees, beside what its connectome is doing.

Both panels are the real thing rather than illustrations. The left is the exact 120x160 grayscale
frame handed to the network -- that is the entirety of the agent's vision, no colour and no extra
resolution. The right plots all 139,248 neurons at their true FlyWire coordinates, projected
frontally (x by y, the view that separates the two optic lobes from the central brain), with
brightness driven by each neuron's settled activity on that step.

Neurons are tinted by super-class because the classes are functionally distinct and it makes the
flow legible: the optic lobes take the image, the central brain sits between, and the 1,303
descending neurons are the ones whose activity becomes the Q values that pick an action.

    python scripts/tools/flywire/render_lap_video.py --capture data/lap_capture/lap.npz --out lap.mp4
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

W, H = 1600, 900
BG = np.array([10, 12, 16], dtype=np.float32)

# BGR tints per super-class (cv2 order), chosen so the functional path reads left-to-right.
CLASS_TINT = {
    0: (235, 170, 90),    # optic              - cool blue
    1: (225, 200, 120),   # visual_projection  - paler blue
    2: (200, 190, 140),   # visual_centrifugal
    3: (150, 220, 140),   # sensory            - green
    4: (150, 220, 140),   # sensory_ascending
    5: (160, 215, 170),   # ascending
    6: (190, 200, 225),   # central            - warm grey
    7: (90, 110, 255),    # descending         - red: these drive the action
    8: (110, 150, 255),   # motor
    9: (170, 170, 200),   # endocrine
}
HILITE = {7}  # classes drawn last and larger, so they are never buried


def put(img, text, org, scale=0.6, color=(215, 220, 230), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, color, thick, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default=str(ROOT / "data" / "lap_capture" / "lap.npz"))
    ap.add_argument("--connectome", default=str(ROOT / "data" / "flywire" / "connectome_783.npz"))
    ap.add_argument("--out", default=str(ROOT / "data" / "lap_capture" / "flybrain_lap.mp4"))
    ap.add_argument("--fps", type=float, default=None, help="default: real time for the action rate")
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    cap = np.load(args.capture)
    activity, frames = cap["activity"], cap["frames"]
    speed, actions, zones = cap["speed"], cap["actions"], cap["zones"]
    race_time = float(cap["race_time"])
    ms = int(cap["ms_per_action"])
    n = len(activity) if args.max_frames is None else min(args.max_frames, len(activity))
    fps = args.fps or (1000.0 / ms)
    print(f"{n} steps, {race_time:.3f}s lap, {ms} ms/action -> {fps:.0f} fps (real time)")

    con = np.load(args.connectome, allow_pickle=True)
    pos, sc = con["position"], con["super_class_id"]
    assert len(pos) == activity.shape[1], f"{len(pos)} neurons vs {activity.shape[1]} recorded"

    # --- frontal projection, laid out in the right-hand panel ------------------------------
    PX, PY, PW, PH = 760, 70, 800, 760
    x, y = pos[:, 0].astype(np.float32), pos[:, 1].astype(np.float32)
    sx = (x - x.min()) / max(x.ptp(), 1)
    sy = (y - y.min()) / max(y.ptp(), 1)
    keep_ar = min(PW / max(x.ptp(), 1), PH / max(y.ptp(), 1))
    ow = x.ptp() * keep_ar
    oh = y.ptp() * keep_ar
    px = (PX + (PW - ow) / 2 + sx * ow).astype(np.int32)
    py = (PY + (PH - oh) / 2 + sy * oh).astype(np.int32)  # FlyWire y increases ventrally
    px = np.clip(px, 0, W - 1)
    py = np.clip(py, 0, H - 1)

    tint = np.array([CLASS_TINT.get(int(c), (180, 180, 180)) for c in sc], dtype=np.float32)
    order = np.argsort([1 if int(c) in HILITE else 0 for c in sc], kind="stable")  # highlights last
    px, py, tint, sc_o = px[order], py[order], tint[order], sc[order]
    act_order = order
    is_hi = np.isin(sc_o, list(HILITE))

    from config_files import config_copy

    names = ["A--", "A-L", "A-R", "---", "--L", "--R", "-B-", "-BL", "-BR", "AB-", "ABL", "ABR"]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, fps, (W, H))
    if not out.isOpened():
        raise SystemExit(f"could not open {args.out} for writing")

    zmin, zmax = int(zones[:n].min()), max(int(zones[:n].max()), 1)

    for i in range(n):
        img = np.empty((H, W, 3), dtype=np.float32)
        img[:] = BG

        # ---- left: the agent's actual vision ----------------------------------------------
        f = frames[i]
        f = f[0] if f.ndim == 3 else f
        view = cv2.resize(f, (660, 495), interpolation=cv2.INTER_NEAREST)
        view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR).astype(np.float32)
        img[150:645, 50:710] = view
        cv2.rectangle(img, (50, 150), (710, 645), (60, 66, 78), 1, cv2.LINE_AA)

        # ---- right: the connectome --------------------------------------------------------
        a = activity[i][act_order].astype(np.float32)
        # The settled state is very sparse: mean |activity| is ~1.5 of 127 and only ~1% of
        # neurons exceed a tenth of the per-frame maximum. Mapped linearly the panel renders
        # essentially black, so a power curve lifts the quiet majority into view while leaving
        # the strongly driven cells saturated.
        inten = (np.abs(a - 128.0) / 127.0) ** 0.35
        np.add.at(img, (py, px), tint * inten[:, None] * 0.52)
        hp, hq, hi = py[is_hi], px[is_hi], inten[is_hi]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                np.add.at(img, (np.clip(hp + dy, 0, H - 1), np.clip(hq + dx, 0, W - 1)),
                          tint[is_hi] * hi[:, None] * 0.38)

        img = np.clip(img, 0, 255)
        frame = img.astype(np.uint8)

        # ---- overlays ---------------------------------------------------------------------
        put(frame, "FLY BRAIN drives ESL-Hockolicious", (50, 60), 0.95, (235, 240, 248))
        put(frame, f"{activity.shape[1]:,} neurons   /   2,700,429 synapses   /   FlyWire FAFB v783",
            (50, 92), 0.52, (130, 140, 155))

        put(frame, "WHAT IT SEES", (50, 132), 0.5, (130, 140, 155))
        put(frame, f"{f.shape[1]}x{f.shape[0]} grayscale", (560, 132), 0.45, (100, 110, 125))

        t = i * ms / 1000.0
        put(frame, f"{t:5.2f}s", (50, 706), 1.5, (235, 240, 248), 2)
        put(frame, f"of {race_time:.3f}s lap", (215, 706), 0.5, (130, 140, 155))
        put(frame, f"{speed[i]:5.0f} km/h", (50, 762), 1.0, (235, 240, 248), 2)

        act = int(actions[i]) if i < len(actions) else 0
        put(frame, f"input   {names[act] if act < len(names) else act}", (50, 810), 0.6, (150, 200, 245))

        # progress bar along the lap
        prog = (zones[i] - zmin) / max(zmax - zmin, 1)
        cv2.rectangle(frame, (50, 840), (710, 852), (32, 36, 44), -1)
        cv2.rectangle(frame, (50, 840), (50 + int(660 * np.clip(prog, 0, 1)), 852), (90, 110, 255), -1)

        put(frame, "optic", (PX, 858), 0.45, CLASS_TINT[0])
        put(frame, "central", (PX + 110, 858), 0.45, CLASS_TINT[6])
        put(frame, "descending -> action", (PX + 250, 858), 0.45, CLASS_TINT[7])

        out.write(frame)
        if i % 100 == 0:
            print(f"  {i}/{n}", flush=True)

    out.release()
    mb = Path(args.out).stat().st_size / 2**20
    print(f"wrote {args.out} ({mb:.1f} MiB, {n/fps:.1f}s at {fps:.0f} fps)")


if __name__ == "__main__":
    main()
