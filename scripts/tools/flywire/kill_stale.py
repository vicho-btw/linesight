"""
Kill everything left over from a previous run, before starting a new one.

Orphaned collector workers have bitten this project three times. The trap is that a
multiprocessing child's command line is just

    python -c "from multiprocessing.spawn import spawn_main; spawn_main(parent_pid=..., ...)"

with no mention of the project anywhere in it. Filtering processes on the project path -- the
obvious thing to do -- reports "0 processes running" while orphaned workers are still holding
GPU memory and relaunching TmForever every 40 seconds. That interference was silently present
underneath a set of benchmark runs whose numbers were then reported as clean.

So orphans are identified by reading parent_pid out of the command line and checking whether
that parent still exists, not by matching the project name.

    python scripts/tools/flywire/kill_stale.py          # kill and report
    python scripts/tools/flywire/kill_stale.py --check  # report only
"""

import argparse
import os
import re
import subprocess
import sys
import time

try:
    import psutil
except ImportError:
    print("psutil is required: pip install psutil")
    sys.exit(2)

PROJECT_HINT = "linesight"
GAME_NAMES = {"TmForever.exe", "TMLoader.exe", "TMInterface.exe"}
KEEP_HINTS = ("live_viewer",)  # the viewer is harmless and worth keeping alive


def classify():
    """Returns (project_procs, orphan_workers, game_procs)."""
    alive = {p.pid for p in psutil.process_iter()}
    # This script lives under the project directory, so its own interpreter path contains the
    # project name and it matches its own filter. Without this it kills itself mid-sweep.
    me = {os.getpid(), os.getppid()}
    project, orphans, games = [], [], []

    for p in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            name = p.info["name"] or ""
            cmd = " ".join(p.info["cmdline"] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

        if p.info['pid'] in me:
            continue
        if name in GAME_NAMES:
            games.append((p, name))
            continue
        if not name.lower().startswith("python"):
            continue
        if any(h in cmd for h in KEEP_HINTS):
            continue

        if "multiprocessing-fork" in cmd or "spawn_main" in cmd:
            m = re.search(r"parent_pid=(\d+)", cmd)
            # No parent, or a parent that no longer exists, means nothing will ever reap it.
            if m and int(m.group(1)) not in alive:
                orphans.append((p, f"orphan worker (parent {m.group(1)} gone)"))
        elif PROJECT_HINT in cmd:
            project.append((p, "project process"))

    return project, orphans, games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report without killing")
    args = ap.parse_args()

    # A dying collector can launch a fresh game, or spawn a worker, AFTER the first sweep has
    # passed over it -- so one pass genuinely is not enough and the second pass regularly finds
    # an orphan the first could not have seen. Sweep until a pass finds nothing.
    killed_any = False
    for sweep in range(4):
        project, orphans, games = classify()
        targets = project + orphans + games
        if not targets:
            if sweep == 0:
                print("nothing stale: no project processes, no orphaned workers, no game instances")
            break
        for p, why in targets:
            try:
                print(f"  {'found' if args.check else 'killing'} pid {p.pid:6d}  {p.name():16s}  {why}")
                if not args.check:
                    p.kill()
                    killed_any = True
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print(f"  pid {p.pid}: {type(e).__name__}")
        if args.check:
            break
        time.sleep(3)

    if args.check or not killed_any:
        return 0

    # The game must fully exit before another can initialise Direct3D; launching too soon gives
    # "DirectX 9 initialisation failed", which then looks like a hardware fault rather than a
    # race against our own teardown.
    print("  waiting for Direct3D and TMInterface sockets to be released...")
    for _ in range(30):
        time.sleep(1)
        if not any(p.name() in GAME_NAMES for p in psutil.process_iter() if _safe_name(p)):
            break
    time.sleep(10)

    project, orphans, games = classify()
    left = len(project) + len(orphans) + len(games)
    print(f"  clean: {left} stale processes remaining" if left == 0 else f"  WARNING: {left} still present")
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        print(f"  GPU memory in use: {out}")
    except Exception:
        pass
    return 0 if left == 0 else 1


def _safe_name(p):
    try:
        return p.name()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


if __name__ == "__main__":
    sys.exit(main())
