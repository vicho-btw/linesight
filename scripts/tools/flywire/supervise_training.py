"""
Keep a training run alive unattended.

There are two distinct ways a Linesight run stops making progress, and they need different
detection:

  crash  - train.py exits. Easy: notice the exit code and start it again.
  hang   - train.py keeps running but stops doing anything. This is the common one: when
           TMInterface stops answering, the client retries forever, printing
           "Connection to TMInterface unsuccessful" without ever raising or exiting. A
           supervisor that only watches for process exit will sit there for hours.

So liveness here is measured by *progress*, not by the process being alive. Tensorboard
scalars are written once per completed rollout, so a stale tensorboard directory means no
rollouts are completing, whatever the process thinks it is doing.

    python scripts/tools/flywire/supervise_training.py
    python scripts/tools/flywire/supervise_training.py --stall-minutes 8
"""

import argparse
import ctypes
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GAME_PROCS = ["TmForever", "TMInterface"]



# Windows idle handling. A blank screensaver or a display sleep is enough to knock a
# fullscreen DirectX game out of its render loop, which stalls TMInterface and halts a run
# that may have been collecting for hours. Rather than depend on the machine's power
# settings staying the way we left them, declare the run busy for as long as it is running;
# Windows then suppresses idle sleep and the screensaver by itself, and restores normal
# behaviour the moment this process exits.
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


def keep_awake(on: bool):
    if not hasattr(ctypes, "windll"):
        return False
    try:
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED if on else 0)
        return ctypes.windll.kernel32.SetThreadExecutionState(flags) != 0
    except Exception:
        return False


def log(msg, logfile: Path):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        with logfile.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def newest_progress(tb_dir: Path):
    """Most recent write to the run's tensorboard files; None if there are none yet."""
    try:
        files = list(tb_dir.glob("events.out.*"))
        return max(f.stat().st_mtime for f in files) if files else None
    except OSError:
        return None


def kill_tree(pid: int):
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                   capture_output=True, text=True)




def directx_dialog_present() -> bool:
    """
    True if TrackMania is showing its "DirectX Initialisation failed" dialog.

    CreateDevice returns D3DERR_DEVICELOST when another process already owns the Direct3D
    device -- most often a previous game instance that outlived its training run. The game
    then sits on a modal dialog forever: it is running, it holds the TMInterface port, and
    it answers nothing. Without this check that costs a full stall timeout every restart,
    with nothing in the logs to say why.
    """
    import ctypes
    from ctypes import wintypes

    if not hasattr(ctypes, "windll"):
        return False
    found = []
    EnumProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def cb(hwnd, _):
        if not ctypes.windll.user32.IsWindowVisible(hwnd):
            return True
        cls = ctypes.create_unicode_buffer(256)
        ctypes.windll.user32.GetClassNameW(hwnd, cls, 256)
        if cls.value != "#32770":          # standard dialog class
            return True
        title = ctypes.create_unicode_buffer(512)
        ctypes.windll.user32.GetWindowTextW(hwnd, title, 512)
        if "directx" in title.value.lower():
            found.append(title.value)
            return False
        return True

    try:
        ctypes.windll.user32.EnumWindows(EnumProc(cb), 0)
    except Exception:
        return False
    return bool(found)


def kill_stale_runs():
    """
    Kill any leftover train.py (and its collector children) from an earlier launch.

    Orphans are not harmless here: a stale run keeps its CUDA context and, worse, keeps
    binding the TMInterface ports, so a fresh run connects, gets its socket pulled out from
    under it mid-rollout, and never completes one. Ten orphaned processes once held 12.9 GB
    of a 16 GB card. Called immediately before each launch, when we own no child yet.
    """
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object { $_.CommandLine -and $_.CommandLine -like '*linesight*' -and "
        "($_.CommandLine -like '*train.py*' -or $_.CommandLine -like '*multiprocessing*') } | "
        "ForEach-Object { taskkill /PID $_.ProcessId /T /F }"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True)


def kill_game():
    for name in GAME_PROCS:
        subprocess.run(["taskkill", "/IM", f"{name}.exe", "/F"], capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="flybrain_hocko")
    ap.add_argument("--stall-minutes", type=float, default=8.0,
                    help="restart if no rollout completes for this long")
    ap.add_argument("--grace-minutes", type=float, default=6.0,
                    help="allow this long after launch before the stall check arms")
    ap.add_argument("--max-restarts", type=int, default=500)
    args = ap.parse_args()

    tb_dir = ROOT / "tensorboard" / args.run
    train_log = ROOT / "logs" / "flybrain_hocko.log"
    sup_log = ROOT / "logs" / "supervisor.log"
    sup_log.parent.mkdir(parents=True, exist_ok=True)

    stall_s = args.stall_minutes * 60
    grace_s = args.grace_minutes * 60

    log(f"supervising '{args.run}' (stall threshold {args.stall_minutes:g} min)", sup_log)
    if keep_awake(True):
        log("idle sleep and screensaver suppressed for the duration of this run", sup_log)
    else:
        log("warning: could not suppress idle sleep; check screensaver settings manually", sup_log)

    for attempt in range(1, args.max_restarts + 1):
        kill_stale_runs()
        kill_game()
        time.sleep(3)

        log(f"--- launch #{attempt}", sup_log)
        with train_log.open("a", encoding="utf-8") as lf:
            proc = subprocess.Popen(
                [sys.executable, str(ROOT / "scripts" / "train.py")],
                cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT,
            )

        started = time.time()
        last_progress = newest_progress(tb_dir)
        last_change = time.time()
        reason = None

        while True:
            time.sleep(20)

            if proc.poll() is not None:
                reason = f"train.py exited with code {proc.returncode}"
                break

            if directx_dialog_present():
                reason = "TrackMania is stuck on a DirectX Initialisation dialog (device lost)"
                log(f"{reason}; restarting immediately", sup_log)
                kill_tree(proc.pid)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
                break

            now = newest_progress(tb_dir)
            if now is not None and (last_progress is None or now > last_progress):
                last_progress = now
                last_change = time.time()

            if time.time() - started < grace_s:
                continue

            idle = time.time() - last_change
            if idle > stall_s:
                reason = f"no rollout completed for {idle / 60:.1f} min - hung, not crashed"
                log(f"{reason}; killing process tree {proc.pid}", sup_log)
                kill_tree(proc.pid)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
                break

        log(f"stopped: {reason}", sup_log)
        kill_game()
        time.sleep(15)

    log("restart limit reached, giving up", sup_log)
    keep_awake(False)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("stopped by user")
    finally:
        keep_awake(False)
