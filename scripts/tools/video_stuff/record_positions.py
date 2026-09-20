"""
Replay a set of ".inputs" files through TMInterface and record the car's position
every action step, so the trajectories can be drawn outside the game.

This is what you need to render a "many cars at once" montage: the .inputs files
only contain key presses, so the game's physics has to be run to turn them into
positions.

The message loop here mirrors game_instance_manager.rollout(), which is the code
path that is exercised during training, rather than inputs_to_gbx.py.

Example:
    python scripts/tools/video_stuff/record_positions.py \
        -i save/hocko_run1/best_runs -o positions.json -p 8477
"""

import argparse
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path

from config_files import config_copy as config
from config_files import user_config
from trackmania_rl import map_loader
from trackmania_rl.tmi_interaction.tminterface2 import MessageType, TMInterface

MS_PER_ACTION = config.ms_per_action  # 50 ms with default settings
KEYMAP = {"up": "accelerate", "down": "brake", "left": "left", "right": "right"}
LINE_RE = re.compile(r"^\s*([\d.]+)\s*-\s*([\d.]+)\s+press\s+(\w+)\s*$")


def parse_inputs_file(path: Path):
    """'1.55-1.6 press right' lines -> one {left,right,accelerate,brake} dict per action step."""
    intervals = []
    for line in path.read_text().splitlines():
        m = LINE_RE.match(line)
        if not m:
            continue
        a, b, key = float(m.group(1)), float(m.group(2)), m.group(3)
        if key in KEYMAP:
            intervals.append((a, b, KEYMAP[key]))
    if not intervals:
        return []
    n_steps = int(round(max(b for _, b, _ in intervals) / (MS_PER_ACTION / 1000))) + 1
    timeline = [{"left": False, "right": False, "accelerate": False, "brake": False} for _ in range(n_steps)]
    for a, b, key in intervals:
        i0 = int(round(a / (MS_PER_ACTION / 1000)))
        i1 = int(round(b / (MS_PER_ACTION / 1000)))
        for i in range(max(0, i0), min(n_steps, i1)):
            timeline[i][key] = True
    return timeline


def launch_game(tmi_port: int) -> int:
    launch_string = (
        'powershell -executionPolicy bypass -command "& {'
        f" $process = start-process -FilePath '{user_config.windows_TMLoader_path}'"
        " -PassThru -ArgumentList "
        f'\'run TmForever "{user_config.windows_TMLoader_profile_name}" /configstring=\\"set custom_port {tmi_port}\\"\';'
        ' echo exit $process.id}"'
    )
    loader_pid = int(subprocess.check_output(launch_string).decode().split("\r\n")[1])
    while True:
        out = subprocess.check_output(
            [
                "powershell",
                "-Command",
                "Get-CimInstance Win32_Process | "
                "Where-Object {$_.Name -eq 'TmForever.exe'} | "
                "ForEach-Object { \"$($_.ParentProcessId) $($_.ProcessId)\" }",
            ],
            text=True,
        )
        for row in out.strip().splitlines():
            parent, pid = (int(x) for x in row.split())
            if parent == loader_pid:
                return pid
        time.sleep(0.2)


def focus_game_window(pid: int, attempts: int = 40):
    """TMNF throttles when unfocused, which stops the on-step callbacks entirely."""
    try:
        import win32com.client
        import win32gui
        import win32process
    except ImportError:
        return
    for _ in range(attempts):
        hwnds = []

        def cb(hwnd, acc):
            if win32gui.IsWindowVisible(hwnd):
                _, found = win32process.GetWindowThreadProcessId(hwnd)
                if found == pid:
                    acc.append(hwnd)
            return True

        win32gui.EnumWindows(cb, hwnds)
        if hwnds:
            try:
                win32com.client.Dispatch("WScript.Shell").SendKeys("%")
                win32gui.SetForegroundWindow(hwnds[0])
                print(f"focused game window {hwnds[0]}")
            except Exception as e:
                print("could not focus game window:", e)
            return
        time.sleep(0.25)
    print("game window not found to focus")


def client_rect_on_screen(pid: int):
    """Screen coords of the game's client area (no title bar / borders)."""
    import win32gui
    import win32process

    hwnds = []

    def cb(hwnd, acc):
        if win32gui.IsWindowVisible(hwnd):
            _, found = win32process.GetWindowThreadProcessId(hwnd)
            if found == pid:
                acc.append(hwnd)
        return True

    win32gui.EnumWindows(cb, hwnds)
    if not hwnds:
        return None
    hwnd = hwnds[0]
    l, t, r, b = win32gui.GetClientRect(hwnd)
    x, y = win32gui.ClientToScreen(hwnd, (l, t))
    return x, y, (r - l) & ~1, (b - t) & ~1  # libx264 wants even dimensions


def start_capture(rect, path, fps, ffmpeg_exe):
    x, y, w, h = rect
    cmd = [
        ffmpeg_exe, "-y", "-loglevel", "error",
        "-f", "gdigrab", "-framerate", str(fps),
        "-offset_x", str(x), "-offset_y", str(y), "-video_size", f"{w}x{h}",
        "-draw_mouse", "0", "-i", "desktop",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        path,
    ]
    print(f"capturing {w}x{h} at ({x},{y}) -> {path}")
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def stop_capture(proc):
    if proc is None:
        return
    try:
        proc.stdin.write(b"q")
        proc.stdin.flush()
        proc.wait(timeout=15)
    except Exception:
        proc.kill()


def wait_for_tminterface_window(pid: int, timeout_s: float = 60.0):
    """The window title lists loaded mods; wait until TMInterface is actually in it."""
    try:
        import win32gui
        import win32process
    except ImportError:
        return
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        titles = []

        def cb(hwnd, acc):
            if win32gui.IsWindowVisible(hwnd):
                _, found = win32process.GetWindowThreadProcessId(hwnd)
                if found == pid:
                    acc.append(win32gui.GetWindowText(hwnd))
            return True

        win32gui.EnumWindows(cb, titles)
        for t in titles:
            if "TMInterface" in t:
                print(f"plugin ready: {t}")
                return
        time.sleep(0.5)
    print("timed out waiting for TMInterface in the window title")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs_dir", "-i", required=True, help="folder of best_runs/<name>/<name>.inputs")
    parser.add_argument("--out", "-o", required=True, help="output json path")
    parser.add_argument("--map_path", "-m", default="ESL-Hockolicious.Challenge.Gbx")
    parser.add_argument("--tmi_port", "-p", type=int, default=8477)
    parser.add_argument("--speed", "-s", type=float, default=8.0)
    parser.add_argument("--keep_open", action="store_true", help="leave the game running when finished")
    parser.add_argument("--verbose", "-v", action="store_true", help="log message types")
    parser.add_argument("--capture", help="mp4 path; screen-capture the game window while the lap plays")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="path to ffmpeg.exe")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--cam", type=int, default=1, help="in-game camera number")
    args = parser.parse_args()

    root = Path(args.inputs_dir)
    files = sorted(root.rglob("*.inputs"))
    if not files:
        raise SystemExit(f"no .inputs files under {root}")
    runs = []
    for f in files:
        tl = parse_inputs_file(f)
        if tl:
            runs.append({"name": f.stem, "timeline": tl, "positions": [], "last_t": -1})
            print(f"  {f.stem}: {len(tl)} action steps ({len(tl) * MS_PER_ACTION / 1000:.2f}s)")
    print(f"{len(runs)} runs to replay on port {args.tmi_port}")

    tm_pid = launch_game(args.tmi_port)
    print(f"game pid {tm_pid}")
    focus_game_window(tm_pid)
    wait_for_tminterface_window(tm_pid)

    iface = TMInterface(args.tmi_port)
    while True:
        try:
            iface.register(20)
            break
        except ConnectionRefusedError:
            time.sleep(0.5)

    idx = 0            # which run we are replaying
    phase = "connect"  # connect -> restart -> countdown -> racing
    map_requested = False
    start_state = None   # simulation state at t=0, used to reset between runs
    cap = None           # ffmpeg capture process
    t_start = time.perf_counter()

    seen = {}
    stalls = 0
    try:
        while idx < len(runs):
            try:
                msgtype = iface._read_int32()
            except socket.timeout:
                # Most often the game lost focus and stopped stepping. Nudge it and retry.
                stalls += 1
                print(f"  ...no message for a while (stall {stalls}/5) phase={phase} idx={idx}; refocusing")
                if stalls >= 5:
                    raise
                focus_game_window(tm_pid, attempts=4)
                continue
            stalls = 0
            if args.verbose:
                name = MessageType(msgtype).name if msgtype in [int(m) for m in MessageType] else str(msgtype)
                if seen.get(name, 0) < 3:
                    print(f"    msg {name} (phase={phase})")
                seen[name] = seen.get(name, 0) + 1

            if msgtype == int(MessageType.SC_RUN_STEP_SYNC):
                _time = iface._read_int32()

                if not map_requested:
                    pass
                elif phase == "restart":
                    iface.give_up()
                    phase = "countdown"
                elif phase == "countdown":
                    if _time >= 0:
                        phase = "racing"
                if phase == "racing":
                    run = runs[idx]
                    step = _time // MS_PER_ACTION
                    if _time == 0 and start_state is None:
                        start_state = iface.get_simulation_state()
                    if args.capture and cap is None and _time == 0:
                        iface.execute_command(f"cam {args.cam}")
                        iface.toggle_interface(False)
                        rect = client_rect_on_screen(tm_pid)
                        if rect:
                            cap = start_capture(rect, args.capture, args.fps, args.ffmpeg)
                    if 0 <= _time and step < len(run["timeline"]):
                        iface.set_input_state(**run["timeline"][step])
                        # the on-step callback fires many times per action step; keep one sample each
                        if _time != run["last_t"]:
                            run["last_t"] = _time
                            st = iface.get_simulation_state()
                            p = st.dyna.current_state.position
                            run["positions"].append([_time, round(p[0], 2), round(p[1], 2), round(p[2], 2)])
                    elif _time >= 0:
                        el = time.perf_counter() - t_start
                        print(f"[{el:6.1f}s] {run['name']}: {len(run['positions'])} points (ran out of inputs)")
                        stop_capture(cap)
                        cap = None
                        idx += 1
                        if idx < len(runs) and start_state is not None:
                            iface.rewind_to_state(start_state)
                        else:
                            phase = "restart"
                iface._respond_to_call(msgtype)

            elif msgtype == int(MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC):
                current = iface._read_int32()
                target = iface._read_int32()
                # Crossing the finish line ends the race and the game stops stepping,
                # so bank the run and restart immediately rather than waiting for a
                # run-step callback that will never come.
                if current == target and phase == "racing":
                    # Un-finish the race: crossing the line ends the simulation and the
                    # game stops stepping. Same trick as game_instance_manager.rollout().
                    st = iface.get_simulation_state()
                    if len(st.cp_data.cp_times) != 0:
                        st.cp_data.cp_times[-1].time = -1
                        iface.rewind_to_state(st)
                    else:
                        iface.prevent_simulation_finish()
                    run = runs[idx]
                    el = time.perf_counter() - t_start
                    print(f"[{el:6.1f}s] {run['name']}: {len(run['positions'])} points (finished)")
                    stop_capture(cap)
                    cap = None
                    idx += 1
                    if idx < len(runs) and start_state is not None:
                        iface.rewind_to_state(start_state)
                    elif idx < len(runs):
                        iface.give_up()
                        phase = "countdown"
                iface._respond_to_call(msgtype)

            elif msgtype == int(MessageType.SC_LAP_COUNT_CHANGED_SYNC):
                iface._read_int32()
                iface._read_int32()
                iface._respond_to_call(msgtype)

            elif msgtype == int(MessageType.SC_REQUESTED_FRAME_SYNC):
                iface._respond_to_call(msgtype)

            elif msgtype == int(MessageType.C_SHUTDOWN):
                iface.close()
                break

            elif msgtype == int(MessageType.SC_ON_CONNECT_SYNC):
                iface.execute_command("toggle_console")  # TMI opens its console on launch
                iface.set_on_step_period(MS_PER_ACTION)
                iface.set_speed(args.speed)
                iface.execute_command(f"set countdown_speed {args.speed}")
                iface.execute_command(f"set autologin {config.username}")
                iface.execute_command("set unfocused_fps_limit false")
                iface.execute_command("set skip_map_load_screens true")
                iface.execute_command("set disable_forced_camera true")
                iface.execute_command("set autorewind false")
                iface.execute_command("set auto_reload_plugins false")
                iface.set_timeout(30_000)
                if iface.is_in_menus():
                    try:
                        map_loader.hide_personal_record_replay(args.map_path, True)
                    except Exception as e:
                        print("hide_personal_record_replay:", e)
                    iface.execute_command(f"map {args.map_path}")
                map_requested = True
                phase = "restart"
                iface._respond_to_call(msgtype)
    except socket.timeout:
        print("socket timed out - the game stopped talking to us")
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        stop_capture(cap)
        done = [r for r in runs if r["positions"]]
        out = {
            "map": args.map_path,
            "ms_per_action": MS_PER_ACTION,
            "runs": [{"name": r["name"], "positions": r["positions"]} for r in done],
        }
        Path(args.out).write_text(json.dumps(out))
        print(f"wrote {args.out}: {len(done)}/{len(runs)} runs recorded")
        if not args.keep_open:
            os.system(f"taskkill /PID {tm_pid} /f >nul 2>&1")


if __name__ == "__main__":
    main()
