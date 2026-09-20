"""
Watch the fly brain drive, live.

While train.py is running, the collector's inference network publishes its settled activity
to a memory-mapped file (see FlyBrain.enable_live_tap). This serves a local page that renders
those 139,248 neurons firing in real time, over a static drawing of the connectome's wiring.

    python scripts/tools/flywire/live_viewer.py
    -> http://127.0.0.1:8130

Reads the mmap only; it never touches the training processes. Safe to start and stop at will.
"""

import argparse
import json
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

TAP_MAGIC = b"FLYTAP01"
TAP_HEADER = 256
OFF_FRAME, OFF_N, OFF_NACT = 8, 16, 20
OFF_RMS, OFF_MAX, OFF_SCALE, OFF_Q, OFF_ACTION, OFF_TIME = 24, 28, 32, 36, 84, 88

ACTIONS = [
    "accel", "left+accel", "right+accel", "coast", "left", "right",
    "brake", "left+brake", "right+brake", "accel+brake",
    "left+accel+brake", "right+accel+brake",
]

GROUPS = [
    ("Optic lobe", "#0099AD", [0]),
    ("Visual relay", "#5FC8D8", [1, 2]),
    ("Sensory / body", "#B88100", [3, 4, 5]),
    ("Central brain", "#98449E", [6]),
    ("Descending", "#DC5E59", [7]),
    ("Motor / other", "#6E7C85", [8, 9]),
]

CANVAS_W, CANVAS_H = 1280, 660
EDGE_SAMPLE = 26000



# Reference times for ESL-Hockolicious. The fly's own best is read live from tensorboard.
REFERENCE_TIMES = [
    ("World record", "tastyy", 53.760, "#E9EEF3"),
    ("Best dense net", "21M frames", 53.830, "#0099AD"),
    ("Teacher", "distilled from", 54.370, "#5FC8D8"),
    ("Author medal", "Nadeo", 55.100, "#6E7C85"),
]

# Every connectome run so far. Frames and furthest zone come from the saved tensorboard logs;
# the finish line is zone 8082.
RUN_HISTORY = [
    ("rl 1", 209008, 604, "diverged"),
    ("rl 2", 257355, 670, "gradients clipped 200x"),
    ("rl 3", 297371, 609, "drive 70-230x too weak"),
    ("rl 4", 173653, 630, "readout scale"),
    ("rl 5", 59390, 609, "integrator unstable"),
    ("rl 6", 29771, 609, "settle loop fixed"),
    ("distilled", 90708, 8082, "FINISHED 57.110s"),
]


class Stats:
    """Live training numbers, read from the run's tensorboard and cached."""

    def __init__(self, run: str, period: float = 15.0):
        self.run = run
        self.period = period
        self._cache = {"frames": 0, "rollouts": 0, "zone_max": 0, "best": None, "median": None, "run": run}
        self._when = 0.0
        self._lock = threading.Lock()

    def get(self):
        import time as _t

        with self._lock:
            if _t.time() - self._when < self.period:
                return self._cache
            self._when = _t.time()
            try:
                from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

                d = ROOT / "tensorboard" / self.run
                if d.exists():
                    ea = EventAccumulator(str(d), size_guidance={"scalars": 0})
                    ea.Reload()
                    tags = ea.Tags()["scalars"]
                    z = ea.Scalars("single_zone_reached_trained_hock") if "single_zone_reached_trained_hock" in tags else []
                    ev = ea.Scalars("eval_race_time_trained_hock") if "eval_race_time_trained_hock" in tags else []
                    fin = [x.value for x in ev if x.value < 300]
                    # The median over a window of eval laps, with laps that did not finish counted
                    # at the cutoff. Lap times here carry ~1.7s of noise, so the best SINGLE lap is
                    # not evidence the policy improved -- trusting it is what cost us a good policy
                    # and recorded three checkpoints that crash at 4.5% of the track.
                    mtag = "eval_median_7lap_hock"
                    med = [x.value for x in ea.Scalars(mtag)] if mtag in tags else []
                    self._cache = {
                        "run": self.run,
                        "frames": int(max((x.step for x in z), default=0)),
                        "rollouts": len(z),
                        "zone_max": int(max((x.value for x in z), default=0)),
                        "best": (min(fin) if fin else None),
                        "median": (min(med) if med else None),
                    }
            except Exception:
                pass
            return self._cache


class Geometry:
    """Neuron screen positions, group ids and a static edge sample, computed once."""

    def __init__(self, npz_path: Path):
        d = np.load(npz_path, allow_pickle=False)
        pos, sc = d["position"], d["super_class_id"]
        self.n = len(sc)

        cls2grp = np.zeros(16, dtype=np.uint8)
        for gi, (_, _, classes) in enumerate(GROUPS):
            for c in classes:
                cls2grp[c] = gi
        self.grp = cls2grp[sc].astype(np.uint8)

        x, y = pos[:, 0].astype(np.float64), pos[:, 1].astype(np.float64)
        pad = 18
        s = min((CANVAS_W - 2 * pad) / (x.max() - x.min()), (CANVAS_H - 2 * pad) / (y.max() - y.min()))
        px = (x - x.min()) * s + pad + ((CANVAS_W - 2 * pad) - (x.max() - x.min()) * s) / 2
        py = (y - y.min()) * s + pad + ((CANVAS_H - 2 * pad) - (y.max() - y.min()) * s) / 2
        py = CANVAS_H - py  # dorsal up
        self.px = np.clip(px, 0, CANVAS_W - 1).astype(np.uint16)
        self.py = np.clip(py, 0, CANVAS_H - 1).astype(np.uint16)

        # Static wiring, stratified toward rare classes so the descending pathway survives.
        src, dst = d["edge_src"].astype(np.int64), d["edge_dst"].astype(np.int64)
        rng = np.random.default_rng(0)
        w = 1.0 / np.bincount(sc, minlength=16)[sc[src]]
        w /= w.sum()
        pick = rng.choice(len(src), size=min(EDGE_SAMPLE, len(src)), replace=False, p=w)
        self.esrc = src[pick].astype(np.uint32)
        self.edst = dst[pick].astype(np.uint32)

        self.counts = [int(((self.grp == gi)).sum()) for gi in range(len(GROUPS))]

    def blob(self) -> bytes:
        return (
            self.px.tobytes() + self.py.tobytes() + self.grp.tobytes()
            + self.esrc.tobytes() + self.edst.tobytes()
        )


class Tap:
    """Reader for the memory-mapped activity file the training process publishes."""

    def __init__(self, path: Path, n: int):
        self.path, self.n, self.mm = path, n, None
        self.lock = threading.Lock()
        # Presence signal: the training process only publishes while this file is fresh.
        self.watch_path = Path(str(path) + ".watch")

    def heartbeat(self):
        try:
            self.watch_path.parent.mkdir(parents=True, exist_ok=True)
            self.watch_path.touch()
        except OSError:
            pass

    def _open(self):
        if self.mm is not None:
            return True
        if not self.path.exists():
            return False
        try:
            size = self.path.stat().st_size
            if size < TAP_HEADER + self.n:
                return False
            self.mm = np.memmap(self.path, dtype=np.uint8, mode="r", shape=(TAP_HEADER + self.n,))
            return True
        except Exception:
            self.mm = None
            return False

    def read(self):
        with self.lock:
            if not self._open():
                return None
            try:
                hdr = bytes(self.mm[:TAP_HEADER])
                if hdr[:8] != TAP_MAGIC:
                    return None
                meta = {
                    "frame": struct.unpack_from("<Q", hdr, OFF_FRAME)[0],
                    "rms": struct.unpack_from("<f", hdr, OFF_RMS)[0],
                    "max": struct.unpack_from("<f", hdr, OFF_MAX)[0],
                    "scale": struct.unpack_from("<f", hdr, OFF_SCALE)[0],
                    "q": list(struct.unpack_from("<12f", hdr, OFF_Q)),
                    "action": struct.unpack_from("<i", hdr, OFF_ACTION)[0],
                    "t": struct.unpack_from("<d", hdr, OFF_TIME)[0],
                }
                return meta, bytes(self.mm[TAP_HEADER:])
            except Exception:
                self.mm = None
                return None


def build_page(geo: Geometry) -> str:
    cfg = {
        "w": CANVAS_W, "h": CANVAS_H, "n": geo.n, "edges": int(len(geo.esrc)),
        "groups": [{"name": g[0], "color": g[1], "count": geo.counts[i]} for i, g in enumerate(GROUPS)],
        "actions": ACTIONS,
        "reference": [{"who": r[0], "sub": r[1], "t": r[2], "c": r[3]} for r in REFERENCE_TIMES],
        "history": [{"run": h[0], "frames": h[1], "zone": h[2], "note": h[3]} for h in RUN_HISTORY],
        "finish_zone": 8082,
    }
    return PAGE.replace("__CFG__", json.dumps(cfg))


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fly Brain — live</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{
  --ground:#0A0D12; --raised:#111721; --line:#222C3A; --line-soft:#1A2430;
  --ink:#E9EEF3; --ink-2:#97A5B3; --ink-3:#66727F;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
  --sans:"IBM Plex Sans",system-ui,sans-serif;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);font-size:14px}
.wrap{max-width:1340px;margin:0 auto;padding:22px 20px 40px}
h1{font-size:15px;font-weight:600;margin:0;letter-spacing:.01em}
.sub{font-family:var(--mono);font-size:11px;color:var(--ink-3);margin-top:3px;letter-spacing:.04em}
.top{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;flex-wrap:wrap;margin-bottom:14px}
.pill{font-family:var(--mono);font-size:11px;padding:5px 10px;border:1px solid var(--line);
  border-radius:2px;color:var(--ink-2);background:var(--raised)}
.pill b{color:var(--ink);font-weight:500}
.pill.bad{border-color:#5A2C2C;color:#DC5E59}
.pill.good{border-color:#1E4A44;color:#5FC8D8}
.stage{position:relative;background:#070A0E;border:1px solid var(--line);border-radius:3px;overflow:hidden}
canvas{display:block;width:100%;height:auto}
#bg{position:absolute;inset:0}
#fg{position:relative}
.cols{display:grid;grid-template-columns:1fr 310px;gap:18px;align-items:start}
@media(max-width:1000px){.cols{grid-template-columns:1fr}}
.panel{background:var(--raised);border:1px solid var(--line);border-radius:3px;padding:14px 15px}
.panel h2{font-family:var(--mono);font-size:10.5px;letter-spacing:.13em;text-transform:uppercase;
  color:var(--ink-3);margin:0 0 12px;font-weight:400}
.arow{display:grid;grid-template-columns:104px 1fr 52px;gap:9px;align-items:center;margin-bottom:5px}
.aname{font-family:var(--mono);font-size:10.5px;color:var(--ink-3);text-align:right;white-space:nowrap}
.abar{height:11px;background:#0C1219;position:relative;border-radius:0 2px 2px 0}
.abar i{display:block;height:100%;background:#2C4A57;border-radius:0 2px 2px 0}
.arow.sel .aname{color:var(--ink)}
.arow.sel .abar i{background:#DC5E59}
.aval{font-family:var(--mono);font-size:10.5px;color:var(--ink-2);font-variant-numeric:tabular-nums}
.legend{display:flex;flex-direction:column;gap:7px;margin-top:2px}
.lg{display:flex;align-items:center;gap:9px;cursor:pointer;font-family:var(--mono);font-size:11px;
  color:var(--ink-2);user-select:none;border:0;background:none;padding:2px 0;text-align:left;width:100%}
.lg[aria-pressed="false"]{opacity:.35}
.lg .sw{width:9px;height:9px;border-radius:50%;flex:none}
.lg .ct{margin-left:auto;color:var(--ink-3);font-size:10px}
.kv{display:flex;justify-content:space-between;font-family:var(--mono);font-size:11px;
  color:var(--ink-3);padding:4px 0;border-bottom:1px solid var(--line-soft)}
.kv b{color:var(--ink);font-weight:500;font-variant-numeric:tabular-nums}
.kv:last-child{border-bottom:0}
.ctrl{display:flex;gap:8px;align-items:center;margin-top:12px;flex-wrap:wrap}
.ctrl label{font-family:var(--mono);font-size:10.5px;color:var(--ink-3)}
input[type=range]{width:92px;accent-color:#0099AD}

.lap{display:grid;grid-template-columns:96px 1fr 52px;gap:8px;align-items:center;margin-bottom:6px}
.lap .w{font-family:var(--mono);font-size:10px;color:var(--ink-3);text-align:right;line-height:1.25}
.lap .w b{color:var(--ink-2);font-weight:500;display:block}
.lap .bar{position:relative;height:13px;background:#0C1219;border-radius:2px}
.lap .bar i{position:absolute;top:0;bottom:0;width:3px;border-radius:2px}
.lap .v{font-family:var(--mono);font-size:10.5px;color:var(--ink-2);font-variant-numeric:tabular-nums;text-align:right}
.lap.me .w b,.lap.me .v{color:#DC5E59}
.hrow{display:grid;grid-template-columns:52px 1fr 40px;gap:8px;font-family:var(--mono);font-size:10px;
  color:var(--ink-3);padding:3px 0;border-bottom:1px solid var(--line-soft)}
.hrow:last-child{border-bottom:0}
.hrow .f{color:var(--ink-2);font-variant-numeric:tabular-nums;text-align:right}
.hrow.ok{color:#DC5E59}
.hrow.ok .f{color:#DC5E59}
.prog{height:5px;background:#0C1219;border-radius:3px;overflow:hidden;margin-top:9px}
.prog i{display:block;height:100%;background:#DC5E59}
.note{font-family:var(--mono);font-size:10.5px;color:var(--ink-3);margin-top:12px;line-height:1.55}
</style></head><body>
<div class="wrap">
  <div class="top">
    <div><h1>Fly brain, driving</h1>
      <div class="sub">FlyWire FAFB v783 · 139,248 neurons · live from the collector's inference network</div></div>
    <div style="display:flex;gap:7px;flex-wrap:wrap">
      <span class="pill" id="p-status">connecting…</span>
      <span class="pill">frame <b id="p-frame">—</b></span>
      <span class="pill"><b id="p-hz">—</b> Hz</span>
      <span class="pill">active <b id="p-act">—</b></span>
    </div>
  </div>

  <div class="cols">
    <div>
      <div class="stage">
        <canvas id="bg"></canvas>
        <canvas id="fg"></canvas>
      </div>
      <div class="ctrl">
        <label>gain</label><input type="range" id="gain" min="20" max="400" value="130">
        <label>trail</label><input type="range" id="trail" min="0" max="92" value="55">
        <label>wiring</label><input type="range" id="wire" min="0" max="100" value="34">
      </div>
      <div class="note" id="note"></div>
    </div>

    <div style="display:flex;flex-direction:column;gap:14px">
      <div class="panel">
        <h2>What it wants to do</h2>
        <div id="actions"></div>
      </div>
      <div class="panel">
        <h2>Populations</h2>
        <div class="legend" id="legend"></div>
      </div>
      <div class="panel">
        <h2>Signal</h2>
        <div id="stats"></div>
      </div>
      <div class="panel">
        <h2>Best laps</h2>
        <div id="laps"></div>
        <div class="prog"><i id="trackprog" style="width:0%"></i></div>
        <div class="note" id="progtxt" style="margin-top:6px"></div>
      </div>
      <div class="panel">
        <h2>Connectome runs</h2>
        <div id="history"></div>
      </div>
    </div>
  </div>
</div>

<script>
const CFG = __CFG__;
const W = CFG.w, H = CFG.h, N = CFG.n;

const bg = document.getElementById('bg'), fg = document.getElementById('fg');
bg.width = fg.width = W; bg.height = fg.height = H;
const bctx = bg.getContext('2d'), fctx = fg.getContext('2d', {alpha:true});

let PX, PY, GRP, ESRC, EDST;
const on = CFG.groups.map(()=>true);
const RGB = CFG.groups.map(g=>[parseInt(g.color.slice(1,3),16),parseInt(g.color.slice(3,5),16),parseInt(g.color.slice(5,7),16)]);

/* accumulation buffer so activity can leave a short trail */
const acc = new Float32Array(W*H*3);
const img = fctx.createImageData(W,H);
for (let i=0;i<W*H;i++) img.data[i*4+3] = 255;

function drawWiring(){
  const alpha = (+document.getElementById('wire').value)/100;
  bctx.fillStyle = '#070A0E'; bctx.fillRect(0,0,W,H);
  if (alpha <= 0 || !ESRC) return;
  bctx.globalCompositeOperation = 'lighter';
  for (let g=0;g<CFG.groups.length;g++){
    if (!on[g]) continue;
    bctx.strokeStyle = CFG.groups[g].color;
    bctx.globalAlpha = (g===4 ? 0.13 : 0.030) * alpha;
    bctx.lineWidth = 0.6;
    bctx.beginPath();
    for (let k=0;k<ESRC.length;k++){
      const s = ESRC[k]; if (GRP[s] !== g) continue;
      const d = EDST[k]; if (!on[GRP[d]]) continue;
      bctx.moveTo(PX[s],PY[s]); bctx.lineTo(PX[d],PY[d]);
    }
    bctx.stroke();
  }
  bctx.globalAlpha = 1; bctx.globalCompositeOperation = 'source-over';
}

function render(activity){
  const decay = (+document.getElementById('trail').value)/100;
  const gain  = (+document.getElementById('gain').value)/100;
  for (let i=0;i<acc.length;i++) acc[i] *= decay;
  let active = 0;
  for (let i=0;i<N;i++){
    const a = activity[i];
    if (a < 6) continue;
    const g = GRP[i]; if (!on[g]) continue;
    active++;
    const v = (a/255) * gain;
    const p = (PY[i]*W + PX[i])*3, c = RGB[g];
    acc[p]   += c[0]*v; acc[p+1] += c[1]*v; acc[p+2] += c[2]*v;
  }
  const d = img.data;
  for (let p=0,q=0;p<acc.length;p+=3,q+=4){
    const r=acc[p], gg=acc[p+1], b=acc[p+2];
    d[q]   = r>255?255:r;
    d[q+1] = gg>255?255:gg;
    d[q+2] = b>255?255:b;
    d[q+3] = (r+gg+b) > 8 ? 255 : 0;
  }
  fctx.putImageData(img,0,0);
  return active;
}

function buildPanels(){
  const leg = document.getElementById('legend');
  CFG.groups.forEach((g,gi)=>{
    const b = document.createElement('button');
    b.className='lg'; b.type='button'; b.setAttribute('aria-pressed','true');
    b.innerHTML = `<span class="sw" style="background:${g.color}"></span>${g.name}<span class="ct">${g.count.toLocaleString()}</span>`;
    b.onclick = ()=>{ on[gi]=!on[gi]; b.setAttribute('aria-pressed',on[gi]?'true':'false'); drawWiring(); };
    leg.appendChild(b);
  });
  const ac = document.getElementById('actions');
  CFG.actions.forEach((name,i)=>{
    const r = document.createElement('div');
    r.className='arow'; r.id='a'+i;
    r.innerHTML = `<div class="aname">${name}</div><div class="abar"><i style="width:0%"></i></div><div class="aval">—</div>`;
    ac.appendChild(r);
  });
  document.getElementById('wire').oninput = drawWiring;
}

function setActions(q, chosen){
  let lo=Infinity, hi=-Infinity;
  for (const v of q){ if(v<lo)lo=v; if(v>hi)hi=v; }
  const span = (hi-lo) || 1;
  q.forEach((v,i)=>{
    const row = document.getElementById('a'+i);
    row.classList.toggle('sel', i===chosen);
    row.querySelector('i').style.width = Math.max(1,((v-lo)/span)*100).toFixed(1)+'%';
    row.querySelector('.aval').textContent = v.toFixed(3);
  });
}

let lastFrame=-1, frames=0, hzT=performance.now(), missed=0;
async function poll(){
  try{
    const r = await fetch('/frame', {cache:'no-store'});
    if (r.status === 204){
      document.getElementById('p-status').textContent='waiting for training…';
      document.getElementById('p-status').className='pill bad';
      missed++;
    } else {
      const buf = new Uint8Array(await r.arrayBuffer());
      const dv = new DataView(buf.buffer, 0, 256);
      const frame = Number(dv.getBigUint64(8, true));
      if (frame !== lastFrame){
        lastFrame = frame; frames++;
        const activity = buf.subarray(256);
        const active = render(activity);
        const q = []; for(let i=0;i<12;i++) q.push(dv.getFloat32(36+i*4,true));
        setActions(q, dv.getInt32(84,true));
        document.getElementById('p-frame').textContent = frame.toLocaleString();
        document.getElementById('p-act').textContent = (100*active/N).toFixed(1)+'%';
        document.getElementById('stats').innerHTML =
          `<div class="kv"><span>activity rms</span><b>${dv.getFloat32(24,true).toFixed(3)}</b></div>`+
          `<div class="kv"><span>peak neuron</span><b>${dv.getFloat32(28,true).toFixed(2)}</b></div>`+
          `<div class="kv"><span>best Q</span><b>${Math.max(...q).toFixed(4)}</b></div>`+
          `<div class="kv"><span>Q spread</span><b>${(Math.max(...q)-Math.min(...q)).toFixed(4)}</b></div>`+
          `<div class="kv"><span>neurons lit</span><b>${active.toLocaleString()}</b></div>`;
        document.getElementById('p-status').textContent='live';
        document.getElementById('p-status').className='pill good';
      }
    }
  }catch(e){
    document.getElementById('p-status').textContent='viewer lost server';
    document.getElementById('p-status').className='pill bad';
  }
  const now = performance.now();
  if (now-hzT > 1000){
    document.getElementById('p-hz').textContent = (frames*1000/(now-hzT)).toFixed(0);
    frames=0; hzT=now;
  }
  setTimeout(poll, 33);
}

(async function init(){
  buildPanels();
  const g = new Uint8Array(await (await fetch('/geometry')).arrayBuffer());
  let o=0;
  PX = new Uint16Array(g.buffer, o, N); o += N*2;
  PY = new Uint16Array(g.buffer, o, N); o += N*2;
  GRP= new Uint8Array (g.buffer, o, N); o += N;
  ESRC=new Uint32Array(g.buffer, o, CFG.edges); o += CFG.edges*4;
  EDST=new Uint32Array(g.buffer, o, CFG.edges);
  drawWiring();
  document.getElementById('note').textContent =
    `${CFG.edges.toLocaleString()} connections drawn as static wiring; all ${N.toLocaleString()} neurons lit by live activity. `+
    `Tap publishes every 4th inference from collector 0.`;
  poll();

/* ---------------------------------------------------- lap times + history */
const LAP_LO = 53.5, LAP_HI = 58.0;
function lapPos(t){ return Math.max(0, Math.min(100, (t-LAP_LO)/(LAP_HI-LAP_LO)*100)); }

function renderLaps(best, median){
  const host=document.getElementById('laps'); if(!host) return;
  const rows = CFG.reference.map(r=>({...r, me:false}));
  // The verified median is the fly's real standing. The best single lap is kept for context but
  // marked, because a single fast lap sits inside ~1.7s of run-to-run noise.
  rows.push({who:'FLY BRAIN', sub: median?'median of 7 eval laps':'no verified window yet',
             t: median, c:'#DC5E59', me:true});
  if(best) rows.push({who:'best single lap', sub:'unverified \u00b11.7s noise', t: best, c:'#8A8A8A', me:false});
  host.innerHTML = rows.map(r=>{
    const has = typeof r.t === 'number' && isFinite(r.t);
    return `<div class="lap${r.me?' me':''}">`+
      `<div class="w"><b>${r.who}</b>${r.sub}</div>`+
      `<div class="bar">${has?`<i style="left:${lapPos(r.t).toFixed(1)}%;background:${r.c}"></i>`:''}</div>`+
      `<div class="v">${has?r.t.toFixed(3):'--'}</div></div>`;
  }).join('');
}

function renderHistory(live){
  const host=document.getElementById('history'); if(!host) return;
  const rows = CFG.history.slice();
  if(live && live.frames) rows.push({run:'rl now', frames:live.frames, zone:live.zone_max, note:'training'});
  host.innerHTML = rows.map(h=>{
    const done = h.zone >= CFG.finish_zone;
    const f = h.frames>=1000 ? (h.frames/1000).toFixed(0)+'k' : String(h.frames);
    return `<div class="hrow${done?' ok':''}"><span>${h.run}</span>`+
      `<span>${h.note}</span><span class="f">${f}</span></div>`;
  }).join('');
}

async function pollStats(){
  try{
    const s = await (await fetch('/stats',{cache:'no-store'})).json();
    renderLaps(s.best, s.median);
    renderHistory(s);
    const pct = Math.min(100, 100*(s.zone_max||0)/CFG.finish_zone);
    document.getElementById('trackprog').style.width = pct.toFixed(1)+'%';
    document.getElementById('progtxt').textContent =
      `${s.run} · ${(s.frames||0).toLocaleString()} frames · furthest zone ${(s.zone_max||0).toLocaleString()}/${CFG.finish_zone.toLocaleString()} (${pct.toFixed(1)}% of track)`;
  }catch(e){}
  setTimeout(pollStats, 15000);
}
renderLaps(null); renderHistory(null); pollStats();

})();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8130)
    ap.add_argument("--connectome", type=Path, default=ROOT / "data" / "flywire" / "connectome_783.npz")
    ap.add_argument("--tap", type=Path, default=ROOT / "data" / "flywire" / "live_activity.mmap")
    ap.add_argument("--run", default=None, help="run name to read live stats from (default: config run_name)")
    args = ap.parse_args()

    print("loading connectome geometry ...")
    geo = Geometry(args.connectome)
    page = build_page(geo).encode("utf-8")
    blob = geo.blob()
    run_name = args.run
    if run_name is None:
        try:
            from config_files import config_copy
            run_name = config_copy.run_name
        except Exception:
            run_name = 'flybrain_rl'
    stats = Stats(run_name)
    tap = Tap(args.tap, geo.n)
    print(f"  {geo.n:,} neurons, {len(geo.esrc):,} static edges")
    print(f"  live stats from tensorboard/{run_name}")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/geometry"):
                self._send(blob, "application/octet-stream")
            elif self.path.startswith("/stats"):
                body = json.dumps(stats.get()).encode("utf-8")
                self._send(body, "application/json")
            elif self.path.startswith("/frame"):
                tap.heartbeat()
                got = tap.read()
                if got is None:
                    self.send_response(204)
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    return
                meta, body = got
                self._send(bytes(self.raw_header(meta)) + body, "application/octet-stream")
            else:
                self._send(page, "text/html; charset=utf-8")

        def raw_header(self, meta):
            h = bytearray(TAP_HEADER)
            struct.pack_into("<8s", h, 0, TAP_MAGIC)
            struct.pack_into("<Q", h, OFF_FRAME, meta["frame"])
            struct.pack_into("<f", h, OFF_RMS, meta["rms"])
            struct.pack_into("<f", h, OFF_MAX, meta["max"])
            struct.pack_into("<f", h, OFF_SCALE, meta["scale"])
            struct.pack_into("<12f", h, OFF_Q, *meta["q"])
            struct.pack_into("<i", h, OFF_ACTION, meta["action"])
            struct.pack_into("<d", h, OFF_TIME, meta["t"])
            return h

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"\n  live viewer -> http://127.0.0.1:{args.port}\n  (ctrl-c to stop; training is untouched)\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
