"use strict";

const $ = (id) => document.getElementById(id);
const SIM_FPS = 30; // replay frames per second at 1x
const TRAIL = 50;
const FLASH = 8;
const C = {
  bg: "#0f1115", panel: "#171a21", dish: "#13202a", rim: "#2c4a5c", food: "#7ee07e",
  cell: "rgba(120,190,255,0.85)", cellRim: "#cfe6ff", ray: "rgba(255,255,255,0.07)",
  rayHit: "rgba(255,214,110,0.75)", trail: "120,190,255", grid: "#262a33", text: "#8a90a0",
  train: "#6cb6ff", eval: "#f0a35e", cursor: "rgba(255,255,255,0.35)",
};

const state = {
  runId: null, epochs: [], epochIdx: -1, metrics: [],
  replay: null, foodAt: [], scoreAt: [],
  frame: 0, playing: false, acc: 0, lastTs: 0, loadToken: 0, chart: null,
};

async function getJSON(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  return res.json();
}

const runBase = () => `/api/runs/${encodeURIComponent(state.runId)}`;

// Food positions and cumulative score for every frame, from food_init + eat events.
function prepareReplay(rep) {
  const foodAt = [], scoreAt = [];
  let food = rep.food_init, score = 0;
  for (const f of rep.frames) {
    if (f.e.length) {
      food = food.slice();
      for (const [i, x, y] of f.e) food[i] = [x, y];
      score += f.e.length;
    }
    foodAt.push(food);
    scoreAt.push(score);
  }
  return { foodAt, scoreAt };
}

// ---------------- data loading ----------------
async function loadRuns() {
  const runs = await getJSON("/api/runs");
  const sel = $("run");
  const prev = sel.value;
  sel.innerHTML = "";
  for (const r of runs) {
    const o = document.createElement("option");
    o.value = r.id;
    o.textContent = `${r.id}  (epoch ${r.latest_epoch ?? "–"}, best ${r.best_eval ?? "–"})`;
    sel.appendChild(o);
  }
  if (prev && runs.some((r) => r.id === prev)) sel.value = prev;
  return runs;
}

async function selectRun(id) {
  Object.assign(state, { runId: id, epochs: [], metrics: [], replay: null, epochIdx: -1 });
  await refreshRun();
  if (state.epochs.length) await loadEpochIdx(state.epochs.length - 1);
}

async function refreshRun() {
  const [metrics, epochs] = await Promise.all([
    getJSON(`${runBase()}/metrics`), getJSON(`${runBase()}/replays`),
  ]);
  state.metrics = metrics;
  state.epochs = epochs;
  $("epoch").max = Math.max(0, epochs.length - 1);
  drawChart();
}

async function loadEpochIdx(i) {
  if (i < 0 || i >= state.epochs.length) return;
  const token = ++state.loadToken;
  const epoch = state.epochs[i];
  state.epochIdx = i;
  $("epoch").value = i;
  $("epochLabel").textContent = `epoch ${epoch}`;
  const rep = await getJSON(`${runBase()}/replays/${epoch}`);
  if (token !== state.loadToken) return; // a newer request superseded this one
  Object.assign(state, prepareReplay(rep), { replay: rep, frame: 0, acc: 0 });
  $("frame").max = rep.frames.length - 1;
  setPlaying(true);
  updateFrameUI();
  drawChart();
  showInfo();
}

// ---------------- playback ----------------
function setPlaying(on) {
  state.playing = on && !!state.replay;
  $("play").textContent = state.playing ? "Pause" : "Play";
}

function setFrame(i) {
  if (!state.replay) return;
  state.frame = Math.max(0, Math.min(i, state.replay.frames.length - 1));
  updateFrameUI();
}

function updateFrameUI() {
  if (!state.replay) return;
  $("frame").value = state.frame;
  $("frameLabel").textContent = `${state.frame} / ${state.replay.frames.length - 1}`;
}

function tick(ts) {
  const dt = state.lastTs ? (ts - state.lastTs) / 1000 : 0;
  state.lastTs = ts;
  if (state.playing && state.replay) {
    state.acc += dt * SIM_FPS * Number($("speed").value);
    const n = Math.floor(state.acc);
    if (n > 0) {
      state.acc -= n;
      const last = state.replay.frames.length - 1;
      setFrame(state.frame + n);
      if (state.frame >= last) setPlaying(false);
    }
  }
  drawDish();
  requestAnimationFrame(tick);
}

// ---------------- rendering ----------------
function drawDish() {
  const c = $("dish"), ctx = c.getContext("2d");
  const W = c.width, H = c.height;
  ctx.fillStyle = C.bg;
  ctx.fillRect(0, 0, W, H);
  const rep = state.replay;
  if (!rep) {
    ctx.fillStyle = C.text;
    ctx.font = "16px system-ui";
    ctx.textAlign = "center";
    ctx.fillText(state.runId ? "loading…" : "no runs yet — start training", W / 2, H / 2);
    ctx.textAlign = "left";
    return;
  }
  const w = rep.world;
  const s = (Math.min(W, H) / 2 - 10) / w.radius;
  const X = (x) => W / 2 + x * s;
  const Y = (y) => H / 2 - y * s; // world y points up

  ctx.fillStyle = C.dish;
  ctx.strokeStyle = C.rim;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(W / 2, H / 2, w.radius * s, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  const fi = state.frame;
  const fr = rep.frames[fi];

  if ($("showTrail").checked) {
    const from = Math.max(0, fi - TRAIL);
    for (let k = from + 1; k <= fi; k++) {
      const a = rep.frames[k - 1].p, b = rep.frames[k].p;
      ctx.strokeStyle = `rgba(${C.trail},${(0.5 * (k - from)) / (fi - from + 1)})`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(X(a[0]), Y(a[1]));
      ctx.lineTo(X(b[0]), Y(b[1]));
      ctx.stroke();
    }
  }

  const fr_ = Math.max(2.5, w.food_r * s);
  ctx.fillStyle = C.food;
  for (const [x, y] of state.foodAt[fi]) {
    ctx.beginPath();
    ctx.arc(X(x), Y(y), fr_, 0, Math.PI * 2);
    ctx.fill();
  }

  // eat flash: expanding ring where a pellet was eaten in the last FLASH frames
  for (let k = Math.max(1, fi - FLASH + 1); k <= fi; k++) {
    const age = (fi - k) / FLASH;
    for (const [i] of rep.frames[k].e) {
      const [x, y] = state.foodAt[k - 1][i];
      ctx.strokeStyle = `rgba(126,224,126,${1 - age})`;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(X(x), Y(y), fr_ + age * 18, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  const [px, py] = fr.p;
  if ($("showRays").checked) {
    for (let j = 0; j < w.n_rays; j++) {
      const ang = fr.h + (j * 2 * Math.PI) / w.n_rays;
      const d = fr.r[j];
      const hit = d < w.ray_len - 1e-6;
      ctx.strokeStyle = hit ? C.rayHit : C.ray;
      ctx.lineWidth = hit ? 1.5 : 1;
      ctx.beginPath();
      ctx.moveTo(X(px), Y(py));
      ctx.lineTo(X(px + d * Math.cos(ang)), Y(py + d * Math.sin(ang)));
      ctx.stroke();
    }
  }

  const cr = Math.max(5, w.cell_r * s);
  ctx.fillStyle = C.cell;
  ctx.strokeStyle = C.cellRim;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(X(px), Y(py), cr, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(X(px), Y(py));
  ctx.lineTo(X(px + 1.8 * w.cell_r * Math.cos(fr.h)), Y(py + 1.8 * w.cell_r * Math.sin(fr.h)));
  ctx.stroke();

  ctx.fillStyle = "#e6e8ec";
  ctx.font = "15px system-ui";
  ctx.fillText(`epoch ${rep.epoch}`, 12, 22);
  ctx.fillText(`score ${state.scoreAt[fi]}`, 12, 42);
  ctx.fillText(`step ${fi} / ${rep.frames.length - 1}`, 12, 62);
}

function drawChart() {
  const c = $("chart"), ctx = c.getContext("2d");
  const W = c.width, H = c.height;
  const m = { l: 44, r: 14, t: 12, b: 28 };
  ctx.fillStyle = C.panel;
  ctx.fillRect(0, 0, W, H);
  const rows = state.metrics;
  if (!rows.length) {
    state.chart = null;
    ctx.fillStyle = C.text;
    ctx.font = "13px system-ui";
    ctx.fillText("no metrics yet", m.l, H / 2);
    return;
  }
  const eMax = Math.max(1, ...rows.map((r) => r.epoch));
  const vals = rows.flatMap((r) => [r.train_score, r.eval_score]).filter((v) => v != null);
  const yMax = Math.max(1, ...vals) * 1.1;
  const x = (e) => m.l + (e / eMax) * (W - m.l - m.r);
  const y = (v) => H - m.b - (v / yMax) * (H - m.t - m.b);

  ctx.font = "11px system-ui";
  ctx.lineWidth = 1;
  for (let k = 0; k <= 4; k++) {
    const v = (yMax * k) / 4;
    ctx.strokeStyle = C.grid;
    ctx.beginPath();
    ctx.moveTo(m.l, y(v));
    ctx.lineTo(W - m.r, y(v));
    ctx.stroke();
    ctx.fillStyle = C.text;
    ctx.fillText(v.toFixed(0), 8, y(v) + 4);
  }
  for (const e of [0, Math.round(eMax / 2), eMax]) ctx.fillText(String(e), x(e) - 6, H - 8);

  const series = (key, color, dots) => {
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    for (const r of rows) {
      if (r[key] == null) continue;
      if (started) ctx.lineTo(x(r.epoch), y(r[key]));
      else { ctx.moveTo(x(r.epoch), y(r[key])); started = true; }
    }
    ctx.stroke();
    if (dots) for (const r of rows) {
      if (r[key] == null) continue;
      ctx.beginPath();
      ctx.arc(x(r.epoch), y(r[key]), 3, 0, Math.PI * 2);
      ctx.fill();
    }
  };
  series("train_score", C.train, false);
  series("eval_score", C.eval, true);

  if (state.epochIdx >= 0) {
    const e = state.epochs[state.epochIdx];
    ctx.strokeStyle = C.cursor;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(x(e), m.t);
    ctx.lineTo(x(e), H - m.b);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  state.chart = { x0: m.l, x1: W - m.r, eMax };
}

function showInfo() {
  const e = state.epochs[state.epochIdx];
  const row = state.metrics.find((r) => r.epoch === e);
  if (!row) { $("info").textContent = ""; return; }
  const fmt = (v) => (typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(4)) : v ?? "–");
  $("info").textContent = Object.entries(row).map(([k, v]) => `${k.padEnd(12)} ${fmt(v)}`).join("\n");
}

// ---------------- wiring ----------------
$("run").addEventListener("change", (ev) => selectRun(ev.target.value).catch(console.error));
$("play").addEventListener("click", () => {
  if (state.replay && state.frame >= state.replay.frames.length - 1) setFrame(0);
  setPlaying(!state.playing);
});
$("stepBtn").addEventListener("click", () => { setPlaying(false); setFrame(state.frame + 1); });
$("frame").addEventListener("input", (ev) => { setPlaying(false); setFrame(Number(ev.target.value)); });
$("epoch").addEventListener("input", (ev) => loadEpochIdx(Number(ev.target.value)).catch(console.error));
$("follow").addEventListener("change", (ev) => {
  if (ev.target.checked && state.epochs.length) loadEpochIdx(state.epochs.length - 1).catch(console.error);
});
$("chart").addEventListener("click", (ev) => {
  if (!state.chart || !state.epochs.length) return;
  const c = $("chart"), rect = c.getBoundingClientRect();
  const px = ((ev.clientX - rect.left) * c.width) / rect.width;
  const e = ((px - state.chart.x0) / (state.chart.x1 - state.chart.x0)) * state.chart.eMax;
  let best = 0;
  state.epochs.forEach((ep, i) => { if (Math.abs(ep - e) < Math.abs(state.epochs[best] - e)) best = i; });
  loadEpochIdx(best).catch(console.error);
});
document.addEventListener("keydown", (ev) => {
  if (ev.target.tagName === "SELECT") return;
  if (ev.code === "Space") { ev.preventDefault(); $("play").click(); }
  else if (ev.code === "ArrowRight") { setPlaying(false); setFrame(state.frame + 1); }
  else if (ev.code === "ArrowLeft") { setPlaying(false); setFrame(state.frame - 1); }
});

setInterval(async () => {
  if (!$("follow").checked || !state.runId) return;
  try {
    const before = state.epochs.length;
    await refreshRun();
    if (state.epochs.length > before) await loadEpochIdx(state.epochs.length - 1);
    await loadRuns();
  } catch (err) {
    console.warn(err);
  }
}, 5000);

(async () => {
  try {
    const runs = await loadRuns();
    if (runs.length) await selectRun(runs[0].id);
  } catch (err) {
    console.error(err);
  }
  requestAnimationFrame(tick);
})();
