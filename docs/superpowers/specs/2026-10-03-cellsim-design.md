# CellSim — Design Spec

Date: 2026-10-03
Status: approved in brainstorming, pending spec review

## 1. Goal

Simulate a single-cell organism controlled by a small neural network that must find
as much food as possible within a fixed time. No dataset exists: the network is
trained from scratch with reinforcement learning (PPO, PyTorch) in a randomized
simulation. Intermediate behavior is recorded to disk during training and can be
replayed later in a browser canvas viewer, stepping through epochs to watch the
behavior evolve.

Success: epoch 0 (random policy) eats ~1–3 pellets per episode; a trained policy
visibly steers toward food, sweeps the dish when nothing is in view, avoids the wall,
and scores well above 20 pellets per episode.

## 2. World (the petri dish)

All quantities are in dish units; one step is one simulation tick.

| Parameter | Value |
|---|---|
| Dish | circle, center (0,0), radius `R = 1.0` |
| Cell radius | `0.03` |
| Food radius | `0.015` |
| Food count | `20` pellets, constant |
| Episode length | `500` steps |
| Eating | pellet eaten when `dist(cell, food) < cell_r + food_r` |
| Respawn | eaten pellet immediately respawns uniformly in disk of radius `0.95`, at least `0.1` from the cell |
| Initial state | cell at a uniform random position in radius `0.8`, random heading, zero velocity; food uniform in radius `0.95` |

**Actions** (2 continuous values, policy output clipped to `[-1, 1]`):
- `turn` → `heading += turn * 0.2` rad
- `thrust` → mapped to `[0, 1]` via `(thrust + 1) / 2` (no reverse)

**Physics** (viscous medium, per step):
```
vel = 0.85 * vel + 0.01 * thrust01 * [cos(heading), sin(heading)]
pos = pos + vel
if |pos| > R - cell_r:  pos projected back onto circle of radius R - cell_r,
                        radial velocity component removed, wall_contact = 1
```
Terminal speed ≈ 0.067 per step, so crossing the dish takes ~30 steps.

**Reward**: `+1` per pellet eaten that step, `-0.01` per step with `wall_contact`.
**Score** (shown to the user): pellets eaten in the episode (wall penalty excluded).

**Episodes** never terminate early. At step 500 the env sets `done = 1` and resets
that dish in place (auto-reset), so batched rollouts are continuous.

## 3. Observations (37 floats)

- **16 rays**, evenly spaced over 360°, angles relative to heading (ray 0 points forward),
  max length `0.5`, cast from the cell center. Each ray gives 2 channels:
  - `food`: `1 - d/0.5` for the nearest food circle hit, `0` if none in range
  - `wall`: `1 - d/0.5` for the dish wall hit, `0` if beyond range
  → 32 values
- **Proprioception** (5): forward speed, lateral speed (velocity in the cell's frame,
  each scaled by `1/0.067`), last `turn`, last `thrust` (both in `[-1,1]`),
  time remaining fraction `1 - t/500`.

Ray–circle intersection and ray–wall (ray inside circle) intersection are computed
analytically and fully vectorized over `(envs, rays, food)`.

## 4. Neural network

Actor-critic with **separate** networks (no shared trunk), ~10k parameters total.

```
obs(37) → RunningNorm ─┬─ Actor:  Linear 37→64, tanh, Linear 64→64, tanh, Linear 64→2  → μ
                       │          learned state-independent log_std (2,), init 0
                       │          π(a|s) = Normal(μ, exp(log_std))
                       └─ Critic: Linear 37→64, tanh, Linear 64→64, tanh, Linear 64→1 → V(s)
```
- Orthogonal init (gain √2 on hidden layers, 0.01 on actor head, 1.0 on critic head), zero bias.
- Sampled actions are stored unclipped for log-prob computation; the env receives clipped actions.
- Deterministic (eval) action = `clip(μ, -1, 1)`.
- `RunningNorm`: running mean/variance of observations, updated from rollout data,
  observations clipped to `[-10, 10]` after normalization; its state is saved in checkpoints.
- No recurrent memory in v1. (A GRU variant may be added later if the feed-forward policy plateaus.)

## 5. Training (PPO)

**Epoch** = one PPO iteration: rollout collection + update. The viewer's slider indexes epochs.

| Hyperparameter | Default |
|---|---|
| Parallel envs | 1024 (all on one device, GPU if available) |
| Rollout length | 128 steps/env → 131,072 samples/epoch |
| Epochs | 500 (~65M env steps) |
| γ / GAE λ | 0.99 / 0.95 |
| Update passes | 4 per epoch |
| Minibatch | 8192 |
| Optimizer | Adam, lr 3e-4, linear decay to 0 over all epochs |
| Clip ε | 0.2 |
| Value loss coef | 0.5 (unclipped MSE) |
| Entropy coef | 0.0 |
| Max grad norm | 0.5 |
| Advantage norm | per minibatch |
| Reward scaling | divide by running std of discounted return |

GAE bootstraps through rollout boundaries with `V(s_T)` and cuts at `done`. Because
episodes end only by time limit, the time-remaining observation keeps value
estimates consistent across the cut.

**Training metrics per epoch**: mean score of episodes completed during the rollout,
policy loss, value loss, entropy, approx KL, clip fraction, steps/sec, wall time.

## 6. Recording and run artifacts

Every `record_every` epochs (default 10), plus epoch 0 (before any update) and the
final epoch, the trainer runs **one deterministic evaluation episode** on a separate
single-env instance seeded with a fixed `eval_seed` (default 0) — identical starting
dish every time, so epochs are directly comparable.

Run directory `runs/<YYYYmmdd-HHMMSS>/`:
```
config.json                       # full config used for the run
metrics.jsonl                     # one JSON object per epoch (+ eval_score on recorded epochs)
replays/epoch_0000.json.gz        # recorded evaluation episodes
checkpoints/epoch_0000.pt         # model state, RunningNorm state, optimizer, config, epoch
checkpoints/latest.pt             # copy of most recent checkpoint
```

**Replay format** (gzipped JSON):
```json
{
  "epoch": 50,
  "score": 23,
  "world": {"radius": 1.0, "cell_r": 0.03, "food_r": 0.015, "ray_len": 0.5, "n_rays": 16},
  "food_init": [[x, y], ...],                  // 20 positions
  "frames": [
    {"p": [x, y], "h": heading, "r": [d0..d15], "e": [[idx, nx, ny], ...]}
  ]
}
```
- `r`: per-ray nearest hit distance (food or wall, whichever is closer; `ray_len` if none), for drawing vision.
- `e`: eat events this frame — food index eaten and its respawn position. The viewer
  reconstructs food state from `food_init` + events, which keeps files small.
- Coordinates are rounded to 4 decimals.

All artifact writes go to a temp file and are atomically renamed, so a reader never
sees a partial file. `Ctrl+C` finishes by saving a checkpoint. `--resume <run_dir>`
reloads `latest.pt` and continues appending to the same run.

## 7. Viewer

`python -m cellsim.serve [--runs-dir runs] [--port 8000]` starts a FastAPI app serving
the static viewer and a read-only JSON API:

| Endpoint | Returns |
|---|---|
| `GET /api/runs` | list of run ids, newest first, with latest epoch and best eval score |
| `GET /api/runs/{id}/metrics` | array of metrics rows |
| `GET /api/runs/{id}/replays` | sorted list of recorded epoch numbers |
| `GET /api/runs/{id}/replays/{epoch}` | replay JSON (gz bytes passed through with `Content-Encoding: gzip`) |

Unknown run/epoch → 404. Run ids are validated against the directory listing (no path traversal).

**Page** (`viewer/index.html` + `viewer/app.js`, plain JS, no build step, no CDN):
- Dish canvas: dish circle, food dots (flash on eat), cell blob with heading marker,
  optional faint rays brightening on hit, optional trail (last ~50 positions),
  HUD with score and `step / 500`.
- Controls: run dropdown, epoch slider snapping to recorded epochs, play/pause,
  single-step, speed (0.25×–8×), frame scrubber, rays/trail toggles.
- Metrics chart (canvas): train mean score and eval score vs epoch; clicking a point
  loads that epoch's replay.
- "Follow latest" toggle: polls every 5 s and auto-loads the newest replay.

## 8. Project layout

```
pyproject.toml          # deps: torch, numpy, fastapi, uvicorn; dev: pytest, httpx
cellsim/config.py       # Config dataclass: world + PPO + run params; to/from JSON
cellsim/env.py          # BatchedDishEnv(n_envs, cfg, device, seed): reset(), step(actions) -> obs, reward, done, info
cellsim/model.py        # ActorCritic, RunningNorm
cellsim/ppo.py          # RolloutBuffer, compute_gae(), ppo_update() — no I/O
cellsim/recorder.py     # record_episode(model, norm, cfg) -> replay dict; save_replay(path, dict)
cellsim/train.py        # CLI entry: epoch loop, metrics, checkpoints, recording, resume, SIGINT
cellsim/serve.py        # FastAPI app + CLI
viewer/index.html, viewer/app.js
tests/test_env.py, tests/test_ppo.py, tests/test_smoke.py
runs/                   # output, gitignored
```

Boundaries: `env.py` and `ppo.py` do no file I/O; `recorder.py`/`train.py` contain no physics;
`serve.py` only reads the run directory.

CLI:
```
python -m cellsim.train --epochs 500 --envs 1024 --record-every 10 [--device cuda] [--resume runs/<id>]
python -m cellsim.serve
```

## 9. Testing

- `test_env.py` (hand-placed scenarios, CPU):
  - ray detects food at known distance and correct ray index; wall distance correct
  - eating yields +1 reward, respawn obeys radius and min-distance rules
  - cell cannot leave the dish; wall contact produces the penalty
  - auto-reset at step 500 with `done = 1`
  - same seed → identical trajectories
- `test_ppo.py`: GAE matches a hand-computed 3-step example including a `done` cut;
  one `ppo_update` on a fixed batch reduces the surrogate loss.
- `test_smoke.py`: 2 epochs with tiny config on CPU in a temp dir → replay, metrics,
  checkpoint exist and load; `--resume` continues epoch numbering; server endpoints
  respond correctly via FastAPI `TestClient`, including 404s.
- Manual acceptance: full default run; viewer shows epoch 0 flailing, rising score
  curve, purposeful foraging in late epochs.

## 10. Out of scope (v1)

Recurrent memory, energy/metabolism, hazards/obstacles, multiple cells, side-by-side
replay comparison, live frame streaming.
