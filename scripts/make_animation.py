"""Render an animated GIF of training progress from a run's recorded replays.

Each selected epoch contributes a short clip of its fixed-seed evaluation episode,
so the same dish is shown at increasing training epochs.

    uv run python scripts/make_animation.py runs/<id> docs/training.gif
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from cellsim.storage import list_replay_epochs, load_replay, read_metrics, replay_path

SS = 2  # supersampling factor for smooth edges
W, DISH_H, CHART_H = 360, 360, 90
H = DISH_H + CHART_H
BG, DISH, RIM = (15, 17, 21), (19, 32, 42), (44, 74, 92)
FOOD, CELL, CELL_RIM = (126, 224, 126), (120, 190, 255), (207, 230, 255)
RAY, RAY_HIT = (36, 44, 52), (255, 214, 110)
TRAIN, EVAL, TEXT = (108, 182, 255), (240, 163, 94), (200, 204, 214)
TRAIL, FLASH = 50, 8


def font(size: int):
    return ImageFont.load_default(size=size * SS)


def food_states(rep: dict) -> list[list]:
    food, out = [p[:] for p in rep["food_init"]], []
    for f in rep["frames"]:
        for i, x, y in f["e"]:
            food[i] = [x, y]
        out.append([p[:] for p in food])
    return out


def draw_chart(d: ImageDraw.ImageDraw, rows: list[dict], epoch: int, e_max: int, y_max: float) -> None:
    x0, x1, y0, y1 = 36 * SS, (W - 12) * SS, (DISH_H + 8) * SS, (H - 16) * SS
    px = lambda e: x0 + (e / e_max) * (x1 - x0)  # noqa: E731
    py = lambda v: y1 - (v / y_max) * (y1 - y0)  # noqa: E731
    d.line([(x0, y1), (x1, y1)], fill=(38, 42, 51), width=SS)
    train = [(px(r["epoch"]), py(r["train_score"])) for r in rows if r.get("train_score") is not None]
    if len(train) > 1:
        d.line(train, fill=TRAIN, width=2 * SS)
    for r in rows:
        if r.get("eval_score") is not None:
            cx, cy = px(r["epoch"]), py(r["eval_score"])
            d.ellipse([cx - 2 * SS, cy - 2 * SS, cx + 2 * SS, cy + 2 * SS], fill=EVAL)
    d.line([(px(epoch), y0), (px(epoch), y1)], fill=(150, 154, 166), width=SS)
    small = font(10)
    d.text((4 * SS, y0 - 2 * SS), f"{y_max:.0f}", fill=(138, 144, 160), font=small)
    d.text((4 * SS, y1 - 9 * SS), "0", fill=(138, 144, 160), font=small)
    d.text((x0, (H - 13) * SS), "epoch", fill=(138, 144, 160), font=small)
    d.text((x1 - 22 * SS, (H - 13) * SS), str(e_max), fill=(138, 144, 160), font=small)
    d.text((x0 + 120 * SS, (H - 13) * SS), "train mean", fill=TRAIN, font=small)
    d.text((x0 + 185 * SS, (H - 13) * SS), "eval (recorded)", fill=EVAL, font=small)


def render(rep: dict, foods: list, fi: int, score: int, rows: list[dict], e_max: int, y_max: float) -> Image.Image:
    img = Image.new("RGB", (W * SS, H * SS), BG)
    d = ImageDraw.Draw(img)
    w = rep["world"]
    cx0, cy0, scale = W * SS / 2, DISH_H * SS / 2, (DISH_H * SS / 2 - 8 * SS) / w["radius"]
    X = lambda x: cx0 + x * scale  # noqa: E731
    Y = lambda y: cy0 - y * scale  # noqa: E731
    R = w["radius"] * scale
    d.ellipse([cx0 - R, cy0 - R, cx0 + R, cy0 + R], fill=DISH, outline=RIM, width=3 * SS)

    frames = rep["frames"]
    for k in range(max(1, fi - TRAIL + 1), fi + 1):
        a, b = frames[k - 1]["p"], frames[k]["p"]
        t = (k - (fi - TRAIL)) / TRAIL
        col = tuple(int(DISH[i] + (CELL[i] - DISH[i]) * 0.55 * t) for i in range(3))
        d.line([(X(a[0]), Y(a[1])), (X(b[0]), Y(b[1]))], fill=col, width=2 * SS)

    fr = max(2.5 * SS, w["food_r"] * scale)
    for x, y in foods[fi]:
        d.ellipse([X(x) - fr, Y(y) - fr, X(x) + fr, Y(y) + fr], fill=FOOD)
    for k in range(max(1, fi - FLASH + 1), fi + 1):
        age = (fi - k) / FLASH
        for i, *_ in frames[k]["e"]:
            x, y = foods[k - 1][i]
            rr = fr + age * 18 * SS
            col = tuple(int(DISH[j] + (FOOD[j] - DISH[j]) * (1 - age)) for j in range(3))
            d.ellipse([X(x) - rr, Y(y) - rr, X(x) + rr, Y(y) + rr], outline=col, width=2 * SS)

    f = frames[fi]
    px_, py_ = f["p"]
    for j in range(w["n_rays"]):
        ang = f["h"] + j * 2 * math.pi / w["n_rays"]
        dist = f["r"][j]
        hit = dist < w["ray_len"] - 1e-6
        d.line([(X(px_), Y(py_)), (X(px_ + dist * math.cos(ang)), Y(py_ + dist * math.sin(ang)))],
               fill=RAY_HIT if hit else RAY, width=(2 if hit else 1) * SS)
    cr = max(5 * SS, w["cell_r"] * scale)
    d.ellipse([X(px_) - cr, Y(py_) - cr, X(px_) + cr, Y(py_) + cr], fill=CELL, outline=CELL_RIM, width=SS)
    d.line([(X(px_), Y(py_)),
            (X(px_ + 1.8 * w["cell_r"] * math.cos(f["h"])), Y(py_ + 1.8 * w["cell_r"] * math.sin(f["h"])))],
           fill=CELL_RIM, width=SS)

    big = font(15)
    d.text((10 * SS, 8 * SS), f"epoch {rep['epoch']}", fill=TEXT, font=big)
    d.text((10 * SS, 28 * SS), f"score {score}", fill=TEXT, font=big)

    draw_chart(d, rows, rep["epoch"], e_max, y_max)
    return img.resize((W, H), Image.LANCZOS)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--epochs", type=int, nargs="*", default=[0, 10, 20, 30, 50, 100, 150, 250, 400, 500])
    p.add_argument("--steps", type=int, default=180, help="episode steps shown per epoch")
    p.add_argument("--stride", type=int, default=3, help="show every Nth step")
    p.add_argument("--ms", type=int, default=50, help="milliseconds per frame")
    args = p.parse_args()

    have = set(list_replay_epochs(args.run_dir))
    epochs = [e for e in args.epochs if e in have]
    rows = read_metrics(args.run_dir)
    e_max = max(r["epoch"] for r in rows)
    y_max = max(v for r in rows for v in (r.get("train_score"), r.get("eval_score")) if v is not None) * 1.1

    images: list[Image.Image] = []
    for e in epochs:
        rep = load_replay(replay_path(args.run_dir, e))
        foods = food_states(rep)
        score, scores = 0, []
        for f in rep["frames"]:
            score += len(f["e"])
            scores.append(score)
        upto = [r for r in rows if r["epoch"] <= e]
        for fi in range(0, min(args.steps, len(rep["frames"]) - 1) + 1, args.stride):
            images.append(render(rep, foods, fi, scores[fi], upto, e_max, y_max))
        images.extend([images[-1]] * 6)  # brief hold on the last frame of each epoch
        print(f"epoch {e}: score after {args.steps} steps = {scores[min(args.steps, len(scores) - 1)]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pal = images[len(images) // 2].quantize(colors=64, method=Image.Quantize.MEDIANCUT)
    frames = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in images]
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=args.ms, loop=0, optimize=True)
    print(f"wrote {args.out}: {len(frames)} frames, {args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
