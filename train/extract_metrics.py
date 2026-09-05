"""Extract step metrics from an openpi train log into train/results/ (CSV + PNG).

openpi's train.py logs `Step N: grad_norm=..., loss=..., param_norm=...` every
log_interval steps but never the lr, so the lr column is reconstructed from the
schedule (openpi defaults unless overridden in the TrainConfig).

Usage:
  python train/extract_metrics.py <train.log> <config_name>
  e.g. python train/extract_metrics.py data/logs/train_scripted.log pi05_ur5e_lora_scripted
       -> train/results/train_metrics_<name>.csv + train_loss_<name>.png
"""
from __future__ import annotations

import csv
import math
import pathlib
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
# openpi CosineDecaySchedule defaults (TrainConfigs in our patches do not override)
WARMUP, PEAK, DECAY, END = 1000, 2.5e-5, 30000, 2.5e-6


def lr_at(step: int) -> float:
    init = PEAK / (WARMUP + 1)
    if step < WARMUP:
        return init + (PEAK - init) * step / WARMUP
    if step < DECAY:
        return 0.5 * (PEAK + END) + 0.5 * (PEAK - END) * math.cos(math.pi * (step - WARMUP) / (DECAY - WARMUP))
    return END


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    log, name = pathlib.Path(sys.argv[1]), sys.argv[2]
    pat = re.compile(r"Step (\d+): grad_norm=([\d.eE+-]+), loss=([\d.eE+-]+), param_norm=([\d.eE+-]+)")
    rows = []
    for line in log.read_text(errors="replace").splitlines():
        m = pat.search(line)
        if m:
            step, gn, loss, pn = m.groups()
            rows.append((int(step), float(gn), float(loss), float(pn), lr_at(int(step))))
    if not rows:
        sys.exit(f"no 'Step N: ...' lines found in {log}")

    RESULTS.mkdir(exist_ok=True)
    csv_path = RESULTS / f"train_metrics_{name}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "grad_norm", "loss", "param_norm", "lr"])
        w.writerows(rows)

    steps, loss, lrs = [r[0] for r in rows], [r[2] for r in rows], [r[4] for r in rows]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(steps, loss, lw=0.8)
    ax1.set_ylabel("loss")
    ax1.set_yscale("log")
    ax1.set_title(f"{name}  {rows[-1][0] // 1000}k steps")
    ax1.grid(alpha=0.3)
    ax2.plot(steps, lrs, color="tab:orange", lw=1.2)
    ax2.set_ylabel("lr")
    ax2.set_xlabel("step")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    png_path = RESULTS / f"train_loss_{name}.png"
    fig.savefig(png_path, dpi=150)
    print(f"{len(rows)} points -> {csv_path.name}, {png_path.name}; loss {loss[0]} -> {loss[-1]}")


if __name__ == "__main__":
    main()
