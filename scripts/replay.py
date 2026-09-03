"""QC an episode HDF5: print stats and dump sampled 3-camera contact sheets.

Example:
    python scripts/replay.py data/ur5e_pickplace/episode_0000.hdf5 --samples 8
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path
import sys

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from expert import PHASES  # noqa: E402


def phase_name(p: int) -> str:
    return "teleop" if p < 0 else PHASES[p]  # teleop episodes store phase=-1


CAMS = ("front_cam", "side_cam", "wrist_cam")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("episode", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "out" / "replay")
    ap.add_argument("--samples", type=int, default=8)
    args = ap.parse_args()

    with h5py.File(args.episode, "r") as f:
        T = int(f.attrs["n_actions"])
        print(f"episode: {args.episode.name}")
        print(f"instruction: {f.attrs['instruction']!r}")
        print(f"n_actions: {T}  (~{T / 20:.1f}s @ 20Hz)")
        action = f["action"][:]
        phase = f["phase"][:]
        qpos = f["observations/qpos"][:]
        cube = f["observations/cube_pos"][:]
        grip = f["observations/gripper"][:]
        counts = {phase_name(p): int((phase[:T] == p).sum()) for p in np.unique(phase[:T])}
        print(f"phase ticks: {counts}")
        print(f"action range: [{action.min():.3f}, {action.max():.3f}]")
        print(f"grip driver q: open={grip.min():.3f} closed={grip.max():.3f}")
        print(f"cube z: start={cube[0, 2]:.3f} min={cube[:, 2].min():.3f} "
              f"max={cube[:, 2].max():.3f} end={cube[-1, 2]:.3f}")
        print(f"cube xy drift while lifted: "
              f"{np.ptp(cube[grip.argmax():, 0], axis=0):.3f}, "
              f"{np.ptp(cube[grip.argmax():, 1], axis=0):.3f}")

        ts = np.linspace(0, T, args.samples, dtype=int)
        args.out.mkdir(parents=True, exist_ok=True)
        for t in ts:
            row = []
            for cam in CAMS:
                jpg = f[f"observations/image_{cam}"][t]
                row.append(np.asarray(Image.open(io.BytesIO(bytes(jpg)))))
            sheet = np.concatenate(row, axis=1)
            p = args.out / f"{args.episode.stem}_t{t:04d}_{phase_name(phase[t])}.png"
            Image.fromarray(sheet).save(p)
        print(f"saved {len(ts)} contact sheets to {args.out}")


if __name__ == "__main__":
    main()
