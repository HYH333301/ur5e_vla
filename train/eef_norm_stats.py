"""Compute EEF-space norm stats for the zero-shot pi05_base probe (pi05_ur5e_eef).

state (8,)  = [tcp_pos(3), tcp_quat(4, wxyz), gripper obs (1)]
action (4,) = [tcp_pos[t+1] - tcp_pos[t] (3), gripper cmd (1)]

The LeRobot dataset stores joint-space data only, so the stats are computed from the
original HDF5 episodes on the HF hub (hyh1234/ur5e_vla). Output mirrors openpi's
norm_stats.json layout; upload it to the base checkpoint dir on the cloud:
  <pi05_base>/assets/hyh1234/ur5e_vla_eef/norm_stats.json

Usage (in .venv-lerobot):
  HF_ENDPOINT=https://hf-mirror.com python train/eef_norm_stats.py [--n 8] [--out data/eef_norm_stats/norm_stats.json]
"""
from __future__ import annotations

import dataclasses
import io
import json
import pathlib

import h5py
import numpy as np
import tyro
from huggingface_hub import HfApi, hf_hub_download

REPO = "hyh1234/ur5e_vla"


def hf_token() -> str:
    env = pathlib.Path(__file__).resolve().parents[1] / ".env"
    for line in io.open(env, encoding="utf-8"):
        if line.startswith("HF_TOKEN="):
            return line.strip().split("=", 1)[1]
    raise SystemExit("HF_TOKEN not found in .env")


@dataclasses.dataclass
class Args:
    episodes_per_source: int = 8
    out: str = "data/eef_norm_stats/norm_stats.json"


def episode_stats(path: pathlib.Path):
    """Sample states/actions of one episode in EEF space."""
    with h5py.File(path, "r") as f:
        tcp = np.asarray(f["observations/tcp_pos"], dtype=np.float64)  # (T+1, 3)
        quat = np.asarray(f["observations/tcp_quat"], dtype=np.float64)  # (T+1, 4) wxyz
        grip_obs = np.asarray(f["observations/gripper"], dtype=np.float64).reshape(-1)  # (T+1,)
        grip_cmd = np.asarray(f["action"][:, 6], dtype=np.float64)  # (T,)

    states = np.concatenate([tcp, quat, grip_obs[:, None]], axis=1)  # (T+1, 8)
    dpos = tcp[1:] - tcp[:-1]  # (T, 3)
    actions = np.concatenate([dpos, grip_cmd[:, None]], axis=1)  # (T, 4)
    return states, actions


def main(args: Args) -> None:
    api = HfApi(token=hf_token())
    files = api.list_repo_files(REPO, repo_type="dataset")
    eps = sorted(f for f in files if f.endswith(".hdf5"))
    per = {src: [f for f in eps if f.startswith(src)] for src in ("scripted", "teleop")}
    pick = []
    for src, lst in per.items():
        idx = np.linspace(0, len(lst) - 1, min(args.episodes_per_source, len(lst))).astype(int)
        pick += [lst[i] for i in idx]
    print(f"sampling {len(pick)} episodes: {[p.split('/')[-1] for p in pick]}")

    states, actions = [], []
    for f in pick:
        local = hf_hub_download(REPO, f, repo_type="dataset", token=hf_token())
        s, a = episode_stats(pathlib.Path(local))
        states.append(s)
        actions.append(a)
    states = np.concatenate(states)
    actions = np.concatenate(actions)
    print(f"frames: states {states.shape}, actions {actions.shape}")

    def block(x: np.ndarray) -> dict:
        return {
            "mean": x.mean(0).tolist(),
            "std": x.std(0).tolist(),
            "q01": np.quantile(x, 0.01, axis=0).tolist(),
            "q99": np.quantile(x, 0.99, axis=0).tolist(),
        }

    out = {"norm_stats": {"state": block(states), "actions": block(actions)}}
    dest = pathlib.Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out))
    print(f"wrote {dest}")
    print("state mean:", np.round(states.mean(0), 4).tolist())
    print("action mean:", np.round(actions.mean(0), 4).tolist())
    print("action q01: ", np.round(np.quantile(actions, 0.01, axis=0), 4).tolist())
    print("action q99: ", np.round(np.quantile(actions, 0.99, axis=0), 4).tolist())


if __name__ == "__main__":
    main(tyro.cli(Args))
