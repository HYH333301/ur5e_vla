"""
Sanity-check a converted UR5e LeRobot dataset the way openpi's data loader will
read it (delta_timestamps over the "actions" key), and optionally compare
frame t of a source HDF5 episode against the dataset.

Usage:
  HF_LEROBOT_HOME=D:/code/ur5e_vla/data/lerobot \
  D:/code/ur5e_vla/.venv-lerobot/Scripts/python examples/ur5e/check_dataset.py \
      --repo-id hyh1234/ur5e_vla_lerobot \
      --hdf5 D:/code/ur5e_vla/data/ur5e_pickplace/episode_0000.hdf5 --frame 100
"""

import io
import pathlib

import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from PIL import Image
import tyro

HORIZON = 10  # pi05_ur5e action horizon


def main(repo_id: str, hdf5: str | None = None, frame: int = 100):
    meta = LeRobotDatasetMetadata(repo_id)
    fps = meta.fps
    print(f"fps={fps}  episodes={meta.total_episodes}  frames={meta.total_frames}")

    dataset = LeRobotDataset(repo_id, delta_timestamps={"actions": [i / fps for i in range(HORIZON)]})
    sample = dataset[frame]
    for k, v in sorted(sample.items()):
        print(f"  {k:18s} {str(np.asarray(v).shape):16s} {np.asarray(v).dtype}")

    img = np.asarray(sample["image"])
    assert img.shape == (3, 240, 320) and img.dtype == np.float32, "bad image format"
    assert np.asarray(sample["state"]).shape == (7,)
    assert np.asarray(sample["actions"]).shape == (HORIZON, 7)
    assert isinstance(sample["task"], str) and sample["task"], "missing task string"

    if hdf5:
        with h5py.File(hdf5) as f:
            gt_actions = f["action"][:]
            gt_state = np.concatenate([f["observations/qpos"][frame], [f["observations/gripper"][frame]]])
        err_a = np.abs(np.asarray(sample["actions"]) - gt_actions[frame : frame + HORIZON]).max()
        err_s = np.abs(np.asarray(sample["state"]) - gt_state).max()
        print(f"\nframe {frame}: state err={err_s:.2e}  action-seq err={err_a:.2e}  task={sample['task']!r}")
        assert err_s < 1e-6 and err_a < 1e-6, "values differ from source HDF5"

        # JPEG round-trip of the same frame: allow small lossy-codec differences
        with h5py.File(hdf5) as f:
            raw = np.asarray(Image.open(io.BytesIO(bytes(f["observations/image_front_cam"][frame]))))
        got = (np.clip(np.asarray(sample["image"]).transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
        print(f"image mean abs diff vs re-decoded JPEG: {np.abs(got.astype(int) - raw.astype(int)).mean():.2f} (0-255)")

    print("\nOK")


if __name__ == "__main__":
    tyro.cli(main)
