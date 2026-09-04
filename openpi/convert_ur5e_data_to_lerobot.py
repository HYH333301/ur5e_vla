"""
Convert UR5e + Robotiq 2F-85 pick-place episodes (MuJoCo collection repo) to a
LeRobot dataset for openpi fine-tuning.

Input: one HDF5 per episode as written by the collection repo (scripts/collect.py
and scripts/teleop_collect.py):

  observations/qpos (T+1, 6)            arm joint positions
  observations/gripper (T+1,)           gripper driver joint position
  observations/image_{front,side,wrist}_cam (T+1,)   JPEG-encoded frames
  action (T, 7)                          6 joint position targets + gripper cmd in [0, 1]
  attrs: instruction, success, ...

Output features (matching the `pi05_ur5e` config in src/openpi/training/config.py):
  image (front camera), side_image, wrist_image   (240, 320, 3)
  state  = [6 joint positions, gripper position]  (7,)
  actions = action                                (7,)
  task   = instruction                            (per-episode task string)

Each episode contributes T frames (obs_t + action_t pairs); the final observation
has no action and is dropped, like the LIBERO example converter. Unsuccessful
episodes are skipped (pass --include-failures to keep them).

Usage (from the openpi project root, in the openpi environment):
  uv run examples/ur5e/convert_ur5e_data_to_lerobot.py \
      --data-dir D:/code/ur5e_vla/data/ur5e_pickplace D:/code/ur5e_vla/data/ur5e_teleop \
      --repo-id hyh1234/ur5e_vla_lerobot

Add --push-to-hub to upload to the Hugging Face Hub (requires HF_TOKEN with write
access to the repo). The resulting dataset is saved under $HF_LEROBOT_HOME.
"""

import io
import pathlib
import shutil

import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
import tyro

CAMS = {"front": "image", "side": "side_image", "wrist": "wrist_image"}
WIDTH, HEIGHT = 320, 240


def decode_jpeg(raw) -> np.ndarray:
    img = np.asarray(Image.open(io.BytesIO(bytes(raw))))
    assert img.shape == (HEIGHT, WIDTH, 3), f"unexpected frame shape {img.shape}"
    return img


def main(
    data_dir: list[str],
    *,
    repo_id: str = "hyh1234/ur5e_vla_lerobot",
    fps: int = 20,
    push_to_hub: bool = False,
    include_failures: bool = False,
    image_writer_threads: int = 8,
    image_writer_processes: int = 0,  # 0 on Windows: multiprocessing save is slow to spawn
):
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        print(f"removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="ur5e",
        fps=fps,
        features={
            **{
                key: {"dtype": "image", "shape": (HEIGHT, WIDTH, 3), "names": ["height", "width", "channel"]}
                for key in CAMS.values()
            },
            "state": {"dtype": "float32", "shape": (7,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )

    total = converted = skipped = 0
    for root in data_dir:
        for path in sorted(pathlib.Path(root).glob("episode_*.hdf5")):
            total += 1
            with h5py.File(path, "r") as f:
                if not include_failures and not bool(f.attrs.get("success", True)):
                    print(f"  [skip] {path.name}: not successful")
                    skipped += 1
                    continue
                actions = f["action"][:].astype(np.float32)  # (T, 7)
                qpos = f["observations/qpos"][:].astype(np.float32)  # (T+1, 6)
                grip = f["observations/gripper"][:].astype(np.float32)  # (T+1,)
                state = np.concatenate([qpos, grip[:, None]], axis=1)  # (T+1, 7)
                task = str(f.attrs["instruction"])
                images = {cam: f[f"observations/image_{cam}_cam"] for cam in CAMS}

                for t in range(len(actions)):
                    dataset.add_frame(
                        {
                            **{key: decode_jpeg(images[cam][t]) for cam, key in CAMS.items()},
                            "state": state[t],
                            "actions": actions[t],
                            "task": task,
                        }
                    )
                dataset.save_episode()
                converted += 1
                print(f"  [ok] {path.name}: {len(actions)} frames  '{task}'")

    print(f"\ndone: {converted}/{total} episodes converted ({skipped} skipped)")
    print(f"dataset saved to {HF_LEROBOT_HOME / repo_id}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["ur5e", "robotiq-2f85", "pick-place", "mujoco"],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
