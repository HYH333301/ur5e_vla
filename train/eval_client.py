"""
Evaluate an openpi policy on the UR5e + Robotiq 2F-85 MuJoCo pick-place task.

This is the inference client: it runs the MuJoCo environment locally (from the
collection repo) and queries a policy server started with scripts/serve_policy.py
(see examples/ur5e/README.md for the full recipe).

Run a server first, e.g. on the training machine (tyro subcommand syntax; see
train/README.md section 4 for the full recipe incl. the SSH tunnel):
  uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
      --policy.config pi05_ur5e_lora --policy.dir checkpoints/pi05_ur5e_lora/<exp>/<step>

Then run this script locally (needs the collection env: mujoco, numpy, h5py,
pillow + `pip install openpi-client`):
  python train/eval_client.py --host 127.0.0.1 --episodes 20
"""

from __future__ import annotations

import collections
import dataclasses
import pathlib
import sys

import numpy as np
import tyro

import openpi_client.image_tools as image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy


@dataclasses.dataclass
class Args:
    # Policy server address
    host: str = "127.0.0.1"
    port: int = 8000
    # Images are letterboxed to this size before being sent (matches training)
    resize_size: int = 224
    # Re-query the policy every N env steps (must be <= action horizon, 10)
    replan_steps: int = 5

    # Evaluation
    episodes: int = 20
    max_steps: int = 600  # 30 s at 20 Hz
    seed: int = 12345

    # Path to the `src` directory of the UR5e collection repo
    ur5e_src: str = str(pathlib.Path(__file__).resolve().parents[1] / "src")
    # Optional directory for per-episode replay videos (front camera)
    video_out_path: str | None = None


def run_episode(env, client: _websocket_client_policy.WebsocketClientPolicy, args: Args, rng) -> tuple[bool, int, list]:
    obs = env.reset(rng)
    prompt = str(env.instruction)
    plan = collections.deque()
    frames = []
    successes = 0

    for t in range(args.max_steps):
        cams = env.render()
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(cams["front_cam"], args.resize_size, args.resize_size)
        )
        side_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(cams["side_cam"], args.resize_size, args.resize_size)
        )
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(cams["wrist_cam"], args.resize_size, args.resize_size)
        )
        frames.append(img)

        if not plan:
            element = {
                "observation/image": img,
                "observation/side_image": side_img,
                "observation/wrist_image": wrist_img,
                "observation/state": np.concatenate([obs["qpos"], [obs["gripper"]]]).astype(np.float32),
                "prompt": prompt,
            }
            action_chunk = client.infer(element)["actions"]
            assert len(action_chunk) >= args.replan_steps, (
                f"want >= {args.replan_steps} actions per chunk, got {len(action_chunk)}"
            )
            plan.extend(action_chunk[: args.replan_steps])

        action = np.asarray(plan.popleft(), dtype=np.float32)
        obs = env.step(action)

        success, dist = env.check_success()
        if success:
            successes += 1
            break

    return successes > 0, t + 1, frames


def main(args: Args) -> None:
    sys.path.insert(0, str(pathlib.Path(args.ur5e_src)))
    import env as _env  # noqa: E402  (collection repo: Ur5eEnv)

    if args.video_out_path:
        pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    env = _env.Ur5eEnv(width=320, height=240)
    client = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)

    n_success = 0
    try:
        for ep in range(args.episodes):
            rng = np.random.default_rng(args.seed + ep)
            success, steps, frames = run_episode(env, client, args, rng)
            n_success += success
            print(f"[episode {ep}] {'SUCCESS' if success else 'FAILURE'} after {steps} steps "
                  f"({steps / 20:.1f} s)   '{env.instruction}'")
            if args.video_out_path and frames:
                import imageio

                tag = "success" if success else "failure"
                imageio.mimwrite(pathlib.Path(args.video_out_path) / f"rollout_{ep:03d}_{tag}.mp4",
                                 frames, fps=20)
    finally:
        env.close()

    print(f"\nsuccess rate: {n_success}/{args.episodes} ({100 * n_success / args.episodes:.1f}%)")


if __name__ == "__main__":
    main(tyro.cli(Args))
