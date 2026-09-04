"""Zero-shot EEF-space eval client for the pi05_ur5e_eef probe (see train/README.md).

Difference from eval_client.py: the state/action space is end-effector space --
  state  (8,) = [tcp_pos(3), tcp_quat(4, wxyz), gripper obs(1)]
  action (4,) = [absolute tcp target pos(3), gripper cmd(1)]
(the server returns absolute positions after un-deltaing). Orientation is fixed
tool-down; the returned target pose is solved to joint targets with the repo IK.

Run the EEF-configured server first, then:
  python train/eval_client_eef.py --host 127.0.0.1 --episodes 3
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

# Keep the model's targets inside the task workspace (arm stays sane if the
# zero-shot head outputs nonsense); targets farther than this are clamped toward tcp.
MAX_STEP = 0.10  # m per control tick
Z_FLOOR, Z_CEIL = 0.66, 0.95  # table top ~0.65; stay above it


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    episodes: int = 3
    max_steps: int = 600  # 30 s at 20 Hz
    seed: int = 12345

    ur5e_src: str = str(pathlib.Path(__file__).resolve().parents[1] / "src")
    video_out_path: str | None = None


def clip_target(target: np.ndarray, tcp: np.ndarray) -> np.ndarray:
    """Clamp the model's target into the workspace and within MAX_STEP of the tcp."""
    t = target.copy()
    t[2] = np.clip(t[2], Z_FLOOR, Z_CEIL)
    delta = t - tcp
    dist = np.linalg.norm(delta)
    if dist > MAX_STEP:
        t = tcp + delta * (MAX_STEP / dist)
    return t


def run_episode(env, ik, client, rng, args: Args) -> tuple[bool, int, list]:
    obs = env.reset(rng)
    prompt = str(env.instruction)
    plan = collections.deque()
    frames = []

    for t in range(args.max_steps):
        cams = env.render()
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(cams["front_cam"], args.resize_size, args.resize_size)
        )
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(cams["wrist_cam"], args.resize_size, args.resize_size)
        )
        frames.append(img)

        if not plan:
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": np.concatenate(
                    [obs["tcp_pos"], obs["tcp_quat"], [obs["gripper"]]]
                ).astype(np.float32),
                "prompt": prompt,
            }
            action_chunk = client.infer(element)["actions"]
            assert len(action_chunk) >= args.replan_steps
            plan.extend(action_chunk[: args.replan_steps])

        action = np.asarray(plan.popleft(), dtype=np.float32)  # [target_pos(3), grip(1)]
        target = clip_target(action[:3], obs["tcp_pos"])
        q, pe, re, ok = ik.solve_with_restarts(
            obs["qpos"], rng, target_pos=target, target_z_dir=np.array([0.0, 0.0, -1.0])
        )
        joint_action = np.concatenate([q if ok else obs["qpos"], [np.clip(action[3], 0.0, 1.0)]])
        obs = env.step(joint_action.astype(np.float32))

        success, dist = env.check_success()
        if success:
            break

    return success, t + 1, frames


def main(args: Args) -> None:
    sys.path.insert(0, str(pathlib.Path(args.ur5e_src)))
    import env as _env  # noqa: E402  (collection repo: Ur5eEnv)

    if args.video_out_path:
        pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    env = _env.Ur5eEnv(width=320, height=240)
    ik = env.ik
    rng = np.random.default_rng(args.seed)
    client = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)

    n_success = 0
    try:
        for ep in range(args.episodes):
            rng_ep = np.random.default_rng(args.seed + ep)
            success, steps, frames = run_episode(env, ik, client, rng_ep, args)
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
