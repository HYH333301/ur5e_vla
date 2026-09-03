"""Collect scripted pick-place demonstrations into per-episode HDF5 files.

Example:
    python scripts/collect.py --episodes 50 --out data/ur5e_pickplace --seed 0
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from env import Ur5eEnv, EpisodeDone, CAMS  # noqa: E402
from expert import PickPlaceExpert, ExpertFailure, PHASES, PH  # noqa: E402
from recorder import EpisodeRecorder, encode_jpeg  # noqa: E402


def check_phase_exit(env: Ur5eEnv, phase: int) -> None:
    """Sanity checks when the expert finishes a phase; abort bad episodes."""
    if phase == PH["close"]:
        q_drv = env.data.qpos[env.grip_qadr]
        if q_drv > 0.6:  # fully closed = nothing between the fingers
            raise EpisodeDone(f"grasp failed: driver closed fully ({q_drv:.2f})")
    elif phase in (PH["lift"], PH["carry"]):
        # transport only: at 'place' the cube is SUPPOSED to end up at table level
        if env.cube_pos()[2] < 0.64:
            raise EpisodeDone("cube was dropped during transport")


def run_episode(env: Ur5eEnv, expert: PickPlaceExpert, rng, rec: EpisodeRecorder) -> tuple[bool, float]:
    obs = env.reset(rng)
    actions, phases = expert.plan(env.cube_xy, env.tgt_xy, obs["qpos"])
    rec.add_obs(obs, {c: encode_jpeg(f) for c, f in env.render().items()}, phases[0])

    prev_phase = phases[0]
    for t, action in enumerate(actions):
        if phases[t] != prev_phase:
            check_phase_exit(env, prev_phase)
            prev_phase = phases[t]
        rec.add_action(action)
        obs = env.step(action)
        rec.add_obs(obs, {c: encode_jpeg(f) for c, f in env.render().items()}, phases[t])

    success, dist = env.check_success()
    return success, dist


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=10, help="successful episodes to save")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "ur5e_pickplace")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=-1,
                    help="first episode index; -1 = continue after the highest "
                         "existing episode index in --out")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--max-attempts", type=int, default=0, help="0 = 5x episodes")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    env = Ur5eEnv(width=args.width, height=args.height)
    expert = PickPlaceExpert(env.ik, rng)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.start < 0:  # never overwrite: continue after the highest existing index
        nums = [int(p.stem.rsplit("_", 1)[1]) for p in args.out.glob("episode_*.hdf5")
                if p.stem.rsplit("_", 1)[1].isdigit()]
        args.start = max(nums) + 1 if nums else 0
    print(f"新回合编号从 episode_{args.start:04d} 开始，输出目录 {args.out}")

    max_attempts = args.max_attempts or args.episodes * 5
    saved, attempts = 0, 0
    t0 = time.time()
    try:
        while saved < args.episodes and attempts < max_attempts:
            attempts += 1
            rec = EpisodeRecorder(CAMS)
            try:
                success, dist = run_episode(env, expert, rng, rec)
            except (ExpertFailure, EpisodeDone) as e:
                print(f"[attempt {attempts}] aborted: {e}")
                continue
            idx = args.start + saved
            path = args.out / f"episode_{idx:04d}.hdf5"
            if success:
                rec.save(path, {
                    "success": True,
                    "instruction": env.instruction,
                    "cube_rgba": env.cube_rgba,
                    "target_rgba": env.target_rgba,
                })
                saved += 1
                size_mb = path.stat().st_size / 1e6
                print(f"[attempt {attempts}] saved {path.name}: T={rec.n_actions} "
                      f"dist={dist:.3f}m {size_mb:.1f}MB  '{env.instruction}'")
            else:
                print(f"[attempt {attempts}] completed but not successful "
                      f"(dist={dist:.3f}m), discarded")
    finally:
        env.close()

    dt = time.time() - t0
    print(f"\ndone: {saved}/{attempts} episodes saved to {args.out} in {dt:.1f}s")


if __name__ == "__main__":
    main()
