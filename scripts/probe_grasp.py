"""Empirically find the TCP z where the 2F-85 straddles the cube correctly.

Sweeps descent heights with long-enough settle times, closes the gripper at
each, and reports the driver stall angle + whether the cube ends up held.
The right GRASP_Z stalls the driver near the cube-width opening (~0.33 rad
for 50 mm) without displacing the cube.
"""
from __future__ import annotations

from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from env import Ur5eEnv  # noqa: E402
from expert import TOOL_DOWN, HOVER_Z  # noqa: E402

CUBE_XY = np.array([-0.55, 0.0])


def settle(env, seconds: float):
    for _ in range(int(seconds / 0.002)):
        mujoco.mj_step(env.model, env.data)


def main():
    rng = np.random.default_rng(0)
    env = Ur5eEnv(width=64, height=48)
    env.reset(rng)
    q_home = env.get_obs()["qpos"].copy()

    for z in np.arange(0.615, 0.672, 0.005):
        env.reset(rng)
        # pin the cube at a fixed spot
        env.data.qpos[env.obj_qadr["cube"]:env.obj_qadr["cube"] + 3] = (*CUBE_XY, 0.6255)
        env.data.qpos[env.obj_qadr["cube"] + 3:env.obj_qadr["cube"] + 7] = (1, 0, 0, 0)
        env.data.qvel[env.obj_vadr["cube"]:env.obj_vadr["cube"] + 6] = 0
        mujoco.mj_forward(env.model, env.data)
        cube0 = env.obj_pos("cube")

        # descend through the hover point exactly like the expert would
        q_h, *_ = env.ik.solve_with_restarts(
            q_home, rng, target_pos=np.array([*CUBE_XY, HOVER_Z]), target_z_dir=TOOL_DOWN,
            q_ref=q_home, max_iters=600, pos_tol=1.5e-3, rot_tol=1e-2)
        env.data.ctrl[:6] = q_h
        settle(env, 1.5)

        q, pe, re, ok = env.ik.solve_with_restarts(
            q_home, rng, target_pos=np.array([*CUBE_XY, z]), target_z_dir=TOOL_DOWN,
            q_ref=q_home, max_iters=600, pos_tol=1.5e-3, rot_tol=1e-2)
        env.data.ctrl[:6] = q
        settle(env, 1.5)
        tcp = env.data.site_xpos[env.ik.site_id].copy()

        env.data.ctrl[6] = 255.0
        settle(env, 0.8)

        driver = env.data.qpos[env.grip_qadr]
        cube1 = env.obj_pos("cube")
        displaced = np.linalg.norm(cube1[:2] - cube0[:2])
        finger_contacts = 0
        for i in range(env.data.ncon):
            names = {mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, env.data.contact[i].geom1),
                     mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, env.data.contact[i].geom2)}
            if any("finger" in (n or "") or "cube" in (n or "") for n in names) and len(names) > 1:
                finger_contacts += 1
        held = finger_contacts >= 2 and cube1[2] > 0.63
        print(f"z={z:.3f} tcp_err={np.linalg.norm(tcp-np.array([*CUBE_XY, z]))*1e3:.0f}mm "
              f"driver={driver:.3f} cube_disp={displaced*1e3:.0f}mm cube_z={cube1[2]:.3f} "
              f"finger_contacts={finger_contacts} {'HELD' if held else ''}")


if __name__ == "__main__":
    main()
