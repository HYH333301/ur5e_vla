"""Sim environment for the UR5e + 2F-85 pick-place task.

20 Hz control loop (position targets for the arm, 0-255 tendon command for the
gripper), per-episode randomization of cube/target placement and colors, and
success checking (cube resting within the target disc).
"""
from __future__ import annotations

import colorsys
from pathlib import Path

import mujoco
import numpy as np

from ik import ArmIK

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "ur5e_2f85.xml"

CAMS = ("front_cam", "side_cam", "wrist_cam")

TABLE_TOP = 0.60
CUBE_HALF = 0.025
CUBE_Z = TABLE_TOP + CUBE_HALF  # 0.625: cube center when resting on the table
XY_LO = np.array([-0.78, -0.28])
XY_HI = np.array([-0.28, 0.28])
MIN_SEP = 0.20  # cube/target minimum separation (m)

CONTROL_HZ = 20
STEPS_PER_TICK = 10  # model timestep 0.002 -> 0.05 s per control tick

# hue buckets -> plain color names for the language instruction
_HUE_NAMES = (
    (0.045, "red"), (0.11, "orange"), (0.19, "yellow"), (0.46, "green"),
    (0.68, "cyan"), (0.82, "blue"), (0.95, "purple"), (1.0, "pink"),
)
_WOOD_HUE = 0.08  # table color; keep cube/target hues distinguishable from it


class EpisodeDone(Exception):
    """Raised to abort an episode early (failed grasp, dropped cube, ...)."""


def hue_name(h: float) -> str:
    for lim, name in _HUE_NAMES:
        if h < lim:
            return name
    return "pink"


def hsv_rgb(h: float, s: float = 0.85, v: float = 0.85) -> np.ndarray:
    return np.asarray(colorsys.hsv_to_rgb(h % 1.0, s, v))


class Ur5eEnv:
    def __init__(self, model_path: str | Path = MODEL_PATH, width: int = 320, height: int = 240):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.ik = ArmIK(self.model)
        self.key_home = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")

        cube_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cube_jid = self.model.body_jntadr[cube_bid]
        self.cube_qadr = self.model.jnt_qposadr[cube_jid]
        self.cube_vadr = self.model.jnt_dofadr[cube_jid]
        self.cube_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self.target_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site")
        self.grip_qadr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_driver_joint")
        ]

        self.width, self.height = width, height
        self._renderer = None  # created lazily; keeps GL context out of headless runs

        self.cube_xy = np.zeros(2)
        self.tgt_xy = np.zeros(2)
        self.instruction = ""
        self.cube_rgba = np.zeros(4)
        self.target_rgba = np.zeros(4)

    # ------------------------------------------------------------------ reset
    def reset(self, rng: np.random.Generator) -> dict:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_home)

        for _ in range(200):
            cube_xy = rng.uniform(XY_LO, XY_HI)
            tgt_xy = rng.uniform(XY_LO, XY_HI)
            if np.linalg.norm(cube_xy - tgt_xy) >= MIN_SEP:
                break
        else:
            raise RuntimeError("failed to sample cube/target placement")

        # two hues: distinct names, both away from the wood tone
        while True:
            h_cube, h_tgt = rng.uniform(0, 1), rng.uniform(0, 1)
            if hue_name(h_cube) != hue_name(h_tgt):
                break
        self.cube_rgba = np.append(hsv_rgb(h_cube, s=0.9, v=0.85), 1.0)
        self.target_rgba = np.append(hsv_rgb(h_tgt, s=0.9, v=0.75), 0.55)
        self.model.geom_rgba[self.cube_geom] = self.cube_rgba
        self.model.site_rgba[self.target_site] = self.target_rgba
        self.model.site_pos[self.target_site] = (*tgt_xy, TABLE_TOP + 0.005)
        c_name, t_name = hue_name(h_cube), hue_name(h_tgt)
        self.instruction = f"pick up the {c_name} cube and place it on the {t_name} target"
        self.cube_xy, self.tgt_xy = cube_xy, tgt_xy

        self.data.qpos[self.cube_qadr:self.cube_qadr + 3] = (*cube_xy, CUBE_Z + 0.001)
        self.data.qpos[self.cube_qadr + 3:self.cube_qadr + 7] = (1, 0, 0, 0)
        self.data.qvel[self.cube_vadr:self.cube_vadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

        for _ in range(100):  # 0.2 s: cube settles, arm holds the hover pose
            mujoco.mj_step(self.model, self.data)
        return self.get_obs()

    # ------------------------------------------------------------------ step
    def step(self, action: np.ndarray) -> dict:
        """Apply (6 joint targets, grip cmd in [0,1]) and advance one 0.05 s tick."""
        self.data.ctrl[:6] = np.clip(action[:6], self.model.actuator_ctrlrange[:6, 0],
                                     self.model.actuator_ctrlrange[:6, 1])
        self.data.ctrl[6] = np.clip(action[6], 0.0, 1.0) * 255.0
        for _ in range(STEPS_PER_TICK):
            mujoco.mj_step(self.model, self.data)
        return self.get_obs()

    # ------------------------------------------------------------------ obs
    def get_obs(self) -> dict:
        d = self.data
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, d.site_xmat[self.ik.site_id])
        return {
            "qpos": d.qpos[self.ik.qadr].copy(),
            "qvel": d.qvel[self.ik.vadr].copy(),
            "gripper": float(d.qpos[self.grip_qadr]),
            "tcp_pos": d.site_xpos[self.ik.site_id].copy(),
            "tcp_quat": quat,
            "cube_pos": d.qpos[self.cube_qadr:self.cube_qadr + 3].copy(),
        }

    def render(self) -> dict[str, np.ndarray]:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
        frames = {}
        for cam in CAMS:
            self._renderer.update_scene(self.data, camera=cam)
            frames[cam] = self._renderer.render()
        return frames

    # ------------------------------------------------------------------ misc
    def cube_pos(self) -> np.ndarray:
        return self.data.qpos[self.cube_qadr:self.cube_qadr + 3].copy()

    def check_success(self) -> tuple[bool, float]:
        cube = self.cube_pos()
        tgt = self.data.site_xpos[self.target_site]
        dist = float(np.linalg.norm(cube[:2] - tgt[:2]))
        on_table = abs(cube[2] - CUBE_Z) < 0.012
        still = float(np.abs(self.data.qvel[self.cube_vadr:self.cube_vadr + 6]).max()) < 0.05
        return bool(dist < 0.06 and on_table and still), dist

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
