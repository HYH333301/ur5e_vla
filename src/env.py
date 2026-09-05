"""Sim environment for the UR5e + 2F-85 pick-place task.

20 Hz control loop (position targets for the arm, 0-255 tendon command for the
gripper), per-episode randomization of object/container placement and colors,
and success checking (task object resting inside the task container).

Scene: 3 graspable objects (cube / sphere / cylinder) and 2 place trays. Each
episode samples a task: pick one object, place it in one tray.
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
CONT_BASE_TOP = 0.606        # tray floor (top of the base plate)
CONT_INNER_HALF = 0.066      # inner half width of a tray (wall inner face)

# object geometry: half extent along z + resting center heights (table / tray)
OBJ = {
    "cube":     dict(half=0.025, rest=TABLE_TOP + 0.025, rest_cont=CONT_BASE_TOP + 0.025),
    "sphere":   dict(half=0.022, rest=TABLE_TOP + 0.022, rest_cont=CONT_BASE_TOP + 0.022),
    "cylinder": dict(half=0.030, rest=TABLE_TOP + 0.030, rest_cont=CONT_BASE_TOP + 0.030),
}
OBJ_NAMES = ("cube", "sphere", "cylinder")
N_CONT = 2

XY_LO = np.array([-0.78, -0.28])          # object sampling box
XY_HI = np.array([-0.28, 0.28])
CONT_LO = np.array([-0.72, -0.22])        # tray sampling box (trays are wide)
CONT_HI = np.array([-0.34, 0.22])
MIN_SEP = 0.20   # object-object (gripper needs clearance to approach)
MIN_SEP_CONT = 0.18  # anything involving a tray footprint (0.148 wide)

CONTROL_HZ = 20
STEPS_PER_TICK = 10  # model timestep 0.002 -> 0.05 s per control tick

# hue buckets -> plain color names for the language instruction
_HUE_NAMES = (
    (0.045, "red"), (0.11, "orange"), (0.19, "yellow"), (0.46, "green"),
    (0.68, "cyan"), (0.82, "blue"), (0.95, "purple"), (1.0, "pink"),
)
_WOOD_HUE = 0.08  # table color; keep object/tray hues distinguishable from it

# tray geom name suffixes (base plate + 4 walls), colors are set per episode
_CONT_GEOMS = ("base", "wall_n", "wall_s", "wall_e", "wall_w")


class EpisodeDone(Exception):
    """Raised to abort an episode early (failed grasp, dropped object, ...)."""


def hue_name(h: float) -> str:
    for lim, name in _HUE_NAMES:
        if h < lim:
            return name
    return "pink"


def hsv_rgb(h: float, s: float = 0.85, v: float = 0.85) -> np.ndarray:
    return np.asarray(colorsys.hsv_to_rgb(h % 1.0, s, v))


def _sample_hues(rng: np.random.Generator, n: int) -> list[float]:
    """n hues with pairwise-distinct names, none reading as the wood tone."""
    hues: list[float] = []
    while len(hues) < n:
        h = rng.uniform(0, 1)
        name = hue_name(h)
        if name == "orange" or name in {hue_name(x) for x in hues}:
            continue
        hues.append(h)
    return hues


class Ur5eEnv:
    def __init__(self, model_path: str | Path = MODEL_PATH, width: int = 320, height: int = 240):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.ik = ArmIK(self.model)
        self.key_home = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")

        self.obj_qadr, self.obj_vadr, self.obj_geom = {}, {}, {}
        for name in OBJ_NAMES:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            jid = self.model.body_jntadr[bid]
            self.obj_qadr[name] = self.model.jnt_qposadr[jid]
            self.obj_vadr[name] = self.model.jnt_dofadr[jid]
            self.obj_geom[name] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_geom")
        self.cont_body, self.cont_geoms = [], []
        for i, tag in enumerate(("a", "b")):
            self.cont_body.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"container_{tag}"))
            self.cont_geoms.append([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"cont_{tag}_{sfx}")
                                    for sfx in _CONT_GEOMS])
        self.grip_qadr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "right_driver_joint")
        ]

        self.width, self.height = width, height
        self._renderer = None  # created lazily; keeps GL context out of headless runs

        self.obj_xy = {name: np.zeros(2) for name in OBJ_NAMES}
        self.cont_xy = [np.zeros(2) for _ in range(N_CONT)]
        self.task_obj = OBJ_NAMES[0]
        self.task_cont = 0
        self.instruction = ""
        self.obj_rgba = {name: np.zeros(4) for name in OBJ_NAMES}
        self.cont_rgba = [np.zeros(4) for _ in range(N_CONT)]

    # ------------------------------------------------------------------ reset
    def reset(self, rng: np.random.Generator) -> dict:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_home)

        for _ in range(1000):  # 5 items, pairwise separation
            obj_xy = [rng.uniform(XY_LO, XY_HI) for _ in OBJ_NAMES]
            cont_xy = [rng.uniform(CONT_LO, CONT_HI) for _ in range(N_CONT)]
            ok = all(np.linalg.norm(a - b) >= MIN_SEP
                     for i, a in enumerate(obj_xy) for b in obj_xy[i + 1:])
            ok = ok and all(np.linalg.norm(a - b) >= MIN_SEP_CONT
                            for a in obj_xy for b in cont_xy)
            ok = ok and np.linalg.norm(cont_xy[0] - cont_xy[1]) >= MIN_SEP_CONT
            if ok:
                break
        else:
            raise RuntimeError("failed to sample scene placement")

        h_objs = _sample_hues(rng, len(OBJ_NAMES))
        h_conts = _sample_hues(rng, N_CONT)
        for i, name in enumerate(OBJ_NAMES):
            self.obj_rgba[name] = np.append(hsv_rgb(h_objs[i], s=0.9, v=0.85), 1.0)
            self.model.geom_rgba[self.obj_geom[name]] = self.obj_rgba[name]
        for i in range(N_CONT):
            self.cont_rgba[i] = np.append(hsv_rgb(h_conts[i], s=0.9, v=0.75), 1.0)
            for gid in self.cont_geoms[i]:
                self.model.geom_rgba[gid] = self.cont_rgba[i]
            self.model.body_pos[self.cont_body[i], :2] = cont_xy[i]

        # the episode's task: pick one object, place it in one tray
        self.task_obj = OBJ_NAMES[int(rng.integers(len(OBJ_NAMES)))]
        self.task_cont = int(rng.integers(N_CONT))
        o_name = hue_name(h_objs[OBJ_NAMES.index(self.task_obj)])
        c_name = hue_name(h_conts[self.task_cont])
        self.instruction = (f"pick up the {o_name} {self.task_obj} "
                            f"and place it in the {c_name} container")
        for i, name in enumerate(OBJ_NAMES):
            self.obj_xy[name] = obj_xy[i]
            self.data.qpos[self.obj_qadr[name]:self.obj_qadr[name] + 3] = (*obj_xy[i], OBJ[name]["rest"] + 0.001)
            self.data.qpos[self.obj_qadr[name] + 3:self.obj_qadr[name] + 7] = (1, 0, 0, 0)
            self.data.qvel[self.obj_vadr[name]:self.obj_vadr[name] + 6] = 0.0
        self.cont_xy = [np.asarray(c, dtype=float) for c in cont_xy]
        mujoco.mj_forward(self.model, self.data)

        for _ in range(100):  # 0.2 s: objects settle, arm holds the hover pose
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
            "objects_pos": np.stack([self.obj_pos(n) for n in OBJ_NAMES]),
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
    def obj_pos(self, name: str) -> np.ndarray:
        return self.data.qpos[self.obj_qadr[name]:self.obj_qadr[name] + 3].copy()

    def obj_vel(self, name: str) -> np.ndarray:
        """6-dof object velocity (linear then angular)."""
        return self.data.qvel[self.obj_vadr[name]:self.obj_vadr[name] + 6].copy()

    def obj_still(self, name: str) -> bool:
        """Linear speed below threshold (angular spin, e.g. a rolling sphere,
        does not count against success)."""
        return float(np.linalg.norm(self.obj_vel(name)[:3])) < 0.05

    @property
    def task_obj_xy(self) -> np.ndarray:
        return self.obj_pos(self.task_obj)[:2].copy()

    def cont_pos(self, i: int) -> np.ndarray:
        """Task-container tray center (xy) from the (static) body pose."""
        return self.data.xpos[self.cont_body[i], :2].copy()

    @property
    def task_cont_xy(self) -> np.ndarray:
        return self.cont_pos(self.task_cont)

    def check_success(self) -> tuple[bool, float]:
        obj = self.obj_pos(self.task_obj)
        tgt = self.task_cont_xy
        dist = float(np.linalg.norm(obj[:2] - tgt[:2]))
        in_tray = dist < 0.045
        at_rest = abs(obj[2] - OBJ[self.task_obj]["rest_cont"]) < 0.012
        still = self.obj_still(self.task_obj)
        return bool(in_tray and at_rest and still), dist

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
