"""Scripted pick-place expert: IK waypoints + joint-space interpolation.

Plans the full action sequence for one episode upfront. Each action is
(6 arm joint position targets, 1 gripper command in [0,1]) at the 20 Hz
control rate.
"""
from __future__ import annotations

import numpy as np

from env import OBJ
from ik import ArmIK

TOOL_DOWN = np.array([0.0, 0.0, -1.0])

HOVER_Z = 0.78   # transit height
GRASP_OFF = 0.005  # TCP above object center at close: cube-calibrated via
                   # scripts/probe_grasp.py (plateau 0.615-0.665); accounts for servo sag
PLACE_OFF = 0.012  # TCP above tray-rest object center at release: small drop, less bounce


def grasp_z(obj: str) -> float:
    return OBJ[obj]["rest"] + GRASP_OFF


def place_z(obj: str) -> float:
    return OBJ[obj]["rest_cont"] + PLACE_OFF

JOINT_SPEED = 0.4   # rad/s: servo kv=400 is heavily overdamped; faster ramps leave
                     # a tracking lag that puts the fingers above the cube at close
GRIP_CLOSE = 1.0
GRIP_OPEN = 0.0
WAIT_TICKS = 12     # 0.6 s dwell for gripper close/open
SETTLE_TICKS = 8    # dwell at motion waypoints, lets the servo finish tracking
DESCEND_TICKS = 12  # extra settle at grasp height: servo converges the last cm slowly

PHASES = ("approach", "descend", "close", "lift", "carry", "place", "open", "retreat")
PH = {name: i for i, name in enumerate(PHASES)}


class ExpertFailure(Exception):
    """IK could not reach a waypoint."""


class PickPlaceExpert:
    def __init__(self, ik: ArmIK, rng: np.random.Generator):
        self.ik = ik
        self.rng = rng

    def _ik(self, q_seed: np.ndarray, pos: np.ndarray) -> np.ndarray:
        q, pe, re, ok = self.ik.solve_with_restarts(
            q_seed, self.rng, target_pos=pos, target_z_dir=TOOL_DOWN,
            q_ref=q_seed, max_iters=600, pos_tol=1.5e-3, rot_tol=1e-2,
        )
        # judge by absolute error: near workspace edges the solver stalls at
        # sub-mm error without flagging convergence
        if pe > 2.5e-3 or re > 2.5e-2:
            raise ExpertFailure(f"IK failed for {np.round(pos, 3)}: "
                                f"pe={pe:.4f} re={re:.4f} ok={ok}")
        return q

    def plan(self, obj_xy, tgt_xy, q_start, obj: str = "cube") -> tuple[np.ndarray, np.ndarray]:
        """Plan pick(obj_xy) -> place(tgt_xy); obj names a shape in env.OBJ.
        Returns (actions (T,7) float32, phases (T,) int8)."""
        dt = 1.0 / 20.0
        steps: list[tuple[np.ndarray, float, int]] = []
        q = np.asarray(q_start, dtype=float).copy()

        def move_to(pos, grip, phase, dwell=SETTLE_TICKS):
            nonlocal q
            q_next = self._ik(q, np.asarray(pos, dtype=float))
            n = max(1, int(np.ceil(np.abs(q_next - q).max() / (JOINT_SPEED * dt))))
            for a in np.linspace(0.0, 1.0, n):
                steps.append((q + a * (q_next - q), grip, phase))
            for _ in range(dwell):
                steps.append((q_next.copy(), grip, phase))
            q = q_next

        def hold(grip, phase, ticks=WAIT_TICKS):
            for _ in range(ticks):
                steps.append((q.copy(), grip, phase))

        move_to((*obj_xy, HOVER_Z), GRIP_OPEN, PH["approach"])
        move_to((*obj_xy, grasp_z(obj)), GRIP_OPEN, PH["descend"], dwell=DESCEND_TICKS)
        hold(GRIP_CLOSE, PH["close"])
        move_to((*obj_xy, HOVER_Z), GRIP_CLOSE, PH["lift"])
        move_to((*tgt_xy, HOVER_Z), GRIP_CLOSE, PH["carry"])
        move_to((*tgt_xy, place_z(obj)), GRIP_CLOSE, PH["place"])
        hold(GRIP_OPEN, PH["open"])
        move_to((*tgt_xy, HOVER_Z), GRIP_OPEN, PH["retreat"])

        actions = np.zeros((len(steps), 7), dtype=np.float32)
        phases = np.zeros(len(steps), dtype=np.int8)
        for t, (qt, grip, ph) in enumerate(steps):
            actions[t, :6] = qt
            actions[t, 6] = grip
            phases[t] = ph
        return actions, phases
