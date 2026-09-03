"""Damped least-squares (DLS) IK for the UR5e arm (6 hinge joints) w.r.t. the gripper TCP.

Two orientation modes:
  - tool-down: align the TCP z-axis (gripper approach axis) with a desired world
    direction (e.g. [0, 0, -1] for top-down grasping). Yaw is left free.
  - full-pose: align the full TCP quaternion.

The solver evaluates forward kinematics on its own MjData so it never corrupts
the live simulation state.
"""
from __future__ import annotations

import mujoco
import numpy as np

ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


def align_error(z_from: np.ndarray, z_to: np.ndarray) -> np.ndarray:
    """3-vector angular error that rotates unit vector z_from onto z_to."""
    z_from = z_from / np.linalg.norm(z_from)
    z_to = z_to / np.linalg.norm(z_to)
    cross = np.cross(z_from, z_to)
    s = np.linalg.norm(cross)
    d = float(np.dot(z_from, z_to))
    if s < 1e-9:
        if d > 0:
            return np.zeros(3)
        # 180 deg: pick any perpendicular axis
        perp = np.cross(z_from, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(z_from, [0.0, 1.0, 0.0])
        return perp / np.linalg.norm(perp) * np.pi
    return cross / s * np.arctan2(s, d)


class ArmIK:
    def __init__(
        self,
        model: mujoco.MjModel,
        site_name: str = "pinch",
        joint_names: tuple[str, ...] = ARM_JOINTS,
    ):
        self.model = model
        self.ik_data = mujoco.MjData(model)  # scratch state, never steps
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"site '{site_name}' not found")
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
        if any(j < 0 for j in jids):
            raise ValueError(f"unknown joint in {joint_names}")
        self.jids = np.asarray(jids)
        self.qadr = np.asarray([model.jnt_qposadr[j] for j in jids])
        self.vadr = np.asarray([model.jnt_dofadr[j] for j in jids])
        self.qlo = model.jnt_range[jids, 0].copy()
        self.qhi = model.jnt_range[jids, 1].copy()
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))

    # ------------------------------------------------------------------ FK
    def _fk(self, q: np.ndarray):
        d = self.ik_data
        mujoco.mj_resetData(self.model, d)
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        pos = d.site_xpos[self.site_id].copy()
        R = d.site_xmat[self.site_id].reshape(3, 3).copy()
        return pos, R

    # ------------------------------------------------------------------ IK
    def solve(
        self,
        q_init: np.ndarray,
        target_pos: np.ndarray,
        target_z_dir: np.ndarray | None = None,
        target_quat: np.ndarray | None = None,
        damping: float = 0.05,
        rot_weight: float = 0.3,
        max_iters: int = 300,
        max_step: float = 0.15,
        pos_tol: float = 5e-4,
        rot_tol: float = 5e-3,
        q_ref: np.ndarray | None = None,
        null_gain: float = 0.05,
    ) -> tuple[np.ndarray, float, float, bool]:
        """Returns (q, pos_err, rot_err, converged).

        Orientation is tool-down mode when target_z_dir is given, full pose when
        target_quat is given, unconstrained when neither is given.
        """
        q = np.clip(np.asarray(q_init, dtype=float).copy(), self.qlo, self.qhi)
        I6 = np.eye(6)
        stall = 0

        def residuals(qv):
            """Returns (e_pos, e_rot); the cost for line search is the squared norm
            of the weighted stack [e_pos; rot_weight*e_rot] -- this is the SAME
            objective the DLS step minimizes, so the step is a guaranteed descent
            direction for it."""
            pos, R = self._fk(qv)
            e_pos = np.asarray(target_pos) - pos
            if target_z_dir is not None:
                e_rot = align_error(R[:, 2], np.asarray(target_z_dir, dtype=float))
            elif target_quat is not None:
                site_quat = np.empty(4)
                mujoco.mju_mat2Quat(site_quat, self.ik_data.site_xmat[self.site_id])
                e_rot = np.empty(3)
                mujoco.mju_subQuat(e_rot, np.asarray(target_quat, dtype=float), site_quat)
            else:
                e_rot = np.zeros(3)
            return e_pos, e_rot

        def cost_of(e_pos, e_rot):
            return float(np.linalg.norm(e_pos) ** 2 + (rot_weight * np.linalg.norm(e_rot)) ** 2)

        best_q, best_cost = q.copy(), np.inf

        for _ in range(max_iters):
            e_pos, e_rot = residuals(q)
            pos_err = float(np.linalg.norm(e_pos))
            rot_err = float(np.linalg.norm(e_rot))
            cost = cost_of(e_pos, e_rot)
            if cost < best_cost:
                best_cost, best_q = cost, q.copy()

            if pos_err < pos_tol and rot_err < rot_tol:
                return q, pos_err, rot_err, True
            if stall >= 5:
                break

            self.ik_data.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.model, self.ik_data)
            mujoco.mj_comPos(self.model, self.ik_data)
            mujoco.mj_jacSite(self.model, self.ik_data, self.jacp, self.jacr, self.site_id)
            J = np.vstack((self.jacp[:, self.vadr], rot_weight * self.jacr[:, self.vadr]))
            e = np.concatenate((e_pos, rot_weight * e_rot))

            # damping scales with residual: bold far away, careful near singularities
            lam = damping * max(1.0, 4.0 * np.sqrt(cost))
            A = J @ J.T + lam**2 * I6
            dq = J.T @ np.linalg.solve(A, e)
            if q_ref is not None:
                N = I6 - J.T @ np.linalg.solve(A, J)  # nullspace projector
                dq += N @ (null_gain * (np.asarray(q_ref) - q))
            n = np.linalg.norm(dq)
            if n > max_step:
                dq *= max_step / n

            # backtracking: halve the step until the stacked-residual cost decreases
            improved = False
            for _ in range(8):
                q_new = np.clip(q + dq, self.qlo, self.qhi)
                e_p2, e_r2 = residuals(q_new)
                if cost_of(e_p2, e_r2) < cost:
                    q, improved, stall = q_new, True, 0
                    break
                dq *= 0.5
            if not improved:
                stall += 1

        e_pos, e_rot = residuals(best_q)
        return best_q, float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_rot)), False

    def solve_with_restarts(
        self,
        q_init: np.ndarray,
        rng: np.random.Generator,
        n_restarts: int = 5,
        jitter: float = 0.25,
        **kwargs,
    ):
        """Try q_init first, then perturbed seeds. Returns the first converged solution."""
        q, pe, re, ok = self.solve(q_init, **kwargs)
        if ok:
            return q, pe, re, ok
        for _ in range(n_restarts):
            seed = np.asarray(q_init) + rng.uniform(-jitter, jitter, size=6)
            q2, pe2, re2, ok2 = self.solve(seed, **kwargs)
            if ok2 and pe2 + 0.3 * re2 < pe + 0.3 * re:
                return q2, pe2, re2, ok2
        return q, pe, re, ok
