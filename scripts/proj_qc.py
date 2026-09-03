"""Exact framing QC: project key scene points into each camera's pixel coordinates.

Avoids vision-model ambiguity: computes where the TCP, cube, target, base and
table corners land in each camera's image and reports out-of-frame points.
"""
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "ur5e_2f85.xml"

WIDTH, HEIGHT = 640, 480

# world-frame points of interest
POINTS = {
    "TCP(pinch)": np.array([-0.52, 0.0, 0.85]),
    "wrist3": None,  # filled from site xpos
    "cube_center": np.array([-0.55, 0.05, 0.660]),
    "target_center": np.array([-0.55, -0.05, 0.605]),
    "base(0,0,0.60)": np.array([0.0, 0.0, 0.60]),
    "tablecorner+far": np.array([0.30, 0.45, 0.60]),
    "tablecorner-near": np.array([-0.90, -0.45, 0.60]),
    "tablecorner-left": np.array([-0.90, 0.45, 0.60]),
    "tablecorner-right": np.array([0.30, -0.45, 0.60]),
}


def project(cam_pos, cam_mat, fovy, p, width, height):
    """Return (u, v, in_front) pixel coords of world point p."""
    local = cam_mat.T @ (p - cam_pos)  # camera frame
    f = (height / 2.0) / np.tan(np.radians(fovy) / 2.0)
    if local[2] >= -1e-6:
        return None, None, False
    u = width / 2.0 + f * local[0] / (-local[2])
    v = height / 2.0 - f * local[1] / (-local[2])
    return u, v, True


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)

    # ground-truth tool axis from kinematics
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
    z_axis = data.site_xmat[sid].reshape(3, 3)[:, 2]
    print(f"TCP world pos: {np.round(data.site_xpos[sid], 4)}  tool z-axis: {np.round(z_axis, 4)}")

    POINTS["wrist3"] = data.xpos[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link")
    ].copy()

    for cam_name in ("front_cam", "side_cam", "wrist_cam"):
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        pos = data.cam_xpos[cid].copy()
        mat = data.cam_xmat[cid].reshape(3, 3).copy()
        fovy = model.cam_fovy[cid]
        print(f"\n== {cam_name}  pos={np.round(pos, 3)} fovy={fovy} ==")
        n_out = 0
        for name, p in POINTS.items():
            u, v, front = project(pos, mat, fovy, p, WIDTH, HEIGHT)
            if not front:
                print(f"  {name:<18} BEHIND CAMERA")
                n_out += 1
                continue
            inset = 8 <= u <= WIDTH - 8 and 8 <= v <= HEIGHT - 8
            if not inset:
                n_out += 1
            print(f"  {name:<18} ({u:6.1f}, {v:6.1f})  {'ok' if inset else 'OUT OF FRAME'}")
        print(f"  -> {n_out} point(s) out of frame" if n_out else "  -> all points in frame")


if __name__ == "__main__":
    main()
