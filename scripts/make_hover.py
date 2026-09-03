"""Compute a hover start pose via IK and verify it is collision-free.

Hover = TCP (pinch site) above the table center, tool z-axis pointing straight
down. Prints the arm configuration as a keyframe row to paste into the model.
"""
from pathlib import Path
import sys

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ik import ArmIK  # noqa: E402

MODEL_PATH = ROOT / "model" / "ur5e_2f85.xml"
HOVER_TCP = np.array([-0.52, 0.0, 0.85])
TOOL_DOWN = np.array([0.0, 0.0, -1.0])


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key)

    ik = ArmIK(model)
    q_home = data.qpos[ik.qadr].copy()
    rng = np.random.default_rng(0)

    q, pe, re, ok = ik.solve_with_restarts(
        q_home, rng, target_pos=HOVER_TCP, target_z_dir=TOOL_DOWN,
        q_ref=q_home, max_iters=400,
    )
    print(f"converged={ok}  pos_err={pe:.5f} m  rot_err={re:.5f} rad")
    print("hover q (rad):", np.round(q, 4).tolist())

    # conditioning check: smallest non-redundant singular value of the stacked J
    d = ik.ik_data
    d.qpos[ik.qadr] = q
    mujoco.mj_kinematics(model, d)
    mujoco.mj_comPos(model, d)
    mujoco.mj_jacSite(model, d, ik.jacp, ik.jacr, ik.site_id)
    J = np.vstack((ik.jacp[:, ik.vadr], 0.3 * ik.jacr[:, ik.vadr]))
    sv = np.linalg.svd(J, compute_uv=False)
    print(f"stacked-J singular values: {np.round(sv, 3)}  (sigma5={sv[4]:.3f}, want > 0.1)")

    # verify in the live model: set arm, keep cube from keyframe
    mujoco.mj_resetDataKeyframe(model, data, key)
    data.qpos[ik.qadr] = q
    data.ctrl[:6] = q
    data.ctrl[6] = 0.0
    mujoco.mj_forward(model, data)

    sid = ik.site_id
    print(f"TCP pos: {np.round(data.site_xpos[sid], 4)}  "
          f"z-axis: {np.round(data.site_xmat[sid].reshape(3, 3)[:, 2], 4)}")

    # settle briefly and check for unwanted contacts
    for _ in range(250):
        mujoco.mj_step(model, data)
    print(f"after 0.5s: ncon={data.ncon}")
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, c.geom1 // 1000)
        b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, c.geom2 // 1000)
        g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        print(f"  contact: {b1}/{g1}  <->  {b2}/{g2}")

    # render check
    out = ROOT / "out" / "render_test"
    out.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=480, width=640)
    for cam in ("front_cam", "side_cam", "wrist_cam"):
        renderer.update_scene(data, camera=cam)
        Image.fromarray(renderer.render()).save(out / f"hover_{cam}.png")
    renderer.close()
    print(f"renders saved to {out}/hover_*.png")

    # ready-to-paste keyframe arm qpos (3 decimals) + gripper zeros
    print("\nkeyframe qpos arm+gripper:")
    print("  " + " ".join(f"{v:.4f}" for v in q) + "  " + "0 0 0 0 0 0 0 0")


if __name__ == "__main__":
    main()
