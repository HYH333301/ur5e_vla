"""Sanity checks for the ur5e_2f85 combined model."""
from pathlib import Path

import mujoco
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "ur5e_2f85.xml"


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} timestep={model.opt.timestep}")

    print("\n-- joints (id, name, qposadr, range) --")
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        adr = model.jnt_qposadr[j]
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            rng = "free"
        else:
            rng = f"[{model.jnt_range[j][0]:.3f}, {model.jnt_range[j][1]:.3f}]"
        print(f"  {j:2d} {name:26s} qposadr={adr:2d} {rng}")

    print("\n-- actuators --")
    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        print(f"  {a}: {name} ctrlrange={model.actuator_ctrlrange[a]}")

    # home keyframe -> forward kinematics
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)

    pinch = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
    attach = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    for sname in ("pinch", "attachment_site"):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, sname)
        xmat = data.site_xmat[sid].reshape(3, 3)
        # columns of xmat are the site axes expressed in world
        print(f"\n-- {sname} @ home --")
        print(f"  pos (world): {np.round(data.site_xpos[sid], 3)}")
        print(f"  x-axis: {np.round(xmat[:, 0], 3)}  y-axis: {np.round(xmat[:, 1], 3)}  "
              f"z-axis (approach): {np.round(xmat[:, 2], 3)}")

    # cube pose at home keyframe
    cube_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    print(f"\ncube body xpos @ home: {np.round(data.xpos[cube_bid], 3)}")

    # step physics with gripper closing to make sure the mechanism holds together
    data.ctrl[6] = 255.0
    for _ in range(500):
        mujoco.mj_step(model, data)
    drv = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_driver_joint")
    q = data.qpos[model.jnt_qposadr[drv]]
    print(f"\nafter 1s of ctrl=255 (close): right_driver qpos={q:.3f} (0=open, 0.8=fully closed)")
    data.ctrl[6] = 0.0
    for _ in range(1000):
        mujoco.mj_step(model, data)
    q = data.qpos[model.jnt_qposadr[drv]]
    print(f"after 2s of ctrl=0 (open):   right_driver qpos={q:.3f}")
    print("\nOK: model loads and simulates.")


if __name__ == "__main__":
    main()
