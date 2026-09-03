"""Render one frame per camera at the home keyframe for visual QC."""
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "ur5e_2f85.xml"
OUT = ROOT / "out" / "render_test"
OUT.mkdir(parents=True, exist_ok=True)

WIDTH, HEIGHT = 640, 480


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    for cam in ("front_cam", "side_cam", "wrist_cam"):
        renderer.update_scene(data, camera=cam)
        rgb = renderer.render()
        path = OUT / f"{cam}.png"
        Image.fromarray(rgb).save(path)
        print(f"saved {path}  mean_rgb={rgb.mean(axis=(0, 1)).round(1)}")
    renderer.close()


if __name__ == "__main__":
    main()
