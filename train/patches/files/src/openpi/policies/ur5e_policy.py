import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_ur5e_example() -> dict:
    """Creates a random input example for the UR5e policy."""
    return {
        "observation/state": np.random.rand(7),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/side_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the red cube and place it on the blue target",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class Ur5eInputs(transforms.DataTransformFn):
    """Converts inputs to the model format. Applied for both training and inference.

    The UR5e MuJoCo environment (see examples/ur5e) sends three real cameras:
    a front view, a side third-person view and a wrist view. Pi0 models expect
    one third-person view and two wrist views, so the side camera is mapped to
    the left wrist slot. All three are real images, so all image masks are True.
    """

    # Determines which model will be used. Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # LeRobot stores images as float32 (C,H,W); inference sends uint8 (H,W,C).
        base_image = _parse_image(data["observation/image"])
        side_image = _parse_image(data["observation/side_image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": side_image,
                "right_wrist_0_rgb": wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class Ur5eOutputs(transforms.DataTransformFn):
    """Converts model outputs back to the environment format (inference only)."""

    def __call__(self, data: dict) -> dict:
        # Actions are [6 joint position targets, 1 gripper command]; the rest is padding.
        return {"actions": np.asarray(data["actions"][..., :7])}


@dataclasses.dataclass(frozen=True)
class Ur5eEefInputs(transforms.DataTransformFn):
    """EEF-space variant of Ur5eInputs for zero-shot probing of pi05_base.

    state = [tcp_pos(3), tcp_quat(4, wxyz), gripper(1)] (8,).
    Two cameras (PI's typical UR layout): front -> base_0_rgb, wrist -> left_wrist_0_rgb;
    the right wrist slot is zero-filled with mask False.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        state = np.asarray(data["observation/state"], dtype=np.float32)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class Ur5eEefOutputs(transforms.DataTransformFn):
    """Model outputs are [target tcp_pos(3) after un-deltaing, gripper cmd(1)]."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :4])}
