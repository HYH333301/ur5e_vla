"""HDF5 episode recorder: T actions and T+1 observations (obs precedes action).

Images are stored JPEG-encoded in variable-length uint8 fields, one dataset
per camera. Layout:

  attrs: success, instruction, cube_rgba, target_rgba, n_actions, ...
  observations/qpos (T+1,6)  qvel (T+1,6)  gripper (T+1,1)
  observations/tcp_pos (T+1,3)  tcp_quat (T+1,4)  cube_pos (T+1,3)
  observations/image_{cam} (T+1,) vlen uint8 (JPEG bytes)
  action (T,7) f32   -- 6 joint targets + gripper cmd in [0,1]
  phase (T+1,) i8    -- expert phase id (see expert.PHASES)
"""
from __future__ import annotations

import io
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

_SCALARS = ("qpos", "qvel", "gripper", "tcp_pos", "tcp_quat", "cube_pos")


def encode_jpeg(rgb: np.ndarray, quality: int = 85) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return np.frombuffer(buf.getvalue(), dtype=np.uint8)


class EpisodeRecorder:
    def __init__(self, cam_names):
        self.cam_names = list(cam_names)
        self._obs: list[dict] = []
        self._frames: list[dict[str, np.ndarray]] = []
        self._phases: list[int] = []
        self._actions: list[np.ndarray] = []

    @property
    def n_actions(self) -> int:
        return len(self._actions)

    def add_obs(self, obs: dict, frames_jpeg: dict[str, np.ndarray], phase: int) -> None:
        self._obs.append(obs)
        self._frames.append(frames_jpeg)
        self._phases.append(int(phase))

    def add_action(self, action) -> None:
        self._actions.append(np.asarray(action, dtype=np.float32))

    def save(self, path: str | Path, attrs: dict) -> None:
        T = len(self._actions)
        if len(self._obs) != T + 1:
            raise ValueError(f"need T+1 obs for T actions, got {len(self._obs)} vs {T + 1}")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            for k, v in attrs.items():
                f.attrs[k] = v
            f.attrs["n_actions"] = T
            g = f.create_group("observations")
            for key in _SCALARS:
                g.create_dataset(key, data=np.stack([o[key] for o in self._obs]).astype(np.float32))
            vlen = h5py.vlen_dtype(np.uint8)
            for cam in self.cam_names:
                ds = g.create_dataset(f"image_{cam}", (T + 1,), dtype=vlen)
                for t, fr in enumerate(self._frames):
                    ds[t] = fr[cam]
            f.create_dataset("action", data=np.stack(self._actions))
            f.create_dataset("phase", data=np.asarray(self._phases, dtype=np.int8))
