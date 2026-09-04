"""Download the LeRobot-format dataset from the HF hub (private: hyh1234/ur5e_vla_lerobot).

Keeps ~2.9 GB out of git. Token is read from <repo>/.env (HF_TOKEN=...).

Usage:
  .venv-lerobot/Scripts/python.exe download_dataset.py [--local-dir data/lerobot/ur5e_vla_lerobot]
"""
import argparse, io, pathlib

from huggingface_hub import snapshot_download


def hf_token() -> str:
    env = pathlib.Path(__file__).resolve().parents[1] / ".env"
    for line in io.open(env, encoding="utf-8"):
        if line.startswith("HF_TOKEN="):
            return line.strip().split("=", 1)[1]
    raise SystemExit(f"HF_TOKEN not found in {env}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-dir", default="data/lerobot/ur5e_vla_lerobot")
    args = ap.parse_args()
    path = snapshot_download(
        repo_id="hyh1234/ur5e_vla_lerobot",
        repo_type="dataset",
        local_dir=args.local_dir,
        token=hf_token(),
    )
    print("dataset at:", path)
