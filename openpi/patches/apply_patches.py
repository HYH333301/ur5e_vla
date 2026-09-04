"""Apply (or export) the UR5e patches onto a local openpi checkout.

openpi-main is not a git repo here, so patches are kept as full file copies in
patches/files/ mirroring the openpi tree. Two directions:

  apply (default): copy patches/files/... -> <openpi_dir>/...
  export:          copy the live patched files from <openpi_dir> back here,
                   so edits made directly in openpi-main can be committed.

Usage:
  python apply_patches.py                       # apply to D:/code/openpi-main
  python apply_patches.py --openpi-dir D:/code/openpi-main --export
"""
import argparse, pathlib, shutil, sys

HERE = pathlib.Path(__file__).resolve().parent
FILES = HERE / "files"

# repo-relative -> destination path inside the openpi checkout
PATCHES = [
    "src/openpi/policies/ur5e_policy.py",   # new file: UR5e input/output transforms
    "src/openpi/training/config.py",        # added: LeRobotUr5eDataConfig + pi05_ur5e(_lora) TrainConfigs
    "scripts/serve_policy.py",              # added: "ur5e" EnvMode + default checkpoint
]


def apply(openpi_dir: pathlib.Path):
    for rel in PATCHES:
        src, dst = FILES / rel, openpi_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"applied {rel}")


def export(openpi_dir: pathlib.Path):
    for rel in PATCHES:
        src, dst = openpi_dir / rel, FILES / rel
        if not src.exists():
            sys.exit(f"missing source file: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"exported {rel}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--openpi-dir", default="D:/code/openpi-main")
    ap.add_argument("--export", action="store_true", help="copy live files back into patches/files/")
    args = ap.parse_args()
    target = pathlib.Path(args.openpi_dir)
    if not (target / "pyproject.toml").exists():
        sys.exit(f"{target} does not look like an openpi checkout")
    export(target) if args.export else apply(target)
