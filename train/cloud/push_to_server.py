"""Sync the local train/ directory to the cloud container (code/ur5e_vla/).

Local edit -> apply_patches.py (sync openpi-main) -> commit -> push_to_server.py.
Bundles train/ (minus local-only tools sshlib/tunnel/putfile/getfile/env.example and
caches) into a tar.gz, uploads via SFTP, extracts on the server with --no-same-owner
(Windows uid breaks plain extraction). Prints what changed.

Usage (from repo root, Git Bash):
  .venv-lerobot/Scripts/python.exe train/cloud/push_to_server.py
"""
from __future__ import annotations

import io
import pathlib
import tarfile

import paramiko

from sshlib import connect

REMOTE_DIR = "/root/shared-nvme/code/ur5e_vla"
EXCLUDE = {"__pycache__", "sshlib.py", "tunnel.py", "putfile.py", "getfile.py", "env.sh.example"}


def make_tarball() -> io.BytesIO:
    src = pathlib.Path(__file__).resolve().parents[1]  # train/
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in sorted(src.rglob("*")):
            rel = p.relative_to(src)
            if any(part in EXCLUDE for part in rel.parts):
                continue
            if p.is_file():
                tar.add(p, arcname=str(rel))
    buf.seek(0)
    return buf


def main() -> None:
    ssh = connect()
    sftp = ssh.open_sftp()
    sftp.putfo(make_tarball(), "/tmp/ur5e_vla_sync.tar.gz")
    sftp.close()
    # --no-same-owner: files come with a Windows uid that the container cannot assign.
    _, out, err = ssh.exec_command(
        f"mkdir -p {REMOTE_DIR} && tar xzf /tmp/ur5e_vla_sync.tar.gz --no-same-owner -C {REMOTE_DIR} "
        f"&& rm /tmp/ur5e_vla_sync.tar.gz && echo SYNC-DONE"
    )
    o, e = out.read().decode(), err.read().decode()
    print(o.strip() or "(no output)")
    if e:
        print("stderr:", e.strip())


if __name__ == "__main__":
    main()
