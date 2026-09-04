"""SFTP upload to the paratera container: python putfile.py <local> <remote-abs-path>"""
import pathlib
import sys

from sshlib import connect

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: putfile.py <local> <remote>")
    local, remote = sys.argv[1], sys.argv[2]
    c = connect()
    parts = remote.strip("/").split("/")
    for i in range(2, len(parts)):  # mkdir -p everything up to the file
        c.exec_command(f"mkdir -p /{'/'.join(parts[:i])}")
    c.open_sftp().put(local, remote)
    _, out, _ = c.exec_command(f"ls -la {remote}")
    print(out.read().decode())
