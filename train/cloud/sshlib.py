"""SSH helper for the paratera cloud container. Reads creds from <repo>/.env, never prints them.

Usage: python sshlib.py "<remote command>"
Slow commands (>280 s) must be nohup'd on the remote side, e.g. via train.sh.
"""
import io, pathlib, sys

import paramiko

ENV = pathlib.Path(__file__).resolve().parents[2] / ".env"


def _creds():
    host = port = user = pw = None
    for line in io.open(ENV, encoding="utf-8"):
        line = line.strip()
        if line.startswith("IP地址") or line.startswith("IP"):
            host = line.split(":", 1)[1].strip()
        elif line.startswith("端口"):
            port = int(line.split(":", 1)[1].strip())
        elif line.startswith("用户名"):
            user = line.split(":", 1)[1].strip()
        elif line.startswith("密码"):
            pw = line.split(":", 1)[1].strip()
    assert all([host, port, user, pw]), f"incomplete SSH creds in {ENV}"
    return host, port, user, pw


def connect():
    host, port, user, pw = _creds()
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, port=port, username=user, password=pw, timeout=30, look_for_keys=False, allow_agent=False)
    return c


def run(c, cmd, timeout=280):
    _, out, err = c.exec_command(cmd, timeout=timeout)
    o, e = out.read().decode("utf-8", "replace"), err.read().decode("utf-8", "replace")
    rc = out.channel.recv_exit_status()
    return rc, o, e


if __name__ == "__main__":
    c = connect()
    rc, o, e = run(c, " ".join(sys.argv[1:]))
    sys.stdout.buffer.write(o.encode("utf-8", "replace"))
    if e:
        sys.stderr.buffer.write(e.encode("utf-8", "replace"))
    sys.exit(rc)
