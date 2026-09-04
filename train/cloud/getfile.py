"""SFTP download from the paratera container: python getfile.py <remote-abs-path> <local>

Reads creds from <repo>/.env, never prints them. Directories are downloaded recursively.
Run from local Windows (MSYS_NO_PATHCONV=1 if invoked from Git Bash).
"""
import pathlib
import stat
import sys

from sshlib import connect

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: getfile.py <remote> <local>")
    remote, local = sys.argv[1], sys.argv[2]
    c = connect()
    sftp = c.open_sftp()
    dst = pathlib.Path(local)
    n = 0

    def download(rpath: str, lpath: pathlib.Path):
        global n
        if str(rpath).endswith("/"):
            rpath = str(rpath).rstrip("/")
        if stat.S_ISDIR(sftp.stat(rpath).st_mode):
            lpath.mkdir(parents=True, exist_ok=True)
            for entry in sftp.listdir_attr(rpath):
                download(f"{rpath}/{entry.filename}", lpath / entry.filename)
        else:
            lpath.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(rpath, str(lpath))
            n += 1

    download(remote, dst)
    print(f"downloaded {n} files -> {dst}")
