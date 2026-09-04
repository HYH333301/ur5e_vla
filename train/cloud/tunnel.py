"""Local SSH port-forward to the paratera cloud container (paramiko, password from .env).

Forwards local_port -> container localhost:remote_port so the local eval client can
reach serve_policy.py running in the container (its ports are not public).

Usage: python tunnel.py [local_port] [remote_port]   (defaults: 8000 8000)
Keep running while evaluating; Ctrl-C stops.
"""
import select
import socket
import sys
import threading

from sshlib import connect

REMOTE_HOST = "127.0.0.1"  # as seen from the container


def pump(conn: socket.socket, chan) -> None:
    try:
        while True:
            r, _, _ = select.select([conn, chan], [], [], 60)
            if not r:
                continue
            if conn in r:
                data = conn.recv(4096)
                if not data:
                    break
                chan.sendall(data)
            if chan in r:
                data = chan.recv(4096)
                if not data:
                    break
                conn.sendall(data)
    except OSError:
        pass
    finally:
        conn.close()
        chan.close()


def main() -> None:
    local_port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    remote_port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

    ssh = connect()
    transport = ssh.get_transport()
    assert transport is not None

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", local_port))
    server.listen(8)
    print(f"tunnel 127.0.0.1:{local_port} -> container:{remote_port}", flush=True)

    def handle(conn: socket.socket):
        chan = transport.open_channel("direct-tcpip", (REMOTE_HOST, remote_port), conn.getpeername())
        if chan is None:
            conn.close()
            return
        pump(conn, chan)

    while True:
        conn, _ = server.accept()
        threading.Thread(target=handle, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    main()
