"""Fetch gs://openpi-assets/checkpoints/pi05_base with aria2 multi-connection into the openpi cache.

GCS via gsutil/fsspec crawls at ~250KB/s in the container network; the plain HTTPS
JSON API + aria2c (x16 connections) reaches ~1.1MB/s aggregate. Prints GCS-FETCH-DONE
on success (train_chain.sh watches for that line).
"""
import json, pathlib, subprocess, urllib.request

PREFIX = "checkpoints/pi05_base/"
CACHE = pathlib.Path("/root/shared-nvme/openpi_cache/openpi-assets")
STAGE = pathlib.Path("/root/shared-nvme/gcs_stage")

def list_objects(prefix):
    out, token = [], None
    while True:
        url = f"https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix={prefix}&maxResults=1000"
        if token:
            url += f"&pageToken={token}"
        with urllib.request.urlopen(url, timeout=90) as r:
            data = json.load(r)
        out += [it["name"] for it in data.get("items", [])]
        token = data.get("nextPageToken")
        if not token:
            return out

names = list_objects(PREFIX)
print(len(names), "objects", flush=True)
STAGE.mkdir(parents=True, exist_ok=True)
inp = STAGE / "aria2.in"
with open(inp, "w") as f:
    for n in names:
        f.write(f"https://storage.googleapis.com/openpi-assets/{n}\n")
        f.write(f"  dir={CACHE}/{'/'.join(n.split('/')[:-1])}\n")
        f.write(f"  out={n.split('/')[-1]}\n")
rc = subprocess.run(["aria2c", "-i", str(inp), "-x", "16", "-s", "16", "-j", "3",
                     "--continue=true", "--max-tries=0", "--retry-wait=5",
                     "--summary-interval=30", "--console-log-level=warn"]).returncode
print("aria2 rc:", rc, flush=True)
if rc == 0:
    print("GCS-FETCH-DONE", flush=True)
