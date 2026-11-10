"""Download one large file with parallel HTTP range requests, then verify its MD5.

Used for DSAD.zip on Figshare (~21 GB). Each part is written to its own offset in a
pre-allocated file; finished parts are recorded so an interrupted run can resume.

Run from the project root:
    python3 scripts/download_parallel.py <url> <output_path> <expected_md5> [workers]
"""

import hashlib
import json
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PART = 256 * 1024 * 1024
CHUNK = 4 * 1024 * 1024


def content_length(url):
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return int(resp.headers["Content-Range"].split("/")[-1])


def main(url, out, md5, workers):
    out = Path(out)
    state_path = out.with_suffix(out.suffix + ".parts.json")
    size = content_length(url)
    done = set(json.loads(state_path.read_text())) if state_path.exists() else set()
    if not out.exists():
        with open(out, "wb") as f:
            f.truncate(size)

    parts = [(i, i * PART, min(size, (i + 1) * PART) - 1) for i in range((size + PART - 1) // PART)]
    todo = [p for p in parts if p[0] not in done]
    lock = threading.Lock()
    progress = {"bytes": 0}
    start = time.time()
    print(f"{size / 1e9:.1f} GB, {len(parts)} parts, {len(todo)} to fetch, {workers} parallel", flush=True)

    def fetch(part):
        index, first, last = part
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={first}-{last}"})
                got = 0
                with urllib.request.urlopen(req, timeout=120) as resp, open(out, "r+b") as f:
                    f.seek(first)
                    while chunk := resp.read(CHUNK):
                        f.write(chunk)
                        got += len(chunk)
                        with lock:
                            progress["bytes"] += len(chunk)
                # A connection closed early ends the read without an error.
                if got != last - first + 1:
                    raise IOError(f"short read: {got} of {last - first + 1} bytes")
                break
            except Exception as exc:  # retry transient network errors
                print(f"  part {index} attempt {attempt + 1} failed: {exc}", flush=True)
                time.sleep(5)
        else:
            raise RuntimeError(f"part {index} failed")
        with lock:
            done.add(index)
            state_path.write_text(json.dumps(sorted(done)))
            rate = progress["bytes"] / (time.time() - start) / 1e6
            print(f"  part {index} done ({len(done)}/{len(parts)})  {rate:.1f} MB/s", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch, todo))

    print("verifying MD5 ...", flush=True)
    h = hashlib.md5()
    with open(out, "rb") as f:
        while block := f.read(64 * 1024 * 1024):
            h.update(block)
    ok = h.hexdigest() == md5
    print(f"MD5 {'OK' if ok else 'MISMATCH'}: {h.hexdigest()}", flush=True)
    if ok:
        state_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 8)
