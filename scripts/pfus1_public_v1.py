"""Build a public EVREN version of PFUS1 (pelvic floor ultrasound; Zenodo 10800787, CC BY 4.0;
plan: docs/plan-pfus1-public-v1.md).

Each folder P000-P109 is one transperineal ultrasound video; every frame has one polygon for each of
8 structures (the sonographer drew 12-51 key frames per video, the rest are linear interpolations,
per the paper). Six folder pairs hold the same video byte for byte; the lower-numbered folder is
kept. Videos are split 72 / 16 / 16 (train / val / test, seeded), so frames of one video never
cross splits. Class order is fixed (the order in the source readme); the JSON files list the
structures in varying order. Points outside the frame (up to 17 px) are clipped to the border;
coordinates are divided by each frame's own size, which varies within a video. Images are
hard-linked unchanged and named <folder>_<frame>.png.

The build stops if a check fails: the dropped folders are exactly the duplicate ones, 104 videos and
13,906 frames, 8 distinct structures per frame, all coordinates in [0, 1], no video in two splits,
no two images share a SHA-256.

Run from the project root with the project venv:
    .venv/bin/python -I scripts/pfus1_public_v1.py
Writes data/pfus1_public/ (images, labels, data.yaml, build_report.json) and
data/upload/pfus1_public_v1_yoloseg.zip
"""

import hashlib
import json
import os
import random
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "pfus1" / "extract" / "data"
OUT = ROOT / "data" / "pfus1_public"
ZIP = ROOT / "data" / "upload" / "pfus1_public_v1_yoloseg.zip"
CLASSES = ["Pubis", "Urethra", "Bladder", "Uterus", "Vagina", "Anus", "Rectum", "Levator ani muscle"]
DROP = {"P011", "P104", "P022", "P024", "P061", "P090"}  # second copies of P003, P010, P013, P023, P057, P087
N_VAL, N_TEST, SEED = 16, 16, 0


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; remove it first so no stale files enter the ZIP")
    index = {n: i for i, n in enumerate(CLASSES)}
    folders = sorted(p.name for p in SRC.iterdir() if p.is_dir())

    first = {}  # SHA-256 of a folder's frames -> folder: identical videos map to one key
    for f in folders:
        h = hashlib.sha256(b"".join(hashlib.sha256(p.read_bytes()).digest()
                                    for p in sorted((SRC / f).glob("*.png")))).hexdigest()
        first.setdefault(h, []).append(f)
    found = {f for g in first.values() for f in g[1:]}
    assert found == DROP, sorted(found ^ DROP)

    videos = [f for f in folders if f not in DROP]
    rng = random.Random(SEED)
    shuffled = rng.sample(videos, len(videos))
    split = {**{v: "test" for v in shuffled[:N_TEST]},
             **{v: "val" for v in shuffled[N_TEST:N_TEST + N_VAL]},
             **{v: "train" for v in shuffled[N_TEST + N_VAL:]}}
    for part in ("images", "labels"):
        for s in ("train", "val", "test"):
            (OUT / part / s).mkdir(parents=True)

    frames, clipped = Counter(), 0
    for v in videos:
        s = split[v]
        for png in sorted((SRC / v).glob("*.png")):
            w, h = Image.open(png).size
            objs = json.loads(png.with_suffix(".json").read_text())
            assert sorted(o["label"] for o in objs) == sorted(CLASSES), png
            lines = []
            for o in sorted(objs, key=lambda o: index[o["label"]]):
                xy = []
                for x, y in o["pol"]:
                    cx, cy = min(max(x, 0), w), min(max(y, 0), h)
                    clipped += (cx, cy) != (x, y)
                    xy += [cx / w, cy / h]
                assert len(xy) >= 6 and all(0 <= c <= 1 for c in xy), png
                lines.append(f"{index[o['label']]} " + " ".join(f"{c:.6f}" for c in xy) + "\n")
            stem = f"{v}_{png.stem}"
            os.link(png, OUT / "images" / s / f"{stem}.png")
            (OUT / "labels" / s / f"{stem}.txt").write_text("".join(lines))
            frames[s] += 1
    (OUT / "data.yaml").write_text("names:\n" + "".join(f"  {i}: \"{n}\"\n" for i, n in enumerate(CLASSES)))

    assert len(videos) == 104 and sum(frames.values()) == 13906, (len(videos), frames)
    for s in ("train", "val", "test"):
        imgs = {p.stem for p in (OUT / "images" / s).glob("*.png")}
        labs = {p.stem for p in (OUT / "labels" / s).glob("*.txt")}
        assert imgs == labs and len(imgs) == frames[s], s
    vids = {s: {p.stem.split("_")[0] for p in (OUT / "images" / s).glob("*.png")} for s in frames}
    assert not (vids["train"] & vids["val"] or vids["train"] & vids["test"] or vids["val"] & vids["test"])
    hashes = Counter(hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT / "images").rglob("*.png"))
    assert max(hashes.values()) == 1, "duplicate images"

    report = {"videos": {s: len(vids[s]) for s in ("train", "val", "test")},
              "frames": {s: frames[s] for s in ("train", "val", "test")}, "polygons": 8 * sum(frames.values()),
              "clipped_points": clipped, "dropped_folders": sorted(DROP),
              "val_videos": sorted(vids["val"]), "test_videos": sorted(vids["test"])}
    (OUT / "build_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED) as zo:
        zo.write(OUT / "data.yaml", "data.yaml")
        for p in sorted((OUT / "images").rglob("*.png")) + sorted((OUT / "labels").rglob("*.txt")):
            zo.write(p, p.relative_to(OUT))
    with zipfile.ZipFile(ZIP) as zi:
        assert zi.testzip() is None
        n = len(zi.namelist())
    print(f"zip {ZIP.stat().st_size / 1e9:.2f} GB, {n} entries -> {ZIP.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
