"""Build a public EVREN version of VISEM-Tracking (semen microscopy video; Zenodo 7293726, CC BY 4.0;
plan: docs/plan-visem-public-v1.md).

20 videos (one per man) of 30 s, 640 x 480 frames with YOLO boxes of sperm, clusters and
small/pinhead sperm. All labelled frames are kept; the 174 unlabelled frames of video 23 (823-1107,
sperm visible but not boxed) are left out. Split by video 14 / 3 / 3 (train / val / test): seeded
draws until val and test each hold at least one cluster and one pinhead video (these classes are
only 20 and 35 distinct objects). Boxes and images unchanged (hard links; names <video>_frame_<n>).
Track ids (labels_ftid) are not carried: EVREN trains no trackers.

The build stops if a check fails: 29,196 frames and 656,335 boxes, every frame labelled, all
coordinates inside [0, 1], no video in two splits.

Run from the project root with the project venv:
    .venv/bin/python -I scripts/visem_public_v1.py
Writes data/visem_public/ (images, labels, data.yaml, build_report.json) and
data/upload/visem_public_v1_yolodet.zip
"""

import json
import os
import random
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "visem_tracking" / "extract" / "VISEM_Tracking_Train_v4" / "Train"
OUT = ROOT / "data" / "visem_public"
ZIP = ROOT / "data" / "upload" / "visem_public_v1_yolodet.zip"
CLASSES = ["Sperm", "Cluster", "Small or pinhead"]  # the source's class mapping 0, 1, 2
N_VAL, N_TEST, SEED = 3, 3, 0


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; remove it first so no stale files enter the ZIP")
    videos = sorted((p.name for p in SRC.iterdir() if p.is_dir()), key=int)
    classes_in = {}
    for v in videos:
        c = set()
        for f in (SRC / v / "labels").glob("*.txt"):
            c.update(int(l.split()[0]) for l in f.read_text().splitlines() if l.strip())
        classes_in[v] = c

    rng = random.Random(SEED)
    for draw in range(1, 1001):
        order = rng.sample(videos, len(videos))
        test, val = order[:N_TEST], order[N_TEST:N_TEST + N_VAL]
        if all({1, 2} <= set().union(*(classes_in[v] for v in part)) for part in (val, test)):
            break
    else:
        raise SystemExit("no split with both rare classes in val and test")
    split = {v: "test" for v in test} | {v: "val" for v in val} | {v: "train" for v in order[N_TEST + N_VAL:]}

    for part in ("images", "labels"):
        for s in ("train", "val", "test"):
            (OUT / part / s).mkdir(parents=True)
    frames, boxes, per_class, skipped = Counter(), 0, Counter(), 0
    for v in videos:
        s = split[v]
        for img in (SRC / v / "images").glob("*.jpg"):
            lab = SRC / v / "labels" / f"{img.stem}.txt"
            if not lab.exists():
                skipped += 1
                continue
            lines = [l for l in lab.read_text().splitlines() if l.strip()]
            assert lines, lab
            for l in lines:
                c, x, y, w, h = l.split()
                x, y, w, h = map(float, (x, y, w, h))
                assert 0 <= x - w / 2 + 1e-6 and x + w / 2 <= 1 + 1e-6 and 0 <= y - h / 2 + 1e-6 and y + h / 2 <= 1 + 1e-6, lab
                per_class[CLASSES[int(c)]] += 1
            os.link(img, OUT / "images" / s / img.name)
            (OUT / "labels" / s / f"{img.stem}.txt").write_text("\n".join(lines) + "\n")
            frames[s] += 1
            boxes += len(lines)
    (OUT / "data.yaml").write_text("names:\n" + "".join(f"  {i}: \"{n}\"\n" for i, n in enumerate(CLASSES)))

    assert sum(frames.values()) == 29196 and boxes == 656335 and skipped == 174, (frames, boxes, skipped)
    vids = {s: {p.stem.split("_")[0] for p in (OUT / "images" / s).glob("*.jpg")} for s in frames}
    assert not (vids["train"] & vids["val"] or vids["train"] & vids["test"] or vids["val"] & vids["test"])
    split_classes = {}
    for s in ("train", "val", "test"):
        c = Counter()
        for f in (OUT / "labels" / s).glob("*.txt"):
            c.update(CLASSES[int(l.split()[0])] for l in f.read_text().splitlines())
        split_classes[s] = dict(c)

    report = {"split_draw": draw, "videos": {s: sorted(vids[s], key=int) for s in ("train", "val", "test")},
              "frames": dict(frames), "boxes": boxes, "per_class": dict(per_class),
              "per_split_class": split_classes, "skipped_unlabelled": skipped}
    (OUT / "build_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED) as zo:
        zo.write(OUT / "data.yaml", "data.yaml")
        for p in sorted((OUT / "images").rglob("*.jpg")) + sorted((OUT / "labels").rglob("*.txt")):
            zo.write(p, p.relative_to(OUT))
    with zipfile.ZipFile(ZIP) as zi:
        assert zi.testzip() is None
        n = len(zi.namelist())
    print(f"zip {ZIP.stat().st_size / 1e9:.2f} GB, {n} entries -> {ZIP.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
