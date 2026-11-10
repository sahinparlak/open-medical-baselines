"""Build the public EVREN version of LapGC-KVAD-30 (docs/plan-lapgc-public-v1.md) from
data/lapgc_yolo/v1: the same frames and labels, except that the six frames present twice in the
source (once in scene S6, once in S7, each copy labelled with its own scene's vessels) are merged
into the S6 frame with the union of both label files.

Images are hard-linked from v1 (never modified); labels are copied. The build stops if any check
fails: 6 pairs, all train, same case, S6 + S7; no duplicate images left; expected split counts;
every image has a label file; polygon lines = v1; val and test unchanged.

Run from the project root with the project venv:
    .venv/bin/python -I scripts/lapgc_public_v1.py
Writes data/lapgc_yolo/public_v1/ (images, labels, data.yaml, merges.csv) and
data/upload/lapgc_kvad30_vessels_public_v1_yoloseg.zip
"""

import csv
import filecmp
import hashlib
import os
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "lapgc_yolo" / "v1"
OUT = ROOT / "data" / "lapgc_yolo" / "public_v1"
ZIP = ROOT / "data" / "upload" / "lapgc_kvad30_vessels_public_v1_yoloseg.zip"
EXPECTED = {"train": 3537, "val": 880, "test": 879}


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def duplicate_groups(images):
    by = defaultdict(list)
    for p in sorted(images.rglob("*.jpg")):
        by[sha256(p)].append(p.relative_to(images))
    return [v for v in by.values() if len(v) > 1]


def label_lines(labels):
    return sum(len(p.read_text().splitlines()) for p in labels.rglob("*.txt"))


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; remove it first so no stale files enter the ZIP")
    groups = duplicate_groups(SRC / "images")
    assert len(groups) == 6 and all(len(g) == 2 for g in groups), groups
    pairs = []
    for g in groups:
        s6, s7 = sorted(g, key=lambda p: p.stem.split("_")[2])
        _, case6, scene6, _ = s6.stem.split("_")
        _, case7, scene7, _ = s7.stem.split("_")
        assert s6.parent.name == s7.parent.name == "train" and case6 == case7, g
        assert (scene6, scene7) == ("S6", "S7"), g
        pairs.append((s6, s7))

    shutil.copytree(SRC / "images", OUT / "images", copy_function=os.link)
    shutil.copytree(SRC / "labels", OUT / "labels")
    shutil.copy2(SRC / "data.yaml", OUT / "data.yaml")
    rows = []
    for s6, s7 in sorted(pairs):
        keep, drop = (OUT / "labels" / p.with_suffix(".txt") for p in (s6, s7))
        a, b = keep.read_text().splitlines(), drop.read_text().splitlines()
        merged = a + [x for x in b if x not in a]
        keep.write_text("\n".join(merged) + "\n")
        drop.unlink()
        (OUT / "images" / s7).unlink()
        rows.append({"kept": s6.stem, "dropped": s7.stem, "lines_kept": len(a), "lines_dropped": len(b),
                     "lines_merged": len(merged)})
    with open(OUT / "merges.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    assert not duplicate_groups(OUT / "images"), "duplicate images left"
    for split, n in EXPECTED.items():
        imgs = {p.stem for p in (OUT / "images" / split).glob("*.jpg")}
        labs = {p.stem for p in (OUT / "labels" / split).glob("*.txt")}
        assert len(imgs) == n and imgs == labs, (split, len(imgs), len(labs))
    lines_v1, lines_out = label_lines(SRC / "labels"), label_lines(OUT / "labels")
    assert lines_out == lines_v1, (lines_out, lines_v1)
    for split in ("val", "test"):
        for part in ("images", "labels"):
            cmp = filecmp.dircmp(SRC / part / split, OUT / part / split)
            assert not (cmp.left_only or cmp.right_only), (split, part)
            _, mismatch, errors = filecmp.cmpfiles(SRC / part / split, OUT / part / split, cmp.common, shallow=False)
            assert not (mismatch or errors), (split, part, mismatch, errors)
    for r in rows:
        print(f"{r['kept']} + {r['dropped']}: {r['lines_kept']} + {r['lines_dropped']} -> {r['lines_merged']} lines")
    print(f"images {EXPECTED} = {sum(EXPECTED.values())}; polygon lines {lines_out} (= v1); val/test = v1")

    ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED) as zo:
        zo.write(OUT / "data.yaml", "data.yaml")
        for p in sorted((OUT / "images").rglob("*.jpg")) + sorted((OUT / "labels").rglob("*.txt")):
            zo.write(p, p.relative_to(OUT))
    with zipfile.ZipFile(ZIP) as zi:
        assert zi.testzip() is None
        names = zi.namelist()
    print(f"zip {ZIP.stat().st_size / 1e9:.2f} GB, {len(names)} entries -> {ZIP.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
