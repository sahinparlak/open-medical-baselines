"""Build the public EVREN version of the laparoscopic gastrectomy frames (Zhang et al., Mil Med Res
2026; figshare 30058882, CC BY 4.0; plan: docs/plan-gastrectomy-public-v1.md).

All 25 source classes in source order (category ids 1-25 -> 0-24; Roboflow's empty parent category 0
is dropped). Frames come from scripts/gastro_frames.load: Roboflow duplicates and frames without
annotations removed, date-free names <group>_<nnnn>.jpg, JPEG bytes unchanged (hard-linked from
data/gastrectomy_mmr2026/frames). Split by surgery group from splits/gastrectomy_surgeries_v1.csv.
COCO pixel polygons are copied to YOLO normalized coordinates (clipped to the frame); each part of
a multi-part segmentation is its own line.

The build stops if a check fails: 6,816 frames; split counts 4,256 / 1,296 / 1,264; every image has
a label file; every class index used; no two images share a SHA-256; no date pattern in a name.

Run from the project root with the project venv:
    .venv/bin/python -I scripts/gastrectomy_public_v1.py
Writes data/gastrectomy_public/v1/ (images, labels, data.yaml, build_report.json) and
data/upload/gastrectomy_public_v1_yoloseg.zip
"""

import csv
import hashlib
import json
import os
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gastro_frames import SPLIT, W, H, ZIP as SRC, load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "gastrectomy_public" / "v1"
ZIP = ROOT / "data" / "upload" / "gastrectomy_public_v1_yoloseg.zip"
EXPECTED = {"train": 4256, "val": 1296, "test": 1264}


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; remove it first so no stale files enter the ZIP")
    coco = json.loads(zipfile.ZipFile(SRC).read("train/_annotations.coco.json"))
    cats = sorted((c["id"], c["name"]) for c in coco["categories"] if c["id"] > 0)
    names = [n for _, n in cats]
    assert len(names) == 25 and len(set(names)) == 25, names
    index = {n: i for i, n in enumerate(names)}
    split_of = {r["group"]: r["split"] for r in csv.DictReader(open(SPLIT))}

    frames = load()
    for part in ("images", "labels"):
        for split in EXPECTED:
            (OUT / part / split).mkdir(parents=True)
    used, per_split, clipped, lines_total = Counter(), Counter(), 0, 0
    for f in frames:
        split = split_of[f["group"]]
        assert f["path"].exists(), f["path"]
        os.link(f["path"], OUT / "images" / split / f"{f['name']}.jpg")
        lines = []
        for name, polys in f["named"].items():
            for p in polys:
                q = p / [W, H]
                clipped += int(((q < 0) | (q > 1)).any())
                q = q.clip(0, 1)
                lines.append(f"{index[name]} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in q))
                used[name] += 1
        (OUT / "labels" / split / f"{f['name']}.txt").write_text("\n".join(lines) + "\n")
        per_split[split] += 1
        lines_total += len(lines)
    (OUT / "data.yaml").write_text("names:\n" + "".join(f"  {i}: \"{n}\"\n" for i, n in enumerate(names)))

    assert len(frames) == 6816, len(frames)
    assert dict(per_split) == EXPECTED, per_split
    for split in EXPECTED:
        imgs = {p.stem for p in (OUT / "images" / split).glob("*.jpg")}
        labs = {p.stem for p in (OUT / "labels" / split).glob("*.txt")}
        assert imgs == labs, split
    assert set(used) == set(names), set(names) - set(used)
    hashes = Counter(hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT / "images").rglob("*.jpg"))
    assert max(hashes.values()) == 1, "duplicate images"
    dated = [p.name for p in OUT.rglob("*") if re.search(r"\d{4}-?\d{2}-?\d{2}", p.name)]
    assert not dated, dated[:5]

    report = {"frames": len(frames), "split": dict(per_split), "polygons": lines_total,
              "source_polygons": len(coco["annotations"]), "polygons_clipped": clipped,
              "per_class": {n: used[n] for n in names}}
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
