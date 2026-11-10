"""Build a public EVREN version of one ARCADE task (coronary X-ray angiography; Zenodo 10390295,
CC0 1.0; plan: docs/plan-arcade-public-v1.md).

The source ships two tasks, each with an official split train 1,000 / val 200 / test 300 and COCO
polygons: syntax (25 SYNTAX coronary segments) and stenosis (1 class). In each task's images only
that task's structures are drawn, so the tasks stay separate datasets. The official split is kept.
Byte-identical images inside a task were annotated twice, differently: one copy is kept (the
lower-numbered one; the val copy for the train/val pair). Empty stenosis train images are missed
labels (218 is identical to val/58, which has 2 stenoses) and are left out. Category ids keep the
source order (1-25 -> 0-24; "stenosis" -> 0 in the stenosis task); names add the segment to the
SYNTAX number. Polygons are divided by the image size (512); images are hard-linked unchanged and
renamed <split>_<nnnn>.png because the source numbers repeat across splits.

The build stops if a check fails: the dropped images are exactly the duplicates and empty images
found, every image has a non-empty label file, all coordinates in [0, 1], every class used in
train, no two images share a SHA-256, official val/test lists unchanged.

Run from the project root with the project venv:
    .venv/bin/python -I scripts/arcade_public_v1.py syntax
Writes data/arcade_public/<task>/ (images, labels, data.yaml, build_report.json) and
data/upload/arcade_<task>_public_v1_yoloseg.zip
"""

import hashlib
import json
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "arcade" / "extract" / "arcade"
SPLITS = ("train", "val", "test")
# SYNTAX segment definitions, in the order of the source's category ids 1-25.
SEGMENTS = [
    "1 RCA proximal", "2 RCA mid", "3 RCA distal", "4 right posterior descending",
    "5 left main", "6 LAD proximal", "7 LAD mid", "8 LAD apical", "9 first diagonal",
    "9a first diagonal a", "10 second diagonal", "10a second diagonal a", "11 proximal circumflex",
    "12 intermediate-anterolateral", "12a obtuse marginal a", "13 distal circumflex",
    "14 left posterolateral", "14a left posterolateral a", "15 left posterior descending",
    "16 right posterolateral", "16a posterolateral branch a", "16b posterolateral branch b",
    "16c posterolateral branch c", "12b obtuse marginal b", "14b left posterolateral b",
]
CLASSES = {"syntax": SEGMENTS, "stenosis": ["stenosis"]}
DROP = {  # (split, source number), from the audit in docs/datasets.md
    "syntax": {("train", 793), ("train", 214), ("train", 972), ("train", 887), ("train", 612)},
    "stenosis": {("train", 799), ("train", 864), ("train", 916), ("train", 218), ("train", 738),
                 ("train", 888)},
}


def main(task):
    names = CLASSES[task]
    out = ROOT / "data" / "arcade_public" / task
    zip_path = ROOT / "data" / "upload" / f"arcade_{task}_public_v1_yoloseg.zip"
    if out.exists():
        raise SystemExit(f"{out} exists; remove it first so no stale files enter the ZIP")

    items = {}  # (split, number) -> (image path, [(class, polygon)])
    for s in SPLITS:
        coco = json.loads((SRC / task / s / "annotations" / f"{s}.json").read_text())
        source_names = {c["id"]: c["name"] for c in coco["categories"]}
        cls = ({c: c - 1 for c in range(1, 26)} if task == "syntax"
               else {c: 0 for c, n in source_names.items() if n == "stenosis"})
        polys = {i["id"]: [] for i in coco["images"]}
        for a in coco["annotations"]:
            assert len(a["segmentation"]) == 1 and len(a["segmentation"][0]) >= 6, a["id"]
            polys[a["image_id"]].append((cls[a["category_id"]], a["segmentation"][0]))
        for i in coco["images"]:
            assert (i["width"], i["height"]) == (512, 512), i
            items[(s, int(Path(i["file_name"]).stem))] = (SRC / task / s / "images" / i["file_name"],
                                                          polys[i["id"]])

    # The audit's drops must be exactly the in-task duplicates (all but one copy) and empty images.
    groups = {}
    for key in sorted(items, key=lambda k: (k[0] != "val", k[0] != "test", k[1])):
        groups.setdefault(hashlib.sha256(items[key][0].read_bytes()).hexdigest(), []).append(key)
    found = {k for g in groups.values() for k in g[1:]} | {k for k, (_, p) in items.items() if not p}
    assert found == DROP[task], (sorted(found ^ DROP[task]))
    assert not {s for s, _ in found} - {"train"}, found

    for part in ("images", "labels"):
        for s in SPLITS:
            (out / part / s).mkdir(parents=True)
    used, polygons = Counter(), 0
    train_used = set()
    for (s, n), (img, polys) in sorted(items.items()):
        if (s, n) in DROP[task]:
            continue
        assert Image.open(img).size == (512, 512), img
        stem = f"{s}_{n:04d}"
        os.link(img, out / "images" / s / f"{stem}.png")
        lines = []
        for c, p in polys:
            xy = [v / 512 for v in p]
            assert all(0 <= v <= 1 for v in xy), (img, min(p), max(p))
            lines.append(f"{c} " + " ".join(f"{v:.6f}" for v in xy) + "\n")
        (out / "labels" / s / f"{stem}.txt").write_text("".join(lines))
        used.update(names[c] for c, _ in polys)
        if s == "train":
            train_used.update(names[c] for c, _ in polys)
        polygons += len(polys)
    (out / "data.yaml").write_text("names:\n" + "".join(f"  {i}: \"{n}\"\n" for i, n in enumerate(names)))

    per_split = Counter()
    for s in SPLITS:
        imgs = {p.stem for p in (out / "images" / s).glob("*.png")}
        labs = {p.stem for p in (out / "labels" / s).glob("*.txt") if p.stat().st_size}
        assert imgs == labs, s
        per_split[s] = len(imgs)
    assert per_split["val"] == 200 and per_split["test"] == 300, per_split
    assert train_used == set(names), set(names) - train_used
    hashes = Counter(hashlib.sha256(p.read_bytes()).hexdigest() for p in (out / "images").rglob("*.png"))
    assert max(hashes.values()) == 1, "duplicate images"

    report = {"task": task, "images": sum(per_split.values()), "split": dict(per_split),
              "polygons": polygons, "dropped": [f"{s}/{n}" for s, n in sorted(DROP[task])],
              "per_class": {n: used[n] for n in names}}
    (out / "build_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zo:
        zo.write(out / "data.yaml", "data.yaml")
        for p in sorted((out / "images").rglob("*.png")) + sorted((out / "labels").rglob("*.txt")):
            zo.write(p, p.relative_to(out))
    with zipfile.ZipFile(zip_path) as zi:
        assert zi.testzip() is None
        n = len(zi.namelist())
    print(f"zip {zip_path.stat().st_size / 1e9:.2f} GB, {n} entries -> {zip_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "syntax")
