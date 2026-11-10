"""Build a public EVREN version of one MODUS subset (urine sediment microscopy; figshare 31166800,
CC BY 4.0; plan: docs/plan-modus-public-v1.md).

The source ships two subsets with their own class lists (10x: 8 classes, 40x: 22 classes; order
as in the source's VOC2YOLO.py). Pascal VOC boxes become YOLO detection boxes; images are
hard-linked from the extracted source unchanged. The official test set is kept; a validation set
of 15% of the official train images (seeded, by image: the source has no sample ids) is carved
out because EVREN training needs one. "difficult" flags are not carried (YOLO has none).
Byte-identical images (40x: five train/test pairs, a leak in the source split; 10x: two pairs in
train) keep one copy, the test copy when there is one.

The build stops if a check fails: image and box counts equal the source, XML sizes equal the image
sizes, no box outside the image, every class used, every image has a label file, no two images
share a SHA-256, all official test images in test.

Run from the project root with the project venv:
    .venv/bin/python -I scripts/modus_public_v1.py 40x
Writes data/modus_public/<subset>/ (images, labels, data.yaml, build_report.json) and
data/upload/modus_<subset>_public_v1_yolodet.zip
"""

import hashlib
import json
import os
import random
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "modus" / "extract"
CLASSES = {
    "40x": ["Erythrocyte", "Crenated erythrocyte", "Leukocyte", "Macrophage",
            "Calcium oxalate crystal (bipyramid)", "Calcium oxalate crystal (oval)",
            "Calcium oxalate crystal (spindle)", "Uric acid crystal", "Triple phosphate crystal",
            "Bilirubin crystal", "Granular cast", "Hyaline cast", "Waxy cast", "Cellular cast",
            "Muddy brown cast", "Fatty cast", "Squamous epithelial cell", "Transitional epithelial cell",
            "Fungal spore", "Sperm", "Air bubble", "Oval fat body"],
    "10x": ["Uric acid crystal", "Calcium oxalate crystal (bipyramid)", "Triple phosphate crystal",
            "Hyaline cast", "Granular cast", "Muddy brown cast", "Waxy cast", "Air bubble"],
}
VAL_SHARE, SEED = 0.15, 0


def read(xml_path, index):
    t = ET.parse(xml_path).getroot()
    w, h = int(t.find("size/width").text), int(t.find("size/height").text)
    lines = []
    for o in t.findall("object"):
        b = o.find("bndbox")
        x0, y0, x1, y1 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h, (xml_path, x0, y0, x1, y1)
        lines.append((index[o.find("name").text.strip()], (x0 + x1) / 2 / w, (y0 + y1) / 2 / h,
                      (x1 - x0) / w, (y1 - y0) / h))
    return (w, h), lines


def main(subset):
    names = CLASSES[subset]
    index = {n: i for i, n in enumerate(names)}
    src = SRC / f"{subset}_magnification"
    out = ROOT / "data" / "modus_public" / subset
    zip_path = ROOT / "data" / "upload" / f"modus_{subset}_public_v1_yolodet.zip"
    if out.exists():
        raise SystemExit(f"{out} exists; remove it first so no stale files enter the ZIP")

    train = sorted(p.stem for p in (src / "labels" / "train").glob("*.xml"))
    test = sorted(p.stem for p in (src / "labels" / "test").glob("*.xml"))
    # Byte-identical images: keep the test copy (official test untouched), else the first one.
    groups = {}
    for official, stems in (("test", test), ("train", train)):
        for stem in stems:
            h = hashlib.sha256((src / "images" / official / f"{stem}.jpg").read_bytes()).hexdigest()
            groups.setdefault(h, []).append(stem)
    dropped = sorted(s for g in groups.values() for s in g[1:])
    assert not set(dropped) & set(test), dropped
    train = [s for s in train if s not in dropped]
    val = sorted(random.Random(SEED).sample(train, round(VAL_SHARE * len(train))))
    split = {**{s: "train" for s in train}, **{s: "val" for s in val}, **{s: "test" for s in test}}
    for part in ("images", "labels"):
        for s in ("train", "val", "test"):
            (out / part / s).mkdir(parents=True)

    used, boxes = Counter(), 0
    for stem, s in sorted(split.items()):
        official = "test" if s == "test" else "train"
        img = src / "images" / official / f"{stem}.jpg"
        size, lines = read(src / "labels" / official / f"{stem}.xml", index)
        assert Image.open(img).size == size, (stem, size)
        os.link(img, out / "images" / s / img.name)
        (out / "labels" / s / f"{stem}.txt").write_text(
            "".join(f"{c} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n" for c, x, y, bw, bh in lines))
        used.update(names[c] for c, *_ in lines)
        boxes += len(lines)
    (out / "data.yaml").write_text("names:\n" + "".join(f"  {i}: \"{n}\"\n" for i, n in enumerate(names)))

    per_split = Counter(split.values())
    assert sum(per_split.values()) == len(train) + len(test) == len(groups)
    assert set(used) == set(names), set(names) - set(used)
    for s in ("train", "val", "test"):
        imgs = {p.stem for p in (out / "images" / s).glob("*.jpg")}
        labs = {p.stem for p in (out / "labels" / s).glob("*.txt")}
        assert imgs == labs and len(imgs) == per_split[s], s
    assert {p.stem for p in (out / "images" / "test").glob("*.jpg")} == set(test)
    hashes = Counter(hashlib.sha256(p.read_bytes()).hexdigest() for p in (out / "images").rglob("*.jpg"))
    assert max(hashes.values()) == 1, "duplicate images"

    report = {"subset": subset, "images": len(split), "split": dict(per_split), "boxes": boxes,
              "dropped_duplicates": dropped, "per_class": {n: used[n] for n in names}}
    (out / "build_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zo:
        zo.write(out / "data.yaml", "data.yaml")
        for p in sorted((out / "images").rglob("*.jpg")) + sorted((out / "labels").rglob("*.txt")):
            zo.write(p, p.relative_to(out))
    with zipfile.ZipFile(zip_path) as zi:
        assert zi.testzip() is None
        n = len(zi.namelist())
    print(f"zip {zip_path.stat().st_size / 1e9:.2f} GB, {n} entries -> {zip_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "40x")
