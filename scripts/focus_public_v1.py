"""Build a public EVREN version of FOCUS (fetal four-chamber view ultrasound; Zenodo 14597550,
CC BY 4.0; docs/datasets.md, FOCUS entry).

300 grey PNG images with one cardiac and one thorax mask each (separate PNGs, 0/255), in the
authors' split training / validation / testing (200 / 50 / 50). The authors' split is kept so that
results stay comparable with their experiments. Four images occur twice, byte-identical in pixels,
each time with a second, slightly different mask; the first copy is kept. training/021 is also
testing/008, so testing/008 is dropped from the test. Result 197 / 50 / 49.

Images are copied unchanged under new names (train_001.png, ...; the source numbers restart per
split). Masks become YOLO polygons (outer contour per part, holes joined by keyholes, as in
dsad_to_yolo_seg.py), classes 0 cardiac, 1 thorax, checked by drawing them again (Dice against the
source mask). The ellipse and rectangle annotations of the source are not used.

The build stops if a check fails: 300 source images, the four known duplicate pairs and no others,
split 197 / 50 / 49, both classes in every image, median polygon Dice >= 0.97 per class.

Run from the project root with the project venv:
    .venv/bin/python -I scripts/focus_public_v1.py
Writes data/focus_public/v1/ (images, labels, data.yaml, build_report.json) and
data/upload/focus_public_v1_yoloseg.zip
"""

import hashlib
import json
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from dsad_to_yolo_seg import mask_to_polygons  # noqa: E402

SRC = ROOT / "data" / "focus" / "raw" / "extracted"
OUT = ROOT / "data" / "focus_public" / "v1"
ZIP = ROOT / "data" / "upload" / "focus_public_v1_yoloseg.zip"
CLASSES = ["cardiac", "thorax"]
SPLITS = {"training": "train", "validation": "val", "testing": "test"}
DUPLICATES = {("training/020", "training/022"), ("training/039", "training/040"),
              ("training/108", "training/109"), ("training/021", "testing/008")}


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; remove it first so no stale files enter the ZIP")
    for part in ("images", "labels"):
        for s in SPLITS.values():
            (OUT / part / s).mkdir(parents=True)

    hashes, dropped, counts, polys = {}, [], Counter(), Counter()
    fidelity = defaultdict(list)
    sources = [(src, p) for src in SPLITS for p in sorted((SRC / src / "images").glob("*.png"))]
    assert len(sources) == 300, len(sources)
    for src, path in sources:
        key = f"{src}/{path.stem}"
        grey = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        h = hashlib.sha256(grey.tobytes()).hexdigest()
        if h in hashes:  # second copy: keep the first copy and its masks
            dropped.append([hashes[h], key])
            continue
        hashes[h] = key
        split = SPLITS[src]
        name = f"{split}_{path.stem}"
        shutil.copyfile(path, OUT / "images" / split / f"{name}.png")
        H, W = grey.shape
        lines = []
        for cid, cls in enumerate(CLASSES):
            mask = cv2.imread(str(SRC / src / "annfiles_mask" / f"{path.stem}-{cls}.png"), cv2.IMREAD_GRAYSCALE)
            assert mask.shape == (H, W) and set(np.unique(mask)) <= {0, 255}, (key, cls)
            mask = mask > 127
            ps = mask_to_polygons(mask)
            assert ps, f"{key}: no polygon for {cls}"
            redraw = np.zeros((H, W), np.uint8)
            for p in ps:
                cv2.fillPoly(redraw, [p.astype(np.int32)], 1)
                lines.append(f"{cid} " + " ".join(f"{v:.6f}" for v in (p / np.array([W, H])).ravel()))
                polys[cls] += 1
            r = redraw.astype(bool)
            fidelity[cls].append(2 * (r & mask).sum() / (r.sum() + mask.sum()))
        (OUT / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + "\n")
        counts[split] += 1

    assert {tuple(d) for d in dropped} == DUPLICATES, dropped
    assert counts == {"train": 197, "val": 50, "test": 49}, counts
    fid = {k: {"median": round(float(np.median(v)), 4), "min": round(float(np.min(v)), 4)}
           for k, v in fidelity.items()}
    assert all(f["median"] >= 0.97 for f in fid.values()), fid

    (OUT / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    report = {"source_images": 300, "images": sum(counts.values()), "dropped_copies": dropped,
              "split": dict(counts), "polygons": dict(polys), "polygon_dice": fid}
    (OUT / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")

    ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(OUT.rglob("*")):
            if f.is_file() and f.name != "build_report.json":
                z.write(f, f.relative_to(OUT))
    print(json.dumps(report, indent=2))
    print(f"wrote {ZIP.relative_to(ROOT)} ({ZIP.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
