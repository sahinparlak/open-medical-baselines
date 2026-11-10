"""Build a public EVREN version of PSFHS (intrapartum transperineal ultrasound; Zenodo 10969427,
CC BY 4.0; plan: docs/plan-psfhs-public-v1.md).

1,358 images (256 x 256, grey) with pixel masks of the pubic symphysis (value 1) and fetal head
(value 2) from two centres. Six images occur twice, byte for byte, each time with a second, slightly different mask (Dice
0.87-1.00); the first copy is kept and the second dropped (all six pairs lie within one split).
Images are written unchanged as PNG; masks become YOLO polygons (outer
contour per part, holes joined by keyholes, as in dsad_to_yolo_seg.py) and are checked by drawing
them again (Dice against the source mask).

Split (no patient ids in the source; a woman's images have consecutive numbers):
- test = Jinan (ObEye; images without the grey scale bar at the left edge), 313;
- val = the contiguous Zhujiang block 04750-04888 (cut where neighbouring images differ most);
- train = the rest of Zhujiang (Esaote; scale bar present).

The build stops if a check fails: 1,358 source images, 6 dropped copies, 1,044 / 308 per centre,
split 905 / 139 / 308,
label values {0, 1, 2}, every image labelled with both classes, no identical images, median polygon
Dice >= 0.97 per class.

Run from the project root with the project venv:
    .venv/bin/python -I scripts/psfhs_public_v1.py
Writes data/psfhs_public/v1/ (images, labels, data.yaml, build_report.json) and
data/upload/psfhs_public_v1_yoloseg.zip
"""

import hashlib
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from dsad_to_yolo_seg import mask_to_polygons  # noqa: E402

SRC = ROOT / "data" / "psfhs" / "raw" / "PSFHS"
OUT = ROOT / "data" / "psfhs_public" / "v1"
ZIP = ROOT / "data" / "upload" / "psfhs_public_v1_yoloseg.zip"
CLASSES = ["pubic_symphysis", "fetal_head"]  # source label values 1 and 2
VAL_FIRST, VAL_LAST = 4750, 4888


def has_scale_bar(grey):
    """Zhujiang (Esaote) images carry a grey scale bar at the left edge (mean ~51 there, else 0)."""
    return grey[5:60, 1:7].mean() > 40


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; remove it first so no stale files enter the ZIP")
    images = sorted((SRC / "image_mha").glob("*.mha"))
    assert len(images) == 1358, len(images)

    for part in ("images", "labels"):
        for s in ("train", "val", "test"):
            (OUT / part / s).mkdir(parents=True)

    counts, centre, polys, hashes, dropped = Counter(), Counter(), Counter(), {}, []
    fidelity = defaultdict(list)
    for path in images:
        n = int(path.stem)
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))          # 3 x H x W, equal channels
        lab = sitk.GetArrayFromImage(sitk.ReadImage(str(SRC / "label_mha" / path.name)))
        assert img.shape == (3, 256, 256) and lab.shape == (256, 256), (path.name, img.shape, lab.shape)
        assert set(np.unique(lab)) == {0, 1, 2}, (path.name, np.unique(lab))
        zh = has_scale_bar(img[0].astype(float))
        split = "test" if not zh else "val" if VAL_FIRST <= n <= VAL_LAST else "train"

        rgb = np.ascontiguousarray(img.transpose(1, 2, 0))
        h = hashlib.sha256(rgb.tobytes()).hexdigest()
        if h in hashes:  # second copy of an image: keep the first copy and its mask
            dropped.append([hashes[h], path.stem])
            continue
        hashes[h] = path.stem
        centre["zhujiang" if zh else "jinan"] += 1
        counts[split] += 1
        cv2.imwrite(str(OUT / "images" / split / f"{path.stem}.png"), rgb[:, :, ::-1])

        lines = []
        for cid, name in enumerate(CLASSES):
            mask = lab == cid + 1
            ps = mask_to_polygons(mask)
            assert ps, f"{path.name}: no polygon for {name}"
            redraw = np.zeros((256, 256), np.uint8)
            for p in ps:
                cv2.fillPoly(redraw, [p.astype(np.int32)], 1)
                lines.append(f"{cid} " + " ".join(f"{v:.6f}" for v in (p / 256.0).ravel()))
                polys[name] += 1
            r = redraw.astype(bool)
            fidelity[name].append(2 * (r & mask).sum() / (r.sum() + mask.sum()))
        (OUT / "labels" / split / f"{path.stem}.txt").write_text("\n".join(lines) + "\n")

    assert len(dropped) == 6, dropped
    assert centre == {"zhujiang": 1044, "jinan": 308}, centre
    assert counts == {"train": 905, "val": 139, "test": 308}, counts
    fid = {k: {"median": round(float(np.median(v)), 4), "min": round(float(np.min(v)), 4)}
           for k, v in fidelity.items()}
    assert all(f["median"] >= 0.97 for f in fid.values()), fid

    (OUT / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    report = {"source_images": 1358, "images": 1352, "dropped_copies": dropped, "centres": dict(centre), "split": dict(counts),
              "val_block": [VAL_FIRST, VAL_LAST], "polygons": dict(polys), "polygon_dice": fid}
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
