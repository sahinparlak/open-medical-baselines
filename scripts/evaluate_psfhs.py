"""Per-class Dice of a PSFHS model (plan: docs/plan-psfhs-baseline.md).

Ground truth: the source pixel masks (data/psfhs/raw/PSFHS/label_mha; 1 pubic symphysis, 2 fetal
head), not the polygons of the EVREN build. Both structures are in every image, so a missed
structure scores Dice 0. Per class: mean and median Dice and the share of images where it was
predicted ("found"); the primary number is the mean of the two class means. A 95% bootstrap CI over
images is printed; images of one woman are not independent (no patient ids), so it is optimistic.

Run from the project root with the project venv (one prediction job at a time):
    .venv/bin/python scripts/evaluate_psfhs.py models/<run>/best.pt [--split val] [--device mps]
Writes results/<run>_psfhs_<split>.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_segmentation import ROOT, dice, patch_legacy_proto, predict, predicted_masks  # noqa: E402

DATA = ROOT / "data" / "psfhs_public" / "v1"
MASKS = ROOT / "data" / "psfhs" / "raw" / "PSFHS" / "label_mha"
CLASSES = ["pubic_symphysis", "fetal_head"]
IMGSZ = 512


def main():
    ap = argparse.ArgumentParser(description="Per-class Dice on the public PSFHS split.")
    ap.add_argument("weights")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    model = YOLO(args.weights)
    patch_legacy_proto(model)
    assert [model.names[i] for i in sorted(model.names)] == CLASSES, "model classes differ"
    run = Path(args.weights).parent.name
    images = sorted((DATA / "images" / args.split).glob("*.png"))

    rows = []
    for path, result in zip(images, predict(model, [str(p) for p in images], args.device, imgsz=IMGSZ)):
        h, w = result.orig_shape
        pred = predicted_masks(result, h, w, CLASSES)
        lab = sitk.GetArrayFromImage(sitk.ReadImage(str(MASKS / f"{path.stem}.mha")))
        assert lab.shape == (h, w), (path.name, lab.shape, (h, w))
        row = {"image": path.stem}
        for cid, cls in enumerate(CLASSES):
            g = lab == cid + 1
            p = pred.get(cls, np.zeros((h, w), bool))
            row[f"{cls}_found"] = int(p.any())
            row[f"{cls}_dice"] = dice(g, p)
        rows.append(row)

    out = ROOT / "results" / f"{run}_psfhs_{args.split}.csv"
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    rng = np.random.default_rng(0)
    d = np.array([[r[f"{c}_dice"] for c in CLASSES] for r in rows])
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    print(f"{args.weights}  PSFHS {args.split}: {len(rows)} images\n")
    print(f"{'class':<17}{'Dice mean':>10}{'median':>8}{'found':>8}{'Dice<0.5':>10}")
    for i, c in enumerate(CLASSES):
        found = np.mean([r[f"{c}_found"] for r in rows])
        print(f"{c:<17}{d[:, i].mean():>10.3f}{np.median(d[:, i]):>8.3f}{found:>8.1%}{int((d[:, i] < 0.5).sum()):>10}")
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\nPRIMARY mean of the 2 class means: {d.mean():.3f}  (95% bootstrap CI over images {lo:.3f}-{hi:.3f})")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
