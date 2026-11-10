"""Polyp Dice and IoU of a HyperKvasir model (plan: docs/plan-hyperkvasir-baseline.md).

Ground truth: the source JPEG masks (data/hyperkvasir/raw/.../masks, threshold 127), not the
polygons of the EVREN build. A polyp is in every image, so a miss scores Dice 0. The predicted mask
is the union of all polyp instances at the original size (confidence 0.25, fixed in the plan).

Per image: Dice, IoU, found (any predicted pixel), false-positive area (predicted pixels outside the
polyp, as a fraction of the image). Reported: mean Dice (primary) and mean IoU, by polyp size
tertile (cuts from the build), the share with Dice < 0.5, and 95% CIs from 2,000 bootstrap
resamples of near-duplicate groups (annotations/hyperkvasir/near_duplicate_pairs.csv).

Trivial baseline (--baseline): the "centre blob", the polyp frequency map of the training masks
(resized to 256 x 256) thresholded where the mean training Dice is highest, resized to each image
and scored the same way.

Run from the project root with the project venv (one prediction job at a time):
    .venv/bin/python -I scripts/evaluate_hyperkvasir.py models/<run>/best.pt [--split val]
    .venv/bin/python -I scripts/evaluate_hyperkvasir.py --baseline [--split val]
Writes results/<run>_hyperkvasir_<split>.csv (run = "centre-blob" for the baseline)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_segmentation import ROOT, dice  # noqa: E402
from hyperkvasir_polyp_public_v1 import groups_from_pairs  # noqa: E402

DATA = ROOT / "data" / "hyperkvasir_polyp_public" / "v1"
MASKS = ROOT / "data" / "hyperkvasir" / "raw" / "extracted" / "segmented-images" / "masks"
IMGSZ = 640
CONF = 0.25
SIZES = ["small", "medium", "large"]


def gt_mask(name):
    return cv2.imread(str(MASKS / f"{name}.jpg"), cv2.IMREAD_GRAYSCALE) > 127


def iou(g, p):
    union = np.logical_or(g, p).sum()
    return round(float(np.logical_and(g, p).sum() / union), 4) if union else 1.0


def centre_blob(train_names):
    """Polyp frequency map of the training masks at 256 x 256, thresholded where the mean training
    Dice is highest (a fixed threshold of 0.5 leaves almost nothing: polyps are not consistently
    central). Only training masks are used."""
    gts = [cv2.resize(gt_mask(n).astype(np.float32), (256, 256), interpolation=cv2.INTER_AREA) > 0.5
           for n in train_names]
    freq = np.mean(gts, axis=0)
    best = max(np.arange(0.02, 0.52, 0.02),
               key=lambda t: np.mean([dice(g, freq >= t) for g in gts]))
    print(f"centre blob: threshold {best:.2f} (train Dice-optimal), {np.mean(freq >= best):.1%} of the image")
    return freq >= best


def main():
    ap = argparse.ArgumentParser(description="Polyp Dice and IoU on the public HyperKvasir split.")
    ap.add_argument("weights", nargs="?")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    assert args.baseline != bool(args.weights), "give weights or --baseline"

    report = json.loads((DATA / "build_report.json").read_text())
    cuts = report["size_tertile_cuts_area_fraction"]
    images = sorted((DATA / "images" / args.split).glob("*.jpg"))
    names = [p.stem for p in images]
    group_of = {n: i for i, g in enumerate(groups_from_pairs(sorted(report["split_of"]))) for n in g}

    if args.baseline:
        run = "centre-blob"
        blob = centre_blob(sorted(n for n, s in report["split_of"].items() if s == "train"))
        preds = []
        for n in names:
            h, w = gt_mask(n).shape
            preds.append(cv2.resize(blob.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0)
    else:
        from ultralytics import YOLO
        from evaluate_segmentation import patch_legacy_proto, predict, predicted_masks
        model = YOLO(args.weights)
        patch_legacy_proto(model)
        assert list(model.names.values()) == ["polyp"], model.names
        run = Path(args.weights).parent.name
        preds = []
        for n, result in zip(names, predict(model, [str(p) for p in images], args.device,
                                            imgsz=IMGSZ, conf=CONF)):
            h, w = result.orig_shape
            preds.append(predicted_masks(result, h, w, ["polyp"]).get("polyp", np.zeros((h, w), bool)))

    rows = []
    for n, p in zip(names, preds):
        g = gt_mask(n)
        assert g.shape == p.shape, (n, g.shape, p.shape)
        a = g.mean()
        rows.append({"image": n, "group": group_of[n], "size": SIZES[int(np.searchsorted(cuts, a))],
                     "polyp_area": round(float(a), 4), "found": int(p.any()), "dice": dice(g, p),
                     "iou": iou(g, p), "fp_area": round(float((p & ~g).mean()), 4)})

    out = ROOT / "results" / f"{run}_hyperkvasir_{args.split}.csv"
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    d = np.array([r["dice"] for r in rows])
    j = np.array([r["iou"] for r in rows])
    grp = np.array([r["group"] for r in rows])
    ug = np.unique(grp)
    idx_of = {g: np.where(grp == g)[0] for g in ug}
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(2000):
        idx = np.concatenate([idx_of[g] for g in rng.choice(ug, len(ug))])
        boot.append((d[idx].mean(), j[idx].mean()))
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)

    print(f"{run}  HyperKvasir {args.split}: {len(rows)} images, {len(ug)} near-duplicate groups\n")
    print(f"PRIMARY mean Dice {d.mean():.3f}  (95% CI over groups {lo[0]:.3f}-{hi[0]:.3f})")
    print(f"mean IoU          {j.mean():.3f}  (95% CI {lo[1]:.3f}-{hi[1]:.3f})")
    print(f"median Dice {np.median(d):.3f}; found {np.mean([r['found'] for r in rows]):.1%}; "
          f"Dice < 0.5: {int((d < 0.5).sum())}; mean false-positive area {np.mean([r['fp_area'] for r in rows]):.3f}")
    print(f"\n{'size':<8}{'n':>4}{'Dice':>8}{'IoU':>8}{'found':>8}")
    for s in SIZES:
        sel = [r for r in rows if r["size"] == s]
        print(f"{s:<8}{len(sel):>4}{np.mean([r['dice'] for r in sel]):>8.3f}"
              f"{np.mean([r['iou'] for r in sel]):>8.3f}{np.mean([r['found'] for r in sel]):>8.1%}")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
