"""Per-class Dice and cardiothoracic ratio of a FOCUS model (plan: docs/plan-focus-baseline.md).

Ground truth: the source masks (data/focus/raw/extracted/<split>/annfiles_mask/<n>-<class>.png,
threshold 127), not the polygons of the EVREN build. Both classes are in every image, so a miss
scores Dice 0. The predicted mask per class is the union of its instances at the original size
(confidence 0.25, fixed in the plan).

Per image and class: Dice, IoU, found. Primary: the mean of the two class means of Dice, with a
95% CI from 2,000 bootstrap resamples of images (no fetus ids: optimistic).
Cardiothoracic ratio from the masks: CTAR = cardiac area / thorax area, CTCR = ratio of the
circumferences of ellipses fitted to the largest contour of each mask; mean absolute difference
between prediction and ground truth (images where both classes were found).

--baseline: per class the training-mask frequency map (256 x 256) thresholded where the training
Dice is highest, resized to each image ("centre blob").
--human: the 4 source images annotated twice (pixel-identical copies), first vs second annotation.

Run from the project root with the project venv (one prediction job at a time):
    .venv/bin/python -I scripts/evaluate_focus.py models/<run>/best.pt [--split val]
    .venv/bin/python -I scripts/evaluate_focus.py --baseline [--split val]
    .venv/bin/python -I scripts/evaluate_focus.py --human
Writes results/<run>_focus_<split>.csv (run = "centre-blob" for the baseline)
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_segmentation import ROOT, dice  # noqa: E402

DATA = ROOT / "data" / "focus_public" / "v1"
SRC = ROOT / "data" / "focus" / "raw" / "extracted"
CLASSES = ["cardiac", "thorax"]
SRC_SPLIT = {"train": "training", "val": "validation", "test": "testing"}
IMGSZ = 640
CONF = 0.25


def gt_mask(key, cls):
    """key = '<source split>/<number>'."""
    s, n = key.split("/")
    return cv2.imread(str(SRC / s / "annfiles_mask" / f"{n}-{cls}.png"), cv2.IMREAD_GRAYSCALE) > 127


def iou(g, p):
    u = np.logical_or(g, p).sum()
    return round(float(np.logical_and(g, p).sum() / u), 4) if u else 1.0


def ellipse_circumference(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    if len(c) < 5:
        return float("nan")
    (_, _), (a, b), _ = cv2.fitEllipse(c)
    a, b = a / 2, b / 2  # semi-axes; Ramanujan's approximation
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


def ratios(card, thor):
    if not card.any() or not thor.any():
        return float("nan"), float("nan")
    return card.sum() / thor.sum(), ellipse_circumference(card) / ellipse_circumference(thor)


def score(key, pred):
    row = {"image": key}
    for c in CLASSES:
        g, p = gt_mask(key, c), pred[c]
        assert g.shape == p.shape, (key, c, g.shape, p.shape)
        row[f"{c}_found"] = int(p.any())
        row[f"{c}_dice"] = dice(g, p)
        row[f"{c}_iou"] = iou(g, p)
    gt_ctar, gt_ctcr = ratios(gt_mask(key, "cardiac"), gt_mask(key, "thorax"))
    pr_ctar, pr_ctcr = ratios(pred["cardiac"], pred["thorax"])
    row.update({"ctar_gt": round(gt_ctar, 4), "ctar_pred": round(pr_ctar, 4),
                "ctcr_gt": round(gt_ctcr, 4), "ctcr_pred": round(pr_ctcr, 4)})
    return row


def centre_blobs():
    report = json.loads((DATA / "build_report.json").read_text())
    dropped = {b for _, b in report["dropped_copies"]}
    keys = [f"training/{p.stem}" for p in sorted((SRC / "training" / "images").glob("*.png"))]
    keys = [k for k in keys if k not in dropped]
    blobs = {}
    for c in CLASSES:
        gts = [cv2.resize(gt_mask(k, c).astype(np.float32), (256, 256), interpolation=cv2.INTER_AREA) > 0.5
               for k in keys]
        freq = np.mean(gts, axis=0)
        best = max(np.arange(0.02, 0.98, 0.02), key=lambda t: np.mean([dice(g, freq >= t) for g in gts]))
        print(f"centre blob {c}: threshold {best:.2f}, {np.mean(freq >= best):.1%} of the image")
        blobs[c] = freq >= best
    return blobs


def human():
    report = json.loads((DATA / "build_report.json").read_text())
    print("first vs second annotation of the pixel-identical copies:")
    for a, b in report["dropped_copies"]:
        d = [dice(gt_mask(a, c), gt_mask(b, c)) for c in CLASSES]
        ra, rb = ratios(gt_mask(a, "cardiac"), gt_mask(a, "thorax")), ratios(gt_mask(b, "cardiac"), gt_mask(b, "thorax"))
        print(f"  {a} vs {b}: Dice cardiac {d[0]:.3f} thorax {d[1]:.3f}; "
              f"|dCTAR| {abs(ra[0] - rb[0]):.4f} |dCTCR| {abs(ra[1] - rb[1]):.4f}")


def main():
    ap = argparse.ArgumentParser(description="Per-class Dice and CTR on the public FOCUS split.")
    ap.add_argument("weights", nargs="?")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--human", action="store_true")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    if args.human:
        return human()
    assert args.baseline != bool(args.weights), "give weights or --baseline"

    images = sorted((DATA / "images" / args.split).glob("*.png"))
    keys = [f"{SRC_SPLIT[args.split]}/{p.stem.split('_')[1]}" for p in images]
    if args.baseline:
        run = "centre-blob"
        blobs = centre_blobs()
        preds = []
        for k in keys:
            h, w = gt_mask(k, "cardiac").shape
            preds.append({c: cv2.resize(blobs[c].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                          for c in CLASSES})
    else:
        from ultralytics import YOLO
        from evaluate_segmentation import patch_legacy_proto, predict, predicted_masks
        model = YOLO(args.weights)
        patch_legacy_proto(model)
        assert [model.names[i] for i in sorted(model.names)] == CLASSES, model.names
        run = Path(args.weights).parent.name
        preds = []
        for result in predict(model, [str(p) for p in images], args.device, imgsz=IMGSZ, conf=CONF):
            h, w = result.orig_shape
            pm = predicted_masks(result, h, w, CLASSES)
            preds.append({c: pm.get(c, np.zeros((h, w), bool)) for c in CLASSES})

    rows = [score(k, p) for k, p in zip(keys, preds)]
    out = ROOT / "results" / f"{run}_focus_{args.split}.csv"
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    d = np.array([[r[f"{c}_dice"] for c in CLASSES] for r in rows])
    rng = np.random.default_rng(0)
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"{run}  FOCUS {args.split}: {len(rows)} images\n")
    print(f"{'class':<9}{'Dice mean':>10}{'median':>8}{'IoU':>8}{'found':>8}{'Dice<0.8':>10}")
    for i, c in enumerate(CLASSES):
        print(f"{c:<9}{d[:, i].mean():>10.3f}{np.median(d[:, i]):>8.3f}"
              f"{np.mean([r[f'{c}_iou'] for r in rows]):>8.3f}{np.mean([r[f'{c}_found'] for r in rows]):>8.1%}"
              f"{int((d[:, i] < 0.8).sum()):>10}")
    print(f"\nPRIMARY mean of the 2 class means: {d.mean():.3f}  (95% bootstrap CI over images {lo:.3f}-{hi:.3f})")
    ok = [r for r in rows if not math.isnan(r["ctar_pred"])]
    for m in ("ctar", "ctcr"):
        err = np.array([abs(r[f"{m}_pred"] - r[f"{m}_gt"]) for r in ok])
        print(f"{m.upper()}: mean |pred - gt| {err.mean():.4f} (median {np.median(err):.4f}, n={len(ok)}); "
              f"gt mean {np.mean([r[f'{m}_gt'] for r in rows]):.3f}")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
