"""Head Dice and head-circumference error of an HC18 model (plan: docs/plan-hc18-baseline.md).

Ground truth: the source outline filled by a fitted ellipse, exactly as in scripts/hc18_public_v1.py,
and the source head circumference (mm) from training_set_pixel_size_and_HC.csv. The predicted mask
is the union of head instances at the original size (confidence 0.25, fixed in the plan).

Per image: Dice, IoU, found, and HC_pred = circumference of the ellipse fitted to the largest
contour of the predicted mask (Ramanujan) x pixel size; error = HC_pred - HC_source (mm).
Reported: mean Dice with a 95% CI from 2,000 bootstrap resamples of patients, mean absolute and
mean signed HC error, share within 3 mm.

--baseline: the centre blob (training-mask frequency map at 256 x 256, threshold with the highest
training Dice) for Dice, and the training mean HC as a constant prediction for the HC error.

Run from the project root with the project venv (one prediction job at a time):
    .venv/bin/python -I scripts/evaluate_hc18.py models/<run>/best.pt [--split val]
    .venv/bin/python -I scripts/evaluate_hc18.py --baseline [--split val]
Writes results/<run>_hc18_<split>.csv (run = "centre-blob" for the baseline)
"""

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_segmentation import ROOT, dice  # noqa: E402
from hc18_public_v1 import filled_mask  # noqa: E402

DATA = ROOT / "data" / "hc18_public" / "v1"
SRC = ROOT / "data" / "hc18" / "raw" / "extracted" / "training_set"
CSV = ROOT / "data" / "hc18" / "raw" / "training_set_pixel_size_and_HC.csv"
IMGSZ = 640
CONF = 0.25


def gt_mask(name):
    return filled_mask(cv2.imread(str(SRC / f"{name}_Annotation.png"), cv2.IMREAD_GRAYSCALE) > 127)


def iou(g, p):
    u = np.logical_or(g, p).sum()
    return round(float(np.logical_and(g, p).sum() / u), 4) if u else 1.0


def circumference_px(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return float("nan")
    c = max(cs, key=cv2.contourArea)
    if len(c) < 5:
        return float("nan")
    (_, _), (a, b), _ = cv2.fitEllipse(c)
    a, b = a / 2, b / 2
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


def main():
    ap = argparse.ArgumentParser(description="Head Dice and HC error on the public HC18 split.")
    ap.add_argument("weights", nargs="?")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    assert args.baseline != bool(args.weights), "give weights or --baseline"

    table = {r["filename"][:-4]: (float(r["pixel size(mm)"]), float(r["head circumference (mm)"]))
             for r in csv.DictReader(open(CSV))}
    report = json.loads((DATA / "build_report.json").read_text())
    images = sorted((DATA / "images" / args.split).glob("*.png"))
    names = [p.stem for p in images]

    if args.baseline:
        run = "centre-blob"
        train = sorted(p.stem for p in (DATA / "images" / "train").glob("*.png"))
        gts = [cv2.resize(gt_mask(n).astype(np.float32), (256, 256), interpolation=cv2.INTER_AREA) > 0.5 for n in train]
        freq = np.mean(gts, axis=0)
        best = max(np.arange(0.02, 0.98, 0.02), key=lambda t: np.mean([dice(g, freq >= t) for g in gts]))
        blob = freq >= best
        mean_hc = float(np.mean([table[n][1] for n in train]))
        print(f"centre blob: threshold {best:.2f}, {blob.mean():.1%} of the image; training mean HC {mean_hc:.1f} mm")
        preds = []
        for n in names:
            h, w = gt_mask(n).shape
            preds.append(cv2.resize(blob.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0)
    else:
        from ultralytics import YOLO
        from evaluate_segmentation import patch_legacy_proto, predict, predicted_masks
        model = YOLO(args.weights)
        patch_legacy_proto(model)
        assert list(model.names.values()) == ["head"], model.names
        run = Path(args.weights).parent.name
        preds = []
        for result in predict(model, [str(p) for p in images], args.device, imgsz=IMGSZ, conf=CONF):
            h, w = result.orig_shape
            preds.append(predicted_masks(result, h, w, ["head"]).get("head", np.zeros((h, w), bool)))

    rows = []
    for n, p in zip(names, preds):
        g = gt_mask(n)
        assert g.shape == p.shape, (n, g.shape, p.shape)
        px, hc = table[n]
        hc_pred = mean_hc if args.baseline else circumference_px(p) * px
        rows.append({"image": n, "patient": re.match(r"(\d+)_", n).group(1), "found": int(p.any()),
                     "dice": dice(g, p), "iou": iou(g, p), "hc_mm": hc, "hc_pred_mm": round(hc_pred, 2),
                     "hc_err_mm": round(hc_pred - hc, 2)})

    out = ROOT / "results" / f"{run}_hc18_{args.split}.csv"
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    d = np.array([r["dice"] for r in rows])
    pat = np.array([r["patient"] for r in rows])
    up = np.unique(pat)
    idx_of = {u: np.where(pat == u)[0] for u in up}
    rng = np.random.default_rng(0)
    boot = [d[np.concatenate([idx_of[u] for u in rng.choice(up, len(up))])].mean() for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    e = np.array([r["hc_err_mm"] for r in rows if not math.isnan(r["hc_err_mm"])])
    print(f"{run}  HC18 {args.split}: {len(rows)} images, {len(up)} patients\n")
    print(f"PRIMARY mean Dice {d.mean():.3f}  (95% CI over patients {lo:.3f}-{hi:.3f}); median {np.median(d):.3f}; "
          f"IoU {np.mean([r['iou'] for r in rows]):.3f}; found {np.mean([r['found'] for r in rows]):.1%}; Dice < 0.9: {int((d < 0.9).sum())}")
    print(f"HC error (mm, n={len(e)}): mean |err| {np.abs(e).mean():.2f}, median |err| {np.median(np.abs(e)):.2f}, "
          f"bias {e.mean():+.2f}, within 3 mm {np.mean(np.abs(e) <= 3):.1%}; source HC mean {np.mean([r['hc_mm'] for r in rows]):.1f}")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
