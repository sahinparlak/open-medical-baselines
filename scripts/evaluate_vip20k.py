"""Vein Dice and IoU of a VIP20K model (plan: docs/plan-vip20k-baseline.md).

Ground truth: the source binary masks (data/vip20k/extracted/.../Mask/<frame>_mask.png, value 1 =
vein), not the polygons of the EVREN build. The predicted mask is the union of all vein instances
at the original size (confidence 0.25, fixed in the plan).

Per frame: Dice, IoU, found (any predicted pixel overlaps the vein), false-positive area (predicted
pixels outside the vein, as a fraction of the frame). The 3 vein-free frames are not scored by Dice;
their predicted area is listed. Reported: mean Dice (primary) and mean IoU with 95% CIs from 2,000
bootstrap resamples of operations (only 4 per split, so the CIs are crude and wide), per operation
and per procedure type (LDP, LPD), the share with Dice < 0.5.

Trivial baseline (--baseline): the vein frequency map of the training masks (at 192 x 108)
thresholded where the mean training Dice is highest, resized to each frame and scored the same way.

Run from the project root with the project venv (one prediction job at a time):
    .venv/bin/python -I scripts/evaluate_vip20k.py models/<run>/best.pt [--split val]
    .venv/bin/python -I scripts/evaluate_vip20k.py --baseline [--split val]
Writes results/<run>_vip20k_<split>.csv (run = "frequency-map" for the baseline)
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_segmentation import ROOT, dice  # noqa: E402

DATA = ROOT / "data" / "vip20k_public" / "v1"
SRC = ROOT / "data" / "vip20k" / "extracted" / "VIP20K" / "VIP20K"
IMGSZ = 1280
CONF = 0.25
SMALL = (192, 108)


def mask_path(name):
    """LDP-06_s01_cvt-3850 -> LDP-PUMCH/Case 06/segment_01/Mask/cvt-3850_mask.png"""
    case, seg, stem = name.split("_", 2)
    proc, num = case.split("-")
    return SRC / f"{proc}-PUMCH" / f"Case {num}" / f"segment_{seg[1:]}" / "Mask" / f"{stem}_mask.png"


def gt_mask(name):
    m = cv2.imread(str(mask_path(name)), cv2.IMREAD_UNCHANGED)
    assert m is not None, name
    return m == 1


def iou(g, p):
    union = np.logical_or(g, p).sum()
    return round(float(np.logical_and(g, p).sum() / union), 4) if union else 1.0


def frequency_map(train_names):
    gts = [cv2.resize(gt_mask(n).astype(np.float32), SMALL, interpolation=cv2.INTER_AREA) > 0.5
           for n in train_names]
    gts = [g for g in gts if g.any()]
    freq = np.mean(gts, axis=0)
    best = max(np.arange(0.02, 0.52, 0.02),
               key=lambda t: np.mean([dice(g, freq >= t) for g in gts]))
    print(f"frequency map: threshold {best:.2f} (train Dice-optimal), {np.mean(freq >= best):.1%} of the frame")
    return freq >= best


def main():
    ap = argparse.ArgumentParser(description="Vein Dice and IoU on the public VIP20K split.")
    ap.add_argument("weights", nargs="?")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    assert args.baseline != bool(args.weights), "give weights or --baseline"

    images = sorted((DATA / "images" / args.split).glob("*.jpg"))
    names = [p.stem for p in images]

    if args.baseline:
        run = "frequency-map"
        train = sorted(p.stem for p in (DATA / "images" / "train").glob("*.jpg"))
        fmap = frequency_map(train).astype(np.uint8)
        preds = (cv2.resize(fmap, (1920, 1080), interpolation=cv2.INTER_NEAREST) > 0 for _ in names)
    else:
        from ultralytics import YOLO
        from evaluate_segmentation import patch_legacy_proto, predict, predicted_masks
        model = YOLO(args.weights)
        patch_legacy_proto(model)
        assert list(model.names.values()) == ["vein"], model.names
        run = Path(args.weights).parent.name

        def gen():
            for result in predict(model, [str(p) for p in images], args.device, imgsz=IMGSZ, conf=CONF):
                h, w = result.orig_shape
                yield predicted_masks(result, h, w, ["vein"]).get("vein", np.zeros((h, w), bool))
        preds = gen()

    rows, empty = [], []
    for n, p in zip(names, preds):
        g = gt_mask(n)
        assert g.shape == p.shape, (n, g.shape, p.shape)
        case = n.split("_")[0]
        if not g.any():
            empty.append((n, round(float(p.mean()), 4)))
            continue
        rows.append({"image": n, "operation": case, "procedure": case[:3],
                     "vein_area": round(float(g.mean()), 4),
                     "found": int(np.logical_and(g, p).any()), "dice": dice(g, p), "iou": iou(g, p),
                     "fp_area": round(float((p & ~g).mean()), 4)})

    out = ROOT / "results" / f"{run}_vip20k_{args.split}.csv"
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    d = np.array([r["dice"] for r in rows])
    j = np.array([r["iou"] for r in rows])
    op = np.array([r["operation"] for r in rows])
    ops = sorted(set(op))
    idx_of = {o: np.where(op == o)[0] for o in ops}
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(2000):
        idx = np.concatenate([idx_of[o] for o in rng.choice(ops, len(ops))])
        boot.append((d[idx].mean(), j[idx].mean()))
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)

    print(f"{run}  VIP20K {args.split}: {len(rows)} frames with a vein, {len(ops)} operations\n")
    print(f"PRIMARY mean Dice {d.mean():.3f}  (95% CI over operations {lo[0]:.3f}-{hi[0]:.3f})")
    print(f"mean IoU          {j.mean():.3f}  (95% CI {lo[1]:.3f}-{hi[1]:.3f})")
    print(f"median Dice {np.median(d):.3f}; found {np.mean([r['found'] for r in rows]):.1%}; "
          f"Dice < 0.5: {(d < 0.5).mean():.1%}; mean false-positive area "
          f"{np.mean([r['fp_area'] for r in rows]):.3f}")
    by = defaultdict(list)
    for r in rows:
        by[r["operation"]].append(r)
        by[r["procedure"]].append(r)
    print(f"\n{'group':<10}{'frames':>7}{'Dice':>8}{'IoU':>8}{'found':>8}")
    for k in ops + ["LDP", "LPD"]:
        sel = by.get(k, [])
        if sel:
            print(f"{k:<10}{len(sel):>7}{np.mean([r['dice'] for r in sel]):>8.3f}"
                  f"{np.mean([r['iou'] for r in sel]):>8.3f}{np.mean([r['found'] for r in sel]):>8.1%}")
    if empty:
        print(f"\nvein-free frames (predicted area): {empty}")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
