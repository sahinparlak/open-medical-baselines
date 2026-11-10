"""Per-structure Dice of a model on the public PFUS1 frames (plan: docs/plan-pfus1-baseline.md).

Ground truth: data/pfus1_public (scripts/pfus1_public_v1.py), polygons rasterized at the frame size.
Every frame has all 8 structures, so each frame scores each structure; a structure the model does
not draw scores 0. Reported per structure (mean over frames, share of frames where it is drawn),
the mean of the 8 structure means, and the same per video (frame means per video, then the mean of
videos), because the frames of one video are near-identical. The ground truth between the
sonographer's key frames is interpolated (paper), so part of any error is the interpolation's.

Run from the project root with the project venv (one prediction job at a time):
    .venv/bin/python scripts/evaluate_pfus1.py models/<run>/best.pt [--split val] [--device mps]
Writes results/<run>_pfus1_<split>.csv
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_segmentation import ROOT, dice, patch_legacy_proto, predict, predicted_masks  # noqa: E402

DATA = ROOT / "data" / "pfus1_public"
IMGSZ = 1024  # the training size of the baseline runs
FIELDS = ["video", "image", "structure", "pred_present", "dice"]


def class_names():
    return [l.split(":", 1)[1].strip().strip('"') for l in (DATA / "data.yaml").read_text().splitlines()[1:] if ":" in l]


def gt_masks(label_path, names, h, w):
    out = {}
    for line in label_path.read_text().splitlines():
        v = line.split()
        m = out.setdefault(names[int(v[0])], np.zeros((h, w), np.uint8))
        cv2.fillPoly(m, [np.round(np.array(v[1:], np.float32).reshape(-1, 2) * [w, h]).astype(np.int32)], 1)
    return {k: v.astype(bool) for k, v in out.items()}


def score(stem, gt, pred, names):
    """One row per structure; the ground truth always has all of them."""
    h, w = next(iter(gt.values())).shape
    rows = []
    for name in names:
        p = pred.get(name, np.zeros((h, w), bool))
        rows.append({"video": stem.split("_")[0], "image": stem, "structure": name,
                     "pred_present": int(p.any()), "dice": dice(gt[name], p)})
    return rows


def summarize(rows, names):
    """Print the per-structure table; return (mean of structure means, mean over videos)."""
    print(f"{'structure':<20}{'frames':>7}{'Dice mean':>11}{'median':>8}{'found':>7}{'video mean':>12}{'worst video':>14}")
    means, vmeans = [], []
    for name in names:
        rs = [r for r in rows if r["structure"] == name]
        d = [r["dice"] for r in rs]
        per = defaultdict(list)
        for r in rs:
            per[r["video"]].append(r["dice"])
        pv = {v: np.mean(x) for v, x in per.items()}
        worst = min(pv, key=pv.get)
        means.append(np.mean(d))
        vmeans.append(np.mean(list(pv.values())))
        print(f"{name:<20}{len(rs):>7}{np.mean(d):>11.3f}{np.median(d):>8.3f}{np.mean([r['pred_present'] for r in rs]):>7.0%}"
              f"{vmeans[-1]:>12.3f}{f'{worst} {pv[worst]:.2f}':>14}")
    print(f"{'mean of structure means':<27}{np.mean(means):>11.3f}{'':>15}{np.mean(vmeans):>12.3f}")
    return float(np.mean(means)), float(np.mean(vmeans))


def main():
    ap = argparse.ArgumentParser(description="Per-structure Dice on the public PFUS1 frames.")
    ap.add_argument("weights")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    names = class_names()
    model = YOLO(args.weights)
    patch_legacy_proto(model)
    assert [model.names[i] for i in sorted(model.names)] == names, "model classes differ from the dataset"
    run = Path(args.weights).parent.name
    images = sorted((DATA / "images" / args.split).glob("*.png"))

    rows = []
    for path, result in zip(images, predict(model, images, args.device, imgsz=IMGSZ)):
        h, w = result.orig_shape
        gt = gt_masks(DATA / "labels" / args.split / f"{path.stem}.txt", names, h, w)
        rows += score(path.stem, gt, predicted_masks(result, h, w, names), names)

    out = ROOT / "results" / f"{run}_pfus1_{args.split}.csv"
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    videos = sorted({r["video"] for r in rows})
    print(f"{args.weights}  PFUS1 {args.split}: {len(images)} frames, {len(videos)} videos\n")
    summarize(rows, names)
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
