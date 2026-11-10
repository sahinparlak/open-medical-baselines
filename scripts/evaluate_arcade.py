"""Per-segment Dice of a model on the public ARCADE coronary segment images (plan: docs/plan-arcade-baseline.md),
or with --task stenosis on the stenosis images (docs/plan-arcade-stenosis-baseline.md).

Ground truth: data/arcade_public/<task> (scripts/arcade_public_v1.py), polygons rasterized at 512 x 512.
Not every segment is in every image, so two measures:
- per segment (primary): Dice on the images whose ground truth has the segment (a miss scores 0),
  mean per segment, then over the segments present in the split; "found" = share of those images
  where the model draws it; "FP images" = images where it is drawn but not in the ground truth.
- per image (challenge-style, our reading of the ARCADE metric): for each image, Dice over the
  segments in the ground truth or the prediction (a segment only in the prediction scores 0), mean
  per image, then over images, with a bootstrap 95% CI over images.
For stenosis (one class, at least one stenosis in every image) both measures are the per-image
stenosis Dice; added per lesion: share of ground-truth stenosis polygons the prediction touches,
and predicted regions (connected components) that touch no ground-truth stenosis.

Run from the project root with the project venv (one prediction job at a time):
    .venv/bin/python scripts/evaluate_arcade.py models/<run>/best.pt [--split val] [--task syntax] [--device mps]
Writes results/<run>_arcade_<split>.csv (syntax) or results/<run>_arcade-stenosis_<split>.csv
(one row per image and segment in ground truth or prediction).
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

DATA = ROOT / "data" / "arcade_public" / "syntax"  # set from --task in main
IMGSZ = 512  # the training size of the baseline runs
FIELDS = ["image", "segment", "gt_present", "pred_present", "dice"]


def class_names():
    return [l.split(":", 1)[1].strip().strip('"') for l in (DATA / "data.yaml").read_text().splitlines()[1:] if ":" in l]


def gt_masks(label_path, names, h, w):
    out = {}
    for line in label_path.read_text().splitlines():
        v = line.split()
        m = out.setdefault(names[int(v[0])], np.zeros((h, w), np.uint8))
        cv2.fillPoly(m, [np.round(np.array(v[1:], np.float32).reshape(-1, 2) * [w, h]).astype(np.int32)], 1)
    return {k: v.astype(bool) for k, v in out.items()}


def score(stem, gt, pred):
    """One row per segment in the ground truth or the prediction."""
    rows = []
    for name in sorted(set(gt) | set(pred)):
        g, p = gt.get(name), pred.get(name)
        rows.append({"image": stem, "segment": name, "gt_present": int(g is not None),
                     "pred_present": int(p is not None),
                     "dice": dice(g, p) if g is not None and p is not None else 0.0})
    return rows


def lesions(label_path, pred, h, w):
    """(ground-truth polygons, of them touched by the prediction, predicted regions, of them
    touching no ground-truth polygon)."""
    polys = []
    for line in label_path.read_text().splitlines():
        m = np.zeros((h, w), np.uint8)
        v = line.split()
        cv2.fillPoly(m, [np.round(np.array(v[1:], np.float32).reshape(-1, 2) * [w, h]).astype(np.int32)], 1)
        polys.append(m.astype(bool))
    gt_all = np.any(polys, axis=0) if polys else np.zeros((h, w), bool)
    hit = sum(1 for m in polys if (m & pred).any())
    n, comp = cv2.connectedComponents(pred.astype(np.uint8))
    fp = sum(1 for k in range(1, n) if not (gt_all & (comp == k)).any())
    return len(polys), hit, n - 1, fp


def per_image(rows):
    per = defaultdict(list)
    for r in rows:
        per[r["image"]].append(r["dice"])
    return {k: float(np.mean(v)) for k, v in per.items()}


def bootstrap_ci(values, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(values)
    means = np.sort(rng.choice(v, (n, len(v))).mean(axis=1))
    return float(means[int(0.025 * n)]), float(means[int(0.975 * n) - 1])


def summarize(rows, names):
    """Print both tables; return (mean of segment means, per-image mean, its CI)."""
    print(f"{'segment':<32}{'images':>7}{'Dice mean':>11}{'found':>7}{'FP images':>11}")
    means = []
    for name in names:
        with_gt = [r for r in rows if r["segment"] == name and r["gt_present"]]
        fp = sum(1 for r in rows if r["segment"] == name and r["pred_present"] and not r["gt_present"])
        if not with_gt:
            print(f"{name:<32}{0:>7}{'—':>11}{'—':>7}{fp:>11}")
            continue
        means.append(np.mean([r["dice"] for r in with_gt]))
        found = np.mean([r["pred_present"] for r in with_gt])
        print(f"{name:<32}{len(with_gt):>7}{means[-1]:>11.3f}{found:>7.0%}{fp:>11}")
    img = per_image(rows)
    lo, hi = bootstrap_ci(list(img.values()))
    print(f"{f'mean of segment means ({len(means)} segments)':<39}{np.mean(means):>11.3f}")
    print(f"per-image Dice (challenge-style), {len(img)} images: {np.mean(list(img.values())):.3f} "
          f"(95% CI {lo:.3f}–{hi:.3f})")
    return float(np.mean(means)), float(np.mean(list(img.values()))), (lo, hi)


def main():
    ap = argparse.ArgumentParser(description="Per-segment and per-image Dice on the public ARCADE syntax images.")
    ap.add_argument("weights")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--task", default="syntax", choices=["syntax", "stenosis"])
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    global DATA
    DATA = ROOT / "data" / "arcade_public" / args.task

    names = class_names()
    model = YOLO(args.weights)
    patch_legacy_proto(model)
    assert [model.names[i] for i in sorted(model.names)] == names, "model classes differ from the dataset"
    run = Path(args.weights).parent.name
    images = sorted((DATA / "images" / args.split).glob("*.png"))

    rows, les = [], np.zeros(4, int)
    for path, result in zip(images, predict(model, images, args.device, imgsz=IMGSZ)):
        h, w = result.orig_shape
        label = DATA / "labels" / args.split / f"{path.stem}.txt"
        gt = gt_masks(label, names, h, w)
        pred = predicted_masks(result, h, w, names)
        rows += score(path.stem, gt, pred)
        if args.task == "stenosis":
            les += lesions(label, pred.get("stenosis", np.zeros((h, w), bool)), h, w)

    suffix = "arcade" if args.task == "syntax" else "arcade-stenosis"
    out = ROOT / "results" / f"{run}_{suffix}_{args.split}.csv"
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    print(f"{args.weights}  ARCADE {args.task} {args.split}: {len(images)} images\n")
    summarize(rows, names)
    if args.task == "stenosis":
        print(f"lesions: {les[1]} of {les[0]} ground-truth stenoses touched ({les[1] / les[0]:.0%}); "
              f"{les[3]} of {les[2]} predicted regions touch no ground-truth stenosis")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
