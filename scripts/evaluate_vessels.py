"""Per-class Dice of a vessel model on the LapGC-KVAD-30 frames (plan: docs/results-vessels-v1.md).

Ground truth: the converted polygons (data/lapgc_yolo/v1, scripts/lapgc_to_yolo_seg.py; polygon
fidelity Dice ≥ 0.978 against the source masks), rasterized at the frame size. Scored as the anatomy
evaluations: per frame and class, rows where the class is in the ground truth or the prediction;
an absent ground truth with a prediction is a false positive. Labels are per scene (a frame is
labelled only for its scene's vessels), so a "false positive" can be a visible but unlabelled
vessel of another scene. A coverage table shows which class each labelled vessel is drawn as
(left/right pairs: LGA/RGA, LGeA_V/RGeA, LGV/RGV).

Run from the project root with the project venv (one prediction job at a time):
    .venv/bin/python scripts/evaluate_vessels.py models/<run>/best.pt [--split val] [--device mps]
Writes results/<run>_lapgc_<split>.csv
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

DATA = ROOT / "data" / "lapgc_yolo" / "v1"
FIELDS = ["case", "image", "class", "gt_present", "pred_present", "dice"]


def gt_masks(label_path, names, h, w):
    out = {}
    for line in label_path.read_text().splitlines():
        v = line.split()
        if not v:
            continue
        name = names[int(v[0])]
        m = out.setdefault(name, np.zeros((h, w), np.uint8))
        cv2.fillPoly(m, [np.round(np.array(v[1:], np.float32).reshape(-1, 2) * [w, h]).astype(np.int32)], 1)
    return {k: v.astype(bool) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description="Per-class Dice on the LapGC-KVAD-30 vessel frames.")
    ap.add_argument("weights")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    names = [l.split(":", 1)[1].strip() for l in (DATA / "data.yaml").read_text().splitlines()[1:] if ":" in l]
    model = YOLO(args.weights)
    patch_legacy_proto(model)
    assert [model.names[i] for i in sorted(model.names)] == names, "model classes differ from the dataset"
    run = Path(args.weights).parent.name
    images = sorted((DATA / "images" / args.split).glob("*.jpg"))

    rows, cover, gt_px = [], defaultdict(lambda: defaultdict(int)), defaultdict(int)
    for path, result in zip(images, predict(model, [str(p) for p in images], args.device)):
        h, w = result.orig_shape
        pred = predicted_masks(result, h, w, names)
        gt = gt_masks(DATA / "labels" / args.split / f"{path.stem}.txt", names, h, w)
        case = path.stem.split("_")[1]
        for cls in names:
            g = gt.get(cls, np.zeros((h, w), bool))
            p = pred.get(cls, np.zeros((h, w), bool))
            if g.any() or p.any():
                rows.append({"case": case, "image": path.stem, "class": cls, "gt_present": int(g.any()),
                             "pred_present": int(p.any()), "dice": dice(g, p)})
            if g.any():
                gt_px[cls] += int(g.sum())
                union = np.zeros((h, w), bool)
                for name, m in pred.items():
                    cover[cls][name] += int((g & m).sum())
                    union |= m
                cover[cls]["nothing"] += int((g & ~union).sum())

    out = ROOT / "results" / f"{run}_lapgc_{args.split}.csv"
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)

    cases = sorted({r["case"] for r in rows}, key=int)
    print(f"{args.weights}  LapGC {args.split}: {len(images)} frames, cases {cases}\n")
    print(f"{'class':<9}{'n gt':>6}{'Dice med':>10}{'Dice mean':>11}{'found':>8}{'false+':>8}   per case mean Dice")
    means = []
    for cls in names:
        rs = [r for r in rows if r["class"] == cls]
        gt = [r for r in rs if r["gt_present"]]
        if not gt:
            print(f"{cls:<9}{0:>6}{'':>10}{'':>11}{'':>8}{len(rs):>8}")
            continue
        d = [r["dice"] for r in gt]
        means.append(np.mean(d))
        per = defaultdict(list)
        for r in gt:
            per[r["case"]].append(r["dice"])
        print(f"{cls:<9}{len(gt):>6}{np.median(d):>10.3f}{np.mean(d):>11.3f}{np.mean([r['pred_present'] for r in gt]):>8.0%}"
              f"{sum(1 for r in rs if not r['gt_present']):>8}   " + " ".join(f"{c}:{np.mean(v):.2f}" for c, v in sorted(per.items(), key=lambda kv: int(kv[0]))))
    allgt = [r["dice"] for r in rows if r["gt_present"]]
    print(f"{'ALL':<9}{len(allgt):>6}{'':>10}{np.mean(allgt):>11.3f}   mean of class means {np.mean(means):.3f}")

    print("\nShare of each labelled vessel's pixels drawn as each class (rows: label; columns over 5% shown):")
    for cls in names:
        if not gt_px[cls]:
            continue
        shares = {c: cover[cls][c] / gt_px[cls] for c in names + ["nothing"]}
        print(f"  {cls:<8} " + ", ".join(f"{c} {s:.0%}" for c, s in sorted(shares.items(), key=lambda kv: -kv[1]) if s >= 0.05))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
