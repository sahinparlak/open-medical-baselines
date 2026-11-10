"""Per-class Dice of a 25-class model on the public gastrectomy dataset (plan:
docs/plan-gastrectomy-baseline.md).

Ground truth: the published polygons (data/gastrectomy_public/v1), rasterized at the frame size.
Per frame and class: Dice on frames whose ground truth has the class (a missed class scores 0),
and whether the class was predicted there ("found"). Predictions on frames without the class are
counted separately ("false+"). Per class mean Dice, then the primary mean over the 23 classes other
than LGV (no train frames) and PV (6 train frames), group means and per surgery primary means.

Run from the project root with the project venv (one prediction job at a time):
    .venv/bin/python scripts/evaluate_gastrectomy_public.py models/<run>/best.pt [--split val] [--device mps]
Writes results/<run>_gastrectomy_public_<split>.csv
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_segmentation import ROOT, dice, patch_legacy_proto, predict, predicted_masks  # noqa: E402
from evaluate_vessels import gt_masks  # noqa: E402

DATA = ROOT / "data" / "gastrectomy_public" / "v1"
FIELDS = ["surgery", "image", "class", "gt_present", "pred_present", "dice"]
EXCLUDED = ("LGV", "PV")
GROUPS = {
    "anatomy": ("colon", "gallbladder", "intestine", "ligament", "liver", "omentum", "pancreas", "spleen", "stomach"),
    "vessels/blood": ("VEIN", "artery", "blood"),
    "instruments": ("Needle", "Needle_holder", "US scalpel", "aspirator", "forceps", "gauze", "hemolock",
                    "hemolock-forceps", "hook", "stapler", "trocar"),
}


def main():
    ap = argparse.ArgumentParser(description="Per-class Dice on the public gastrectomy dataset.")
    ap.add_argument("weights")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    names = [n for _, n in sorted(yaml.safe_load((DATA / "data.yaml").read_text())["names"].items())]
    assert sorted(EXCLUDED + sum(GROUPS.values(), ())) == sorted(names), "groups do not cover the classes"
    model = YOLO(args.weights)
    patch_legacy_proto(model)
    assert [model.names[i] for i in sorted(model.names)] == names, "model classes differ from the dataset"
    run = Path(args.weights).parent.name
    images = sorted((DATA / "images" / args.split).glob("*.jpg"))

    rows = []
    for path, result in zip(images, predict(model, [str(p) for p in images], args.device)):
        h, w = result.orig_shape
        pred = predicted_masks(result, h, w, names)
        gt = gt_masks(DATA / "labels" / args.split / f"{path.stem}.txt", names, h, w)
        for cls in names:
            g = gt.get(cls, np.zeros((h, w), bool))
            p = pred.get(cls, np.zeros((h, w), bool))
            if g.any() or p.any():
                rows.append({"surgery": path.stem.split("_")[0], "image": path.stem, "class": cls,
                             "gt_present": int(g.any()), "pred_present": int(p.any()), "dice": dice(g, p)})

    out = ROOT / "results" / f"{run}_gastrectomy_public_{args.split}.csv"
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)

    surgeries = sorted({r["surgery"] for r in rows})
    print(f"{args.weights}  gastrectomy public {args.split}: {len(images)} frames, surgeries {surgeries}\n")
    print(f"{'class':<17}{'n gt':>6}{'Dice mean':>11}{'Dice med':>10}{'found':>8}{'false+':>8}   per surgery mean Dice")
    mean, per_surgery = {}, defaultdict(dict)
    for cls in names:
        rs = [r for r in rows if r["class"] == cls]
        gt = [r for r in rs if r["gt_present"]]
        fp = sum(1 for r in rs if not r["gt_present"])
        if not gt:
            print(f"{cls:<17}{0:>6}{'':>11}{'':>10}{'':>8}{fp:>8}")
            continue
        d = [r["dice"] for r in gt]
        mean[cls] = float(np.mean(d))
        for s in surgeries:
            ds = [r["dice"] for r in gt if r["surgery"] == s]
            if ds:
                per_surgery[s][cls] = float(np.mean(ds))
        note = "   (excluded from the primary mean)" if cls in EXCLUDED else ""
        print(f"{cls:<17}{len(gt):>6}{mean[cls]:>11.3f}{np.median(d):>10.3f}"
              f"{np.mean([r['pred_present'] for r in gt]):>8.0%}{fp:>8}   "
              + " ".join(f"{s}:{v:.2f}" for s, v in sorted((s, per_surgery[s][cls]) for s in surgeries if cls in per_surgery[s]))
              + note)

    primary = [c for c in names if c not in EXCLUDED and c in mean]
    print(f"\nPRIMARY mean over {len(primary)} classes (not {', '.join(EXCLUDED)}): "
          f"{np.mean([mean[c] for c in primary]):.3f}")
    for group, classes in GROUPS.items():
        cs = [c for c in classes if c in mean]
        print(f"  {group:<15}{np.mean([mean[c] for c in cs]):.3f}  ({len(cs)} classes)")
    print("Per surgery primary mean (classes present in that surgery):")
    for s in surgeries:
        cs = [c for c in primary if c in per_surgery[s]]
        print(f"  {s}  {np.mean([per_surgery[s][c] for c in cs]):.3f}  ({len(cs)} classes)")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
