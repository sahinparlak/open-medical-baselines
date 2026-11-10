"""Box metrics of a detection model on the public MODUS 40x images (plan: docs/plan-modus-baseline.md),
or with --dataset visem on the public VISEM-Tracking frames (docs/plan-visem-baseline.md; 640 px).

Ultralytics' own detection validation on the local copy (data/modus_public/40x,
scripts/modus_public_v1.py), at the training size 1280 with its defaults (conf 0.001, IoU 0.7):
per class AP50, AP50-95, precision, recall and the number of boxes, and the means over the
classes present in the split (mAP50, mAP50-95). Run twice: with max_det 1,500, because some
images hold up to ~1,000 cells and the default cap of 300 detections would cut recall there, and
with the default 300 for comparison with EVREN's numbers. Classes with few boxes (fatty cast, oval
fat body, air bubble) give unstable APs.

Run from the project root with the project venv (one prediction job at a time):
    .venv/bin/python scripts/evaluate_modus.py models/<run>/best.pt [--split val] [--dataset modus40x] [--device mps]
Writes results/<run>_<dataset>_<split>.csv (one row per class and max_det).
"""

import argparse
import csv
import sys
import tempfile
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_segmentation import ROOT  # noqa: E402

DATASETS = {"modus40x": (ROOT / "data" / "modus_public" / "40x", 1280),  # data, training imgsz
            "visem": (ROOT / "data" / "visem_public", 640)}
DATA, IMGSZ = DATASETS["modus40x"]  # set from --dataset in main
MAX_DETS = (1500, 300)
FIELDS = ["max_det", "class", "images", "boxes", "precision", "recall", "ap50", "ap50_95"]


def class_names():
    return [l.split(":", 1)[1].strip().strip('"') for l in (DATA / "data.yaml").read_text().splitlines()[1:] if ":" in l]


def data_yaml(names, tmp):
    """data.yaml with absolute split paths for Ultralytics' validator."""
    p = Path(tmp) / "data.yaml"
    p.write_text(f"path: {DATA}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
                 + "".join(f"  {i}: \"{n}\"\n" for i, n in enumerate(names)))
    return p


def main():
    ap = argparse.ArgumentParser(description="Per-class box AP on the public MODUS 40x images.")
    ap.add_argument("weights")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--dataset", default="modus40x", choices=list(DATASETS))
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    global DATA, IMGSZ
    DATA, IMGSZ = DATASETS[args.dataset]

    names = class_names()
    model = YOLO(args.weights)
    assert [model.names[i] for i in sorted(model.names)] == names, "model classes differ from the dataset"
    run = Path(args.weights).parent.name
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        yaml = data_yaml(names, tmp)
        for max_det in MAX_DETS:
            m = model.val(data=str(yaml), split=args.split, imgsz=IMGSZ, max_det=max_det, device=args.device,
                          batch=4, plots=False, verbose=False, project=tmp, name=f"md{max_det}")
            box = m.box
            for k, c in enumerate(box.ap_class_index):
                p, r, ap50, ap = box.class_result(k)
                rows.append({"max_det": max_det, "class": names[c], "images": int(m.nt_per_image[c]),
                             "boxes": int(m.nt_per_class[c]), "precision": round(p, 4), "recall": round(r, 4),
                             "ap50": round(ap50, 4), "ap50_95": round(ap, 4)})
            print(f"max_det {max_det}: mAP50 {box.map50:.3f}  mAP50-95 {box.map:.3f}  "
                  f"({len(box.ap_class_index)} classes present)")

    out = ROOT / "results" / f"{run}_{args.dataset}_{args.split}.csv"
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)

    print(f"\n{args.weights}  {args.dataset} {args.split}, max_det {MAX_DETS[0]}:")
    print(f"{'class':<38}{'images':>7}{'boxes':>7}{'P':>7}{'R':>7}{'AP50':>7}{'AP50-95':>9}")
    for r in sorted((r for r in rows if r["max_det"] == MAX_DETS[0]), key=lambda r: -r["boxes"]):
        print(f"{r['class']:<38}{r['images']:>7}{r['boxes']:>7}{r['precision']:>7.3f}{r['recall']:>7.3f}"
              f"{r['ap50']:>7.3f}{r['ap50_95']:>9.3f}")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
