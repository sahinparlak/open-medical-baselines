"""Evaluate a YOLO segmentation model on DSAD surgeries with per-class Dice.

Ground truth is the original DSAD merged masks (not our polygons), so conversion error does
not flatter the result. Two modes:

- multilabel (default): the fully labeled multilabel frames. For each image and class present
  in either prediction or ground truth, Dice is computed; images where the class is absent in
  both are skipped. Absent masks are true negatives, so false positives count.
- single organ (--single-organ): the single-organ folders. Only the folder's organ is
  segmented, so Dice, detection rate and false positives are scored for that organ only (a
  frame that sits in several folders is scored for each of its organs). Other classes have no
  masks and are checked against the DSAD weak labels instead: "weak+ found" = predicted where
  the weak label says the class is visible, "weak- pred" = predicted where it says the class
  is not visible. Weak labels miss some small appearances (docs/findings-dsad.md), so
  "weak- pred" is an upper bound for false alarms.

Results are summarised per class (median and mean Dice, detection rate) next to the human
inter-annotator Dice from results/dsad_label_audit.csv (same source: multilabel or single).

Only the eleven DSAD classes are scored. A model with more classes (v6: omentum 11, gallbladder
12) is read with its own class names; DSAD labels none of the extra classes, so they are ignored.

Run from the project root with the project venv:
    .venv/bin/python scripts/evaluate_segmentation.py models/<run>/best.pt [val|test] [--single-organ] [--device mps]
Writes results/<run>_<split>_dice.csv or results/<run>_<split>_single_dice.csv.
"""

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dsad_frames import collect_frames  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ML = ROOT / "data" / "dsad" / "multilabel"
CLASSES = ["abdominal_wall", "colon", "small_intestine", "stomach", "liver", "spleen",
           "pancreas", "ureter", "vesicular_glands", "inferior_mesenteric_artery",
           "intestinal_veins"]
IMGSZ = 1280
CHUNK = 16
NMS_RETRIES = 2
SINGLE_FIELDS = ["surgery", "image", "class", "role", "gt_present", "pred_present", "dice", "weak_visible", "pred_frac"]


class NMSTimeouts(logging.Handler):
    """Counts Ultralytics' "NMS time limit exceeded" warnings. On a time-out Ultralytics stops NMS
    and returns no predictions for the rest of the batch, which then score as missed organs. This
    happened when two evaluations ran at once (2026-09-28): whole chunks of single-organ frames
    came back empty, also in the first v1-m single-organ result."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.count = 0

    def emit(self, record):
        if "NMS time limit" in record.getMessage():
            self.count += 1


NMS_TIMEOUTS = NMSTimeouts()
_ultralytics_log = logging.getLogger("ultralytics")
_ultralytics_log.addHandler(NMS_TIMEOUTS)
if _ultralytics_log.getEffectiveLevel() > logging.WARNING:  # YOLO_VERBOSE=False would hide the warning
    _ultralytics_log.setLevel(logging.WARNING)


def human_dice(source="multilabel"):
    path = ROOT / "results" / "dsad_label_audit.csv"
    if not path.exists():
        return {}
    by = defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r.get("source", "multilabel") != source:
            continue
        for k in ("dice_12", "dice_13", "dice_23"):
            if r.get(k):
                by[r["organ"]].append(float(r[k]))
    return {k: float(np.median(v)) for k, v in by.items()}


def predicted_masks(result, h, w, names=CLASSES):
    """Union of the predicted masks per class at the original image size. Without retina_masks
    the masks come at the letterboxed input size; its padding (none for DSAD's 1280×1024 frames
    at imgsz 1280, 8–9 px top and bottom for CholecSeg8k's 854×480) is cropped before resizing,
    as in Ultralytics' scale_image."""
    out = {}
    if result.masks is None:
        return out
    data = result.masks.data.cpu().numpy()
    mh, mw = data.shape[1:]
    gain = min(mh / h, mw / w)
    padh, padw = (mh - h * gain) / 2, (mw - w * gain) / 2
    top, left, bottom, right = int(padh), int(padw), int(mh - padh), int(mw - padw)
    for mask, cls in zip(data, result.boxes.cls.cpu().numpy().astype(int)):
        m = cv2.resize(mask[top:bottom, left:right], (w, h), interpolation=cv2.INTER_NEAREST) > 0.5
        name = names[cls]
        out[name] = out[name] | m if name in out else m
    return out


def model_names(model):
    """Class names by id. An 11-class model keeps the DSAD names (as in evaluate_gastrectomy.py);
    a larger one must contain all eleven, or its DSAD classes would silently score as missed."""
    if len(model.names) == len(CLASSES):
        return list(CLASSES)
    names = [model.names[i] for i in sorted(model.names)]
    missing = [c for c in CLASSES if c not in names]
    if missing:
        sys.exit(f"model classes {names} lack the DSAD classes {missing}")
    return names


def patch_legacy_proto(model):
    """EVREN trains with an Ultralytics build whose Segment26 head uses the classic Proto on
    the first feature level only. Newer local versions pass all levels to the proto module.
    Feed the classic Proto just the first level, as the training build did."""
    head = model.model.model[-1]
    proto = getattr(head, "proto", None)
    if proto is None or type(proto).__name__ != "Proto":
        return
    cls = type(proto)
    if getattr(cls, "_legacy_patched", False):
        return
    # Patch the class, not the instance: predictors copy/fuse the module and move it to
    # the device, so an instance-bound wrapper would keep pointing at the old CPU copy.
    original = cls.forward
    cls.forward = lambda self, x: original(self, x[0] if isinstance(x, (list, tuple)) else x)
    cls._legacy_patched = True


def dice(gt, p):
    return round(float(2 * np.logical_and(gt, p).sum() / (gt.sum() + p.sum())), 4)


def predict(model, images, device, **kwargs):
    """Yield one result per image. Ultralytics loads a list of paths into memory and runs it
    as one batch, which overflows MPS for large lists, so predict in chunks. A chunk whose NMS
    timed out is predicted again; if it keeps timing out, stop rather than score empty frames.
    Not streamed, so that NMS has run on the whole chunk before the time-out check."""
    kwargs = {"imgsz": IMGSZ, "retina_masks": False, **kwargs}
    for start in range(0, len(images), CHUNK):
        chunk = [str(p) for p in images[start:start + CHUNK]]
        for _ in range(NMS_RETRIES + 1):
            before = NMS_TIMEOUTS.count
            results = model.predict(chunk, device=device, verbose=False, **kwargs)
            if NMS_TIMEOUTS.count == before:
                break
        else:
            sys.exit(f"NMS timed out {NMS_RETRIES + 1} times on the chunk starting at {chunk[0]}: "
                     "rerun with no other prediction job running")
        yield from results


def evaluate_multilabel(model, surgeries, device, names=CLASSES):
    rows = []
    for surgery in surgeries:
        images = sorted((ML / surgery).glob("image*.png"))
        for image, result in zip(images, predict(model, images, device)):
            idx = image.stem.replace("image", "")
            h, w = result.orig_shape
            pred = predicted_masks(result, h, w, names)
            for name in CLASSES:
                gt_path = ML / surgery / "merged" / f"mask{idx}_{name}.png"
                gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) > 127 if gt_path.exists() else np.zeros((h, w), bool)
                p = pred.get(name, np.zeros((h, w), bool))
                if not gt.any() and not p.any():
                    continue
                rows.append({"surgery": surgery, "image": idx, "class": name, "gt_present": int(gt.any()),
                             "pred_present": int(p.any()), "dice": dice(gt, p)})
    return rows


def evaluate_single_organ(model, surgeries, device, names=CLASSES):
    frames = sorted(collect_frames(set(surgeries), require_weak=False).values(), key=lambda f: f["name"])
    rows = []
    for f, result in zip(frames, predict(model, [f["image"] for f in frames], device)):
        h, w = result.orig_shape
        pred = predicted_masks(result, h, w, names)
        for name in CLASSES:
            p = pred.get(name, np.zeros((h, w), bool))
            row = {"surgery": f["surgery"], "image": f["name"], "class": name}
            if name in f["experts"]:
                gt = cv2.imread(str(f["experts"][name]), cv2.IMREAD_GRAYSCALE) > 127
                if not gt.any() and not p.any():
                    continue
                row.update(role="expert", gt_present=int(gt.any()), pred_present=int(p.any()), dice=dice(gt, p))
            else:
                row.update(role="other", gt_present="", pred_present=int(p.any()), dice="")
            row.update(weak_visible=int(name in f["visible"]) if f["weak_known"] else "", pred_frac=round(float(p.mean()), 4))
            rows.append(row)
    return rows


def summarize_multilabel(rows, human):
    print(f"{'class':<18}{'n gt':>6}{'Dice med':>10}{'Dice mean':>11}{'found':>8}{'false+':>8}{'human':>8}")
    for name in CLASSES:
        rs = [r for r in rows if r["class"] == name]
        gt = [r for r in rs if r["gt_present"]]
        if not gt:
            continue
        d = [r["dice"] for r in gt]
        found = sum(r["pred_present"] for r in gt) / len(gt)
        fp = sum(1 for r in rs if not r["gt_present"])
        print(f"{name:<18}{len(gt):>6}{np.median(d):>10.3f}{np.mean(d):>11.3f}{found:>8.0%}{fp:>8}"
              f"{human.get(name, float('nan')):>8.2f}")


def summarize_single_organ(rows, human):
    def rate(rs):
        return f"{sum(r['pred_present'] for r in rs) / len(rs):.0%} of {len(rs)}" if rs else "-"

    print(f"{'class':<28}{'n gt':>6}{'Dice med':>10}{'Dice mean':>11}{'found':>8}{'false+':>8}"
          f"{'weak+ found':>16}{'weak- pred':>16}{'human':>8}")
    for name in CLASSES:
        rs = [r for r in rows if r["class"] == name]
        gt = [r for r in rs if r["role"] == "expert" and r["gt_present"]]
        fp = sum(1 for r in rs if r["role"] == "expert" and not r["gt_present"])
        other = [r for r in rs if r["role"] == "other"]
        d = [r["dice"] for r in gt]
        dice_cols = (f"{len(gt):>6}{np.median(d):>10.3f}{np.mean(d):>11.3f}"
                     f"{sum(r['pred_present'] for r in gt) / len(gt):>8.0%}{fp:>8}") if gt else f"{0:>6}{'-':>10}{'-':>11}{'-':>8}{'-':>8}"
        print(f"{name:<28}{dice_cols}{rate([r for r in other if r['weak_visible'] == 1]):>16}"
              f"{rate([r for r in other if r['weak_visible'] == 0]):>16}{human.get(name, float('nan')):>8.2f}")


def main():
    ap = argparse.ArgumentParser(description="Per-class Dice of a YOLO segmentation model on DSAD surgeries.")
    ap.add_argument("weights")
    ap.add_argument("split", nargs="?", default="val")
    ap.add_argument("--single-organ", action="store_true", help="score the single-organ folders, not the multilabel frames")
    ap.add_argument("--device", default="cpu", help="cpu (recorded multilabel results) or mps (~7x faster)")
    args = ap.parse_args()

    splits = {r["surgery"]: r["split"] for r in csv.DictReader(open(ROOT / "splits" / "dsad_surgeries_v1.csv"))}
    model = YOLO(args.weights)
    patch_legacy_proto(model)
    names = model_names(model)
    run = Path(args.weights).parent.name
    if args.single_organ:
        surgeries = sorted(s for s in splits if splits[s] == args.split)
        rows = evaluate_single_organ(model, surgeries, args.device, names)
        out, fields = ROOT / "results" / f"{run}_{args.split}_single_dice.csv", SINGLE_FIELDS
    else:
        surgeries = sorted(s for s in splits if splits[s] == args.split and (ML / s).is_dir())
        rows = evaluate_multilabel(model, surgeries, args.device, names)
        out, fields = ROOT / "results" / f"{run}_{args.split}_dice.csv", list(rows[0])

    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)

    print(f"{args.weights}  split={args.split}  surgeries={surgeries}  device={args.device}\n")
    if args.single_organ:
        summarize_single_organ(rows, human_dice("single"))
    else:
        summarize_multilabel(rows, human_dice("multilabel"))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
