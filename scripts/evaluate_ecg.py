"""Evaluate an ECG classifier on the public ECG images split (plan: docs/plan-ecg-baseline.md).

Predicts the square 1024 images (data/ecg_images_square/v1/<split>/<class>/*.jpg) at imgsz 1024,
so YOLO's centre crop removes nothing. Reports per class sensitivity, specificity, PPV, NPV, F1 and
one-vs-rest AUROC, the confusion matrix, accuracy, macro F1 and macro AUROC, each with a 95% CI from
2,000 bootstrap resamples of report-ID groups (images with a pixel-identical printed ID field are
resampled together; IDs are hashed, never decoded). The majority class of train (abnormal_heartbeat)
is the trivial baseline.

Run from the project root with the project venv:
    .venv/bin/python -I scripts/evaluate_ecg.py models/<run>/best.pt --split val
Writes results/<run>_ecg_<split>_cases.csv and results/<run>_ecg_<split>_summary.json
"""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "ecg_images_square" / "v1"
RAW = ROOT / "data" / "ecg_images" / "raw"
CLASSES = ["normal", "abnormal_heartbeat", "MI", "history_of_MI"]
PREFIX = {"normal": "Normal", "abnormal_heartbeat": "HB", "MI": "MI", "history_of_MI": "PMI"}
MAJORITY = "abnormal_heartbeat"
ID_FIELD = (70, 113, 125, 272)  # as in scripts/ecg_images_public_v1.py


def report_group(cls, name):
    n = int(re.fullmatch(rf"{PREFIX[cls]}_(\d+)\.jpg", name).group(1))
    g = np.asarray(Image.open(RAW / cls / f"{PREFIX[cls]}({n}).jpg").convert("L"))
    f = g[ID_FIELD[0]:ID_FIELD[1], ID_FIELD[2]:ID_FIELD[3]] < 128
    return hashlib.sha1(f.tobytes()).hexdigest()[:12]


def metrics(y, p, prob):
    """y, p: class indices; prob: n x 4 probabilities."""
    out, f1s, aucs = {}, [], []
    for k, c in enumerate(CLASSES):
        t, q = y == k, p == k
        tp, fp, fn, tn = (t & q).sum(), (~t & q).sum(), (t & ~q).sum(), (~t & ~q).sum()
        sens = tp / (tp + fn) if tp + fn else np.nan
        spec = tn / (tn + fp) if tn + fp else np.nan
        ppv = tp / (tp + fp) if tp + fp else np.nan
        npv = tn / (tn + fn) if tn + fn else np.nan
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
        auc = roc_auc_score(t, prob[:, k]) if 0 < t.sum() < len(t) else np.nan
        out[c] = {"n": int(t.sum()), "sens": sens, "spec": spec, "ppv": ppv, "npv": npv, "f1": f1, "auroc": auc}
        f1s.append(f1)
        aucs.append(auc)
    out["accuracy"] = float((y == p).mean())
    out["macro_f1"] = float(np.nanmean(f1s))
    out["macro_auroc"] = float(np.nanmean(aucs))
    return out


def flatten(m):
    flat = {k: m[k] for k in ("accuracy", "macro_f1", "macro_auroc")}
    for c in CLASSES:
        for k, v in m[c].items():
            if k != "n":
                flat[f"{c}.{k}"] = v
    return flat


def main():
    ap = argparse.ArgumentParser(description="Per-class metrics on the public ECG split.")
    ap.add_argument("weights")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    names = model.names
    assert sorted(names.values()) == sorted(CLASSES), names
    col = [CLASSES.index(names[i]) for i in range(len(names))]  # model output -> our order

    files = sorted((DATA / args.split).glob("*/*.jpg"))
    rows, probs = [], []
    for i in range(0, len(files), 16):
        chunk = files[i:i + 16]
        for f, r in zip(chunk, model.predict([str(f) for f in chunk], imgsz=1024, device=args.device, verbose=False)):
            pr = np.zeros(len(CLASSES))
            pr[col] = r.probs.data.cpu().numpy()
            cls = f.parent.name
            rows.append({"file": f.name, "true": cls, "pred": CLASSES[int(pr.argmax())],
                         "group": report_group(cls, f.name), **{f"p_{c}": round(float(pr[k]), 5) for k, c in enumerate(CLASSES)}})
            probs.append(pr)
    prob = np.array(probs)
    y = np.array([CLASSES.index(r["true"]) for r in rows])
    p = prob.argmax(1)
    groups = np.array([r["group"] for r in rows])

    point = metrics(y, p, prob)
    base_p = np.full_like(y, CLASSES.index(MAJORITY))
    base_prob = np.zeros_like(prob)
    base_prob[:, CLASSES.index(MAJORITY)] = 1
    baseline = metrics(y, base_p, base_prob)

    rng = np.random.default_rng(0)
    ug = np.unique(groups)
    idx_of = {g: np.where(groups == g)[0] for g in ug}
    boots = []
    for _ in range(args.boot):
        idx = np.concatenate([idx_of[g] for g in rng.choice(ug, len(ug))])
        boots.append(flatten(metrics(y[idx], p[idx], prob[idx])))
    ci = {k: [float(np.nanpercentile([b[k] for b in boots], 2.5)), float(np.nanpercentile([b[k] for b in boots], 97.5))]
          for k in boots[0]}

    cm = np.zeros((4, 4), int)
    for a, b in zip(y, p):
        cm[a, b] += 1
    run = Path(args.weights).parent.name
    summary = {"run": run, "split": args.split, "images": len(rows), "groups": int(len(ug)),
               "true_counts": dict(Counter(r["true"] for r in rows)),
               "point": flatten(point), "ci95": ci, "majority_baseline": flatten(baseline),
               "confusion_rows_true_cols_pred": {"order": CLASSES, "matrix": cm.tolist()}}
    out = ROOT / "results" / f"{run}_ecg_{args.split}"
    with open(f"{out}_cases.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    Path(f"{out}_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")

    print(f"{run} {args.split}: {len(rows)} images, {len(ug)} report-ID groups")
    print(f"accuracy {point['accuracy']:.3f} {ci['accuracy']}  (majority {baseline['accuracy']:.3f})")
    print(f"macro F1 {point['macro_f1']:.3f} {ci['macro_f1']}  (majority {baseline['macro_f1']:.3f})")
    print(f"macro AUROC {point['macro_auroc']:.3f} {ci['macro_auroc']}")
    for c in CLASSES:
        m = point[c]
        print(f"  {c:20s} n={m['n']:3d} sens {m['sens']:.2f} spec {m['spec']:.2f} "
              f"F1 {m['f1']:.2f} AUROC {m['auroc']:.2f}")
    print("confusion (rows true, cols pred):", CLASSES)
    print(cm)


if __name__ == "__main__":
    main()
