"""External check of a VIP20K vein model on portal-vein (PV) frames of other centres
(plan: docs/plan-vip20k-baseline.md, step 6).

Sets (never trained on by the VIP20K model; other centres and cameras):
- gastrectomy: PV-labelled frames of data/gastrectomy_public/v1, train and val only (the test
  surgeries g03, g16, g17 are the Layer 1 held-out test and stay untouched);
- lapgc: PV-labelled frames (scene S7) of data/lapgc_yolo/public_v1, all splits.
Only PV is labelled there (other veins may be visible but unlabelled), so a predicted vein outside
the PV mask is not an error. Reported per frame: found (any predicted pixel on the PV), coverage
(share of the PV mask predicted as vein) and predicted vein area (share of the frame, to show
that coverage is not bought by painting everything). Means with 95% CIs from 2,000 bootstrap
resamples of surgeries (gastrectomy) or cases (lapgc).

Run from the project root with the project venv (one prediction job at a time):
    .venv/bin/python -I scripts/external_pv_vip20k.py models/<run>/best.pt
    .venv/bin/python -I scripts/external_pv_vip20k.py --baseline
Writes results/<run>_vip20k_external_pv.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_segmentation import ROOT  # noqa: E402
from evaluate_vip20k import CONF, IMGSZ, frequency_map  # noqa: E402

SETS = {
    "gastrectomy": (ROOT / "data" / "gastrectomy_public" / "v1", 3, ["train", "val"]),
    "lapgc": (ROOT / "data" / "lapgc_yolo" / "public_v1", 12, ["train", "val", "test"]),
}


def group_of(set_name, stem):
    return stem.split("_")[0] if set_name == "gastrectomy" else "_".join(stem.split("_")[:2])


def frames():
    out = []
    for name, (root, cls, splits) in SETS.items():
        names = (root / "data.yaml").read_text()
        assert ("PV" in names), name
        for split in splits:
            for lab in sorted((root / "labels" / split).glob("*.txt")):
                polys = [np.array(line.split()[1:], float).reshape(-1, 2)
                         for line in lab.read_text().splitlines() if line.split() and int(line.split()[0]) == cls]
                if polys:
                    img = root / "images" / split / f"{lab.stem}.jpg"
                    assert img.exists(), img
                    out.append((name, img, polys))
    return out


def pv_mask(polys, h, w):
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(p * [w, h]).astype(np.int32) for p in polys], 1)
    return m.astype(bool)


def boot_ci(values, groups, rng):
    gs = sorted(set(groups))
    idx_of = {g: np.where(groups == g)[0] for g in gs}
    b = [values[np.concatenate([idx_of[g] for g in rng.choice(gs, len(gs))])].mean() for _ in range(2000)]
    return np.percentile(b, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description="PV found/coverage of a VIP20K model on other centres.")
    ap.add_argument("weights", nargs="?")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    assert args.baseline != bool(args.weights), "give weights or --baseline"

    items = frames()
    if args.baseline:
        run = "frequency-map"
        train = sorted(p.stem for p in (ROOT / "data" / "vip20k_public" / "v1" / "images" / "train").glob("*.jpg"))
        fmap = frequency_map(train).astype(np.uint8)
        preds = (cv2.resize(fmap, (1920, 1080), interpolation=cv2.INTER_NEAREST) > 0 for _ in items)
    else:
        from ultralytics import YOLO
        from evaluate_segmentation import patch_legacy_proto, predict, predicted_masks
        model = YOLO(args.weights)
        patch_legacy_proto(model)
        assert list(model.names.values()) == ["vein"], model.names
        run = Path(args.weights).parent.name

        def gen():
            for result in predict(model, [str(i) for _, i, _ in items], args.device, imgsz=IMGSZ, conf=CONF):
                h, w = result.orig_shape
                yield predicted_masks(result, h, w, ["vein"]).get("vein", np.zeros((h, w), bool))
        preds = gen()

    rows = []
    for (name, img, polys), p in zip(items, preds):
        h, w = p.shape
        g = pv_mask(polys, h, w)
        assert g.any(), img
        inter = np.logical_and(g, p).sum()
        rows.append({"set": name, "image": img.stem, "split": img.parent.name, "group": group_of(name, img.stem),
                     "pv_area": round(float(g.mean()), 4), "found": int(inter > 0),
                     "coverage": round(float(inter / g.sum()), 4), "pred_area": round(float(p.mean()), 4)})

    out = ROOT / "results" / f"{run}_vip20k_external_pv.csv"
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    rng = np.random.default_rng(0)
    print(f"{run}  external PV check (found, coverage of the PV mask, predicted vein area)\n")
    for name in SETS:
        sel = [r for r in rows if r["set"] == name]
        groups = np.array([r["group"] for r in sel])
        found = np.array([r["found"] for r in sel], float)
        cov = np.array([r["coverage"] for r in sel])
        area = np.array([r["pred_area"] for r in sel])
        fl, fh_ = boot_ci(found, groups, rng)
        cl, ch = boot_ci(cov, groups, rng)
        print(f"{name}: {len(sel)} frames, {len(set(groups))} groups")
        print(f"  PV found {found.mean():.1%} (95% CI {fl:.1%}-{fh_:.1%}); mean coverage {cov.mean():.3f} "
              f"(95% CI {cl:.3f}-{ch:.3f}), median {np.median(cov):.3f}; mean predicted area {area.mean():.3f}")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
