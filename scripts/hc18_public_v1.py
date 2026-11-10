"""Build a public EVREN version of HC18 (fetal head circumference ultrasound; Zenodo 1327317,
CC BY 4.0; docs/datasets.md, HC18 entry).

999 grey PNG training images, each with an annotation PNG that holds the outline of the fetal head
ellipse. The outline is turned into a filled mask by fitting an ellipse to its pixels and drawing it
filled (plus the outline pixels): on the 995 closed outlines this equals filling the contour
(Dice 1.000); 4 heads touch the image edge, where the outline is not closed inside the image.
The mask becomes a YOLO polygon (class 0 head), checked by drawing it again.

Patients come from the file names: NNN_HC is patient NNN, NNN_2HC ... NNN_4HC are repeats (806
patients). 392_2HC is a pixel-identical copy of 392_HC with the same annotation and is dropped
(998 images). Patients are assigned 70/15/15 by a seeded search over 2,000 random greedy orders,
keeping the assignment closest to the target image counts in each head-circumference tertile
(the HC of the patient's first image, from the source CSV; a proxy for gestational age).
Images are copied unchanged under their source names.

The build stops if a check fails: 999 source images, 806 patients, one dropped copy, no patient in
two splits, every image with a polygon, median polygon Dice >= 0.97.

Run from the project root with the project venv:
    .venv/bin/python -I scripts/hc18_public_v1.py
Writes data/hc18_public/v1/ (images, labels, data.yaml, build_report.json) and
data/upload/hc18_public_v1_yoloseg.zip
"""

import csv
import hashlib
import json
import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from dsad_to_yolo_seg import mask_to_polygons  # noqa: E402

SRC = ROOT / "data" / "hc18" / "raw" / "extracted" / "training_set"
CSV = ROOT / "data" / "hc18" / "raw" / "training_set_pixel_size_and_HC.csv"
OUT = ROOT / "data" / "hc18_public" / "v1"
ZIP = ROOT / "data" / "upload" / "hc18_public_v1_yoloseg.zip"
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
TRIES = 2000


def filled_mask(outline):
    ys, xs = np.nonzero(outline)
    ellipse = cv2.fitEllipse(np.stack([xs, ys], 1).astype(np.float32))
    m = np.zeros(outline.shape, np.uint8)
    cv2.ellipse(m, ellipse, 1, -1)
    return m.astype(bool) | outline


def assign(patients, stratum, n_images):
    """Seeded greedy search over patients; each goes to the split furthest below its target in the
    patient's HC tertile (counted in images)."""
    target = np.array([[sum(n_images[p] for p in patients if stratum[p] == s) * f for s in range(3)]
                       for f in SPLITS.values()])
    rng = np.random.default_rng(0)
    best, best_cost = None, np.inf
    for _ in range(TRIES):
        have = np.zeros_like(target)
        out = {}
        for i in rng.permutation(len(patients)):
            p = patients[i]
            k = int(np.argmax((target[:, stratum[p]] - have[:, stratum[p]]) / target[:, stratum[p]]))
            have[k, stratum[p]] += n_images[p]
            out[p] = list(SPLITS)[k]
        cost = ((have - target) ** 2).sum()
        if cost < best_cost:
            best, best_cost = out, cost
    return best


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; remove it first so no stale files enter the ZIP")
    hc = {r["filename"][:-4]: float(r["head circumference (mm)"]) for r in csv.DictReader(open(CSV))}
    names = sorted(p.stem for p in SRC.glob("*.png") if not p.stem.endswith("_Annotation"))
    assert len(names) == 999 and set(names) == set(hc), len(names)

    hashes, dropped = {}, []
    for n in names:
        h = hashlib.sha256(cv2.imread(str(SRC / f"{n}.png"), cv2.IMREAD_GRAYSCALE).tobytes()).hexdigest()
        if h in hashes:
            dropped.append([hashes[h], n])
        else:
            hashes[h] = n
    assert sorted(dropped[0]) == ["392_2HC", "392_HC"] and len(dropped) == 1, dropped
    dropped = [["392_HC", "392_2HC"]]  # [kept, dropped]: the patient's first image is kept
    names = [n for n in names if n != "392_2HC"]

    patient = {n: re.fullmatch(r"(\d+)_\d?HC", n).group(1) for n in names}
    patients = sorted(set(patient.values()))
    assert len(patients) == 806, len(patients)
    n_images = Counter(patient.values())
    first_hc = {p: hc[f"{p}_HC"] for p in patients}
    cuts = np.quantile(list(first_hc.values()), [1 / 3, 2 / 3])
    stratum = {p: int(np.searchsorted(cuts, v)) for p, v in first_hc.items()}
    split_of_patient = assign(patients, stratum, n_images)

    for part in ("images", "labels"):
        for s in SPLITS:
            (OUT / part / s).mkdir(parents=True)
    counts, by_tertile, fidelity, polys = Counter(), Counter(), [], 0
    for n in names:
        s = split_of_patient[patient[n]]
        outline = cv2.imread(str(SRC / f"{n}_Annotation.png"), cv2.IMREAD_GRAYSCALE) > 127
        m = filled_mask(outline)
        H, W = m.shape
        shutil.copyfile(SRC / f"{n}.png", OUT / "images" / s / f"{n}.png")
        ps = mask_to_polygons(m)
        assert ps, f"{n}: no polygon"
        redraw = np.zeros((H, W), np.uint8)
        lines = []
        for p in ps:
            cv2.fillPoly(redraw, [p.astype(np.int32)], 1)
            lines.append("0 " + " ".join(f"{v:.6f}" for v in (p / np.array([W, H])).ravel()))
        polys += len(ps)
        r = redraw.astype(bool)
        fidelity.append(2 * (r & m).sum() / (r.sum() + m.sum()))
        (OUT / "labels" / s / f"{n}.txt").write_text("\n".join(lines) + "\n")
        counts[s] += 1
        by_tertile[f"{s}/{['low', 'mid', 'high'][stratum[patient[n]]]}"] += 1

    assert len({(patient[n], split_of_patient[patient[n]]) for n in names}) == len(patients)
    fid = {"median": round(float(np.median(fidelity)), 4), "min": round(float(np.min(fidelity)), 4)}
    assert fid["median"] >= 0.97, fid

    (OUT / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: head\n")
    report = {"source_images": 999, "images": len(names), "patients": len(patients),
              "dropped_copies": dropped, "split": dict(counts),
              "patients_per_split": dict(Counter(split_of_patient.values())),
              "hc_tertile_cuts_mm": [round(float(c), 1) for c in cuts],
              "split_by_hc_tertile": dict(sorted(by_tertile.items())), "polygons": polys,
              "polygon_dice": fid, "split_of_patient": split_of_patient}
    (OUT / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")

    ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(OUT.rglob("*")):
            if f.is_file() and f.name != "build_report.json":
                z.write(f, f.relative_to(OUT))
    print(json.dumps({k: v for k, v in report.items() if k != "split_of_patient"}, indent=2))
    print(f"wrote {ZIP.relative_to(ROOT)} ({ZIP.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
