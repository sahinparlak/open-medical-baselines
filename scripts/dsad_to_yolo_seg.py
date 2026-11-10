"""Convert the fully labeled DSAD multilabel subset to YOLO segmentation format.

Only `multilabel/` frames are used: every one of its 7 structures is annotated in every
frame (merged mask of 3 annotators), so absent masks are true negatives. Single-organ
folders are partially labeled and are handled later with model-assisted completion.

Each connected component of a mask becomes one polygon of that class. Holes cannot be
stored in YOLO polygons, so each hole is joined to its outer contour by a zero-width cut
("keyhole"), which fills back to the same shape. Fidelity is measured by rasterizing the
polygons again and computing Dice against the source mask.

Images are written as JPEG (quality 95) at full resolution; file names keep the surgery
id: s<surgery>_ml_<index>.jpg. Splits come from splits/dsad_surgeries_v1.csv.

Run from the project root with the project venv:
    .venv/bin/python scripts/dsad_to_yolo_seg.py
Writes data/dsad_yolo/multilabel_v1/ and data/upload/dsad_multilabel_v1_yoloseg.zip
"""

import csv
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ML = ROOT / "data" / "dsad" / "multilabel"
OUT = ROOT / "data" / "dsad_yolo" / "multilabel_v1"
ZIP = ROOT / "data" / "upload" / "dsad_multilabel_v1_yoloseg.zip"

CLASSES = ["abdominal_wall", "colon", "small_intestine", "stomach", "liver", "spleen",
           "pancreas", "ureter", "vesicular_glands", "inferior_mesenteric_artery",
           "intestinal_veins"]
CLASS_ID = {name: i for i, name in enumerate(CLASSES)}
MIN_AREA = 25        # px; drop specks
EPSILON = 1.0        # px; polygon simplification tolerance


def keyhole(outer, holes):
    """Join holes to the outer contour with zero-width bridges."""
    poly = outer.reshape(-1, 2).tolist()
    for hole in holes:
        h = hole.reshape(-1, 2).tolist()
        o = np.array(poly)
        hi, oi = min(((i, int(np.argmin(((o - p) ** 2).sum(1)))) for i, p in enumerate(h)),
                     key=lambda t: ((o[t[1]] - h[t[0]]) ** 2).sum())
        h = h[hi:] + h[:hi]
        poly = poly[:oi + 1] + h + [h[0]] + poly[oi:]
    return np.array(poly, dtype=np.int32)


def mask_to_polygons(mask):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    polys = []
    if hierarchy is None:
        return polys
    hierarchy = hierarchy[0]
    for i, c in enumerate(contours):
        if hierarchy[i][3] != -1 or cv2.contourArea(c) < MIN_AREA:
            continue
        holes, child = [], hierarchy[i][2]
        while child != -1:
            if cv2.contourArea(contours[child]) >= MIN_AREA:
                holes.append(cv2.approxPolyDP(contours[child], EPSILON, True))
            child = hierarchy[child][0]
        outer = cv2.approxPolyDP(c, EPSILON, True)
        if len(outer) >= 3:
            polys.append(keyhole(outer, holes) if holes else outer.reshape(-1, 2))
    return polys


def convert_surgery(args):
    surgery, split = args
    src = ML / surgery
    stats = []
    for image in sorted(src.glob("image*.png")):
        idx = image.stem.replace("image", "")
        name = f"s{surgery}_ml_{idx}"
        img = cv2.imread(str(image))
        h, w = img.shape[:2]
        cv2.imwrite(str(OUT / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines = []
        for mask_path in sorted((src / "merged").glob(f"mask{idx}_*.png")):
            organ = mask_path.stem.split("_", 1)[1]
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
            if not mask.any():
                continue
            polys = mask_to_polygons(mask)
            recon = np.zeros((h, w), np.uint8)
            for p in polys:
                cv2.fillPoly(recon, [p], 1)
                coords = " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in p)
                lines.append(f"{CLASS_ID[organ]} {coords}")
            inter = np.logical_and(recon > 0, mask).sum()
            stats.append((organ, 2 * inter / (recon.sum() + mask.sum()), len(polys)))
        (OUT / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return stats


def main():
    split_of = {r["surgery"]: r["split"] for r in csv.DictReader(open(ROOT / "splits" / "dsad_surgeries_v1.csv"))}
    for part in ("images", "labels"):
        for split in ("train", "val", "test"):
            (OUT / part / split).mkdir(parents=True, exist_ok=True)
    jobs = [(s.name, split_of[s.name]) for s in sorted(ML.iterdir()) if s.is_dir()]
    with ProcessPoolExecutor() as pool:
        stats = [x for part in pool.map(convert_surgery, jobs) for x in part]

    (OUT / "data.yaml").write_text("names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)))

    by = defaultdict(list)
    for organ, d, n in stats:
        by[organ].append((d, n))
    print(f"{'structure':<18}{'masks':>7}{'Dice median':>13}{'Dice min':>10}{'<0.99':>7}{'polys/mask':>12}")
    for organ in CLASSES:
        if organ in by:
            ds = np.array([d for d, _ in by[organ]])
            ns = np.array([n for _, n in by[organ]])
            print(f"{organ:<18}{len(ds):>7}{np.median(ds):>13.4f}{ds.min():>10.4f}{(ds < 0.99).sum():>7}{ns.mean():>12.2f}")

    ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED) as z:
        z.write(OUT / "data.yaml", "data.yaml")
        for f in sorted(OUT.rglob("*")):
            if f.is_file() and f.name != "data.yaml" and not f.name.startswith("._"):
                z.write(f, f.relative_to(OUT))
    counts = {s: len(list((OUT / "images" / s).glob("*.jpg"))) for s in ("train", "val", "test")}
    print(f"\nimages {counts}  zip {ZIP.stat().st_size / 1e9:.2f} GB → {ZIP.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
