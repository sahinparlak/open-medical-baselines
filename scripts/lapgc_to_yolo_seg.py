"""Convert LapGC-KVAD-30 (gastric cancer key vessels, CC BY 4.0; docs/datasets.md) to YOLO
segmentation format for EVREN: a separate vessel dataset, not part of the anatomy classes.

Masks are RGB PNGs with one colour per vessel class (color.json, by scene S1-S8); black is
background. The 15 classes are numbered in color.json order; "&" in a name becomes "_"
(LGeA&V -> LGeA_V). Pixels of colours outside the table are left unlabelled and reported.
The official split by case is kept: Training Set -> train (20 cases), Validation Set -> val (5),
Test Set -> test (5; closed until the final model, as for DSAD).

Polygons as in scripts/dsad_to_yolo_seg.py (one per connected component, holes joined by
keyhole cuts); fidelity is checked by rasterizing them again (Dice against the source mask).
Frames are written as JPEG (quality 95): lapgc_<case>_<scene>_<n>.jpg. Frames or masks that do
not decode (truncated PNGs in the source) are skipped and listed in the manifest. Each frame's vessel list in
the per-case CSV is compared with the colours found in its mask.

Run from the project root with the project venv:
    .venv/bin/python scripts/lapgc_to_yolo_seg.py
Writes data/lapgc_yolo/v1/ (images, labels, data.yaml, manifest.csv) and
data/upload/lapgc_kvad30_vessels_v1_yoloseg.zip
"""

import ast
import csv
import io
import json
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dsad_to_yolo_seg import mask_to_polygons  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "lapgc_kvad30" / "LapGC-KVAD-30.zip"
OUT = ROOT / "data" / "lapgc_yolo" / "v1"
ZIP = ROOT / "data" / "upload" / "lapgc_kvad30_vessels_v1_yoloseg.zip"
SPLITS = {"Training Set": "train", "Validation Set": "val", "Test Set": "test"}


def class_table():
    colours = json.loads(zipfile.ZipFile(SRC).read("color.json"))
    return [(name, tuple(rgb)) for scene in sorted(colours, key=lambda s: int(s[1:])) for name, rgb in colours[scene].items()]


def convert_case(args):
    folder, case = args
    split = SPLITS[folder]
    z = zipfile.ZipFile(SRC)
    table = class_table()
    rows = list(csv.DictReader(io.StringIO(z.read(f"{folder}/{case}/{case}.csv").decode("utf-8-sig"))))
    stats, unknown, manifest = [], Counter(), []
    for r in rows:
        scene = r["Scene"]
        img = cv2.imdecode(np.frombuffer(z.read(f"{folder}/{case}/{scene}/frames/{r['FileName']}"), np.uint8), cv2.IMREAD_COLOR)
        mask = cv2.imdecode(np.frombuffer(z.read(f"{folder}/{case}/{scene}/annotations/{r['MaskName']}"), np.uint8),
                            cv2.IMREAD_COLOR)
        name = "lapgc_" + Path(r["FileName"]).stem
        if img is None or mask is None:  # truncated PNG in the source: skipped and reported
            manifest.append({"frame": name, "case": case, "scene": scene, "split": split,
                             "vessels_csv": r["Vessel"], "vessels_mask": "", "csv_matches_mask": 0,
                             "skipped": "frame unreadable" if img is None else "mask unreadable"})
            continue
        mask = mask[..., ::-1]  # BGR -> RGB
        h, w = img.shape[:2]
        assert mask.shape[:2] == (h, w), r["FileName"]
        cv2.imwrite(str(OUT / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        flat = mask.reshape(-1, 3)
        known = np.zeros(len(flat), bool) | (flat == 0).all(1)
        lines, found = [], []
        for cls, (cname, rgb) in enumerate(table):
            m = (flat == rgb).all(1)
            known |= m
            if not m.any():
                continue
            m = m.reshape(h, w)
            polys = mask_to_polygons(m)
            recon = np.zeros((h, w), np.uint8)
            for p in polys:
                cv2.fillPoly(recon, [p], 1)
                lines.append(f"{cls} " + " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in p))
            if polys:
                found.append(cname)
            inter = np.logical_and(recon > 0, m).sum()
            stats.append((cname, 2 * inter / (recon.sum() + m.sum()), len(polys), int(m.sum())))
        for rgb, n in Counter(map(tuple, flat[~known])).items():
            unknown[rgb] += n
        (OUT / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        listed = ast.literal_eval(r["Vessel"]) if r["Vessel"].strip().startswith("[") else [r["Vessel"]]
        manifest.append({"frame": name, "case": case, "scene": scene, "split": split,
                         "vessels_csv": "+".join(listed), "vessels_mask": "+".join(found),
                         "csv_matches_mask": int(set(listed) == set(found)), "skipped": ""})
    return stats, unknown, manifest


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for part in ("images", "labels"):
        for split in SPLITS.values():
            (OUT / part / split).mkdir(parents=True)
    table = class_table()
    names = [n.replace("&", "_") for n, _ in table]
    z = zipfile.ZipFile(SRC)
    cases = sorted({(x.split("/")[0], x.split("/")[1]) for x in z.namelist() if x.split("/")[0] in SPLITS and x.count("/") > 1},
                   key=lambda fc: (list(SPLITS).index(fc[0]), int(fc[1])))
    print(f"{len(cases)} cases: " + ", ".join(f"{s} {sum(1 for f, _ in cases if SPLITS[f] == s)}" for s in SPLITS.values()))
    with ProcessPoolExecutor(8) as pool:
        results = list(pool.map(convert_case, cases))

    (OUT / "data.yaml").write_text("names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    manifest = [m for _, _, ms in results for m in ms]
    with open(OUT / "manifest.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(manifest[0]))
        wr.writeheader()
        wr.writerows(manifest)

    by = defaultdict(list)
    unknown = Counter()
    for stats, u, _ in results:
        unknown.update(u)
        for cname, d, n, px in stats:
            by[cname].append((d, n, px))
    print(f"\n{'class':<10}{'masks':>7}{'px median':>11}{'Dice median':>13}{'Dice min':>10}{'<0.99':>7}{'polys/mask':>12}")
    for cname, _ in table:
        v = by[cname]
        if not v:
            print(f"{cname:<10}{0:>7}")
            continue
        ds, ns, px = (np.array(x) for x in zip(*v))
        print(f"{cname:<10}{len(ds):>7}{int(np.median(px)):>11}{np.median(ds):>13.4f}{ds.min():>10.4f}{(ds < 0.99).sum():>7}{ns.mean():>12.2f}")
    total = sum(px for v in by.values() for _, _, px in v)
    print(f"\nunlabelled pixels of colours outside the table: {sum(unknown.values())} "
          f"({sum(unknown.values()) / max(total, 1):.4%} of labelled vessel pixels); top: {unknown.most_common(5)}")
    skipped = [m for m in manifest if m["skipped"]]
    print(f"skipped (unreadable PNG in the source): {len(skipped)}"
          + (f": {[(m['frame'], m['split'], m['skipped']) for m in skipped]}" if skipped else ""))
    manifest = [m for m in manifest if not m["skipped"]]
    mism = [m for m in manifest if not m["csv_matches_mask"]]
    print(f"frames whose CSV vessel list differs from the mask colours: {len(mism)} of {len(manifest)}"
          + (f"; e.g. {[(m['frame'], m['vessels_csv'], m['vessels_mask']) for m in mism[:3]]}" if mism else ""))
    per_split = Counter(m["split"] for m in manifest)
    empty = Counter(m["split"] for m in manifest if not m["vessels_mask"])
    print(f"frames {dict(per_split)}; without any vessel polygon {dict(empty)}")

    ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED) as zo:
        zo.write(OUT / "data.yaml", "data.yaml")
        for p in sorted((OUT / "images").rglob("*.jpg")) + sorted((OUT / "labels").rglob("*.txt")):
            zo.write(p, p.relative_to(OUT))
    print(f"zip {ZIP.stat().st_size / 1e9:.2f} GB -> {ZIP.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
