"""Build a public EVREN image version of PTB-XL (PhysioNet, CC BY 4.0; docs/datasets.md, PTB-XL entry;
plan docs/plan-ptbxl-public-v1.md).

PTB-XL holds signals, not images. Each 10-second 12-lead record (500 Hz, mV) is drawn here as a
standard paper ECG: 3 x 4 layout (I, aVR, V1, V4 / II, aVL, V2, V5 / III, aVF, V3, V6; each column
2.5 s of its own time window) plus a 10-second lead II rhythm strip, 25 mm/s, 10 mm/mV, red
1 mm / 5 mm grid, a 1 mV calibration pulse per row, lead names, no other text (no patient data,
no date, no device). Scale 8 px per mm: 2100 x 1360 px, PNG (lossless; 185 KB against 1.3 MB as JPEG q90).

Task: the 5 diagnostic superclasses of the PTB-XL paper (NORM, MI, STTC, CD, HYP; from
scp_statements.csv, diagnostic statements only, any likelihood), single-label records only (16,244
of 21,799; 5,144 records with more than one superclass and 411 without a diagnostic statement are
left out). Split: the authors' patient-wise stratified folds, 1-8 train, 9 val, 10 test.

The build stops if a check fails: the record counts per split and class equal the metadata count,
every signal 5000 x 12 without NaN, every image 2100 x 1360.

Run from the project root with the project venv (needs wfdb):
    .venv/bin/python -I scripts/ptbxl_public_v1.py [--limit N --sample-dir DIR]
Writes data/ptbxl_public/v1/<split>/<class>/<ecg_id>.png, build_report.json (local only) and
data/upload/ptbxl_public_v1_cls.zip
"""

import argparse
import ast
import csv
import json
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import wfdb

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "ptbxl" / "extracted" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
OUT = ROOT / "data" / "ptbxl_public" / "v1"
ZIP = ROOT / "data" / "upload" / "ptbxl_public_v1_cls.zip"
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
FOLDS = {"train": set(range(1, 9)), "val": {9}, "test": {10}}

PX = 8                      # pixels per mm
FS = 500
MM_S, MM_MV = 25, 10        # paper speed, gain
LEFT, RIGHT, TOP, BOTTOM = 60, 40, 40, 40
ROW_MM = 40
W = LEFT + 10 * MM_S * PX + RIGHT          # 2100
H = TOP + 4 * ROW_MM * PX + BOTTOM         # 1360
LAYOUT = [["I", "AVR", "V1", "V4"], ["II", "AVL", "V2", "V5"], ["III", "AVF", "V3", "V6"]]
LABEL = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}
MINOR, MAJOR, TRACE = (205, 205, 255), (140, 140, 235), (0, 0, 0)    # BGR


def grid():
    img = np.full((H, W, 3), 255, np.uint8)
    for x in range(LEFT % PX, W, PX):
        cv2.line(img, (x, 0), (x, H - 1), MINOR, 1)
    for y in range(TOP % PX, H, PX):
        cv2.line(img, (0, y), (W - 1, y), MINOR, 1)
    for x in range(LEFT, W, 5 * PX):
        cv2.line(img, (x, 0), (x, H - 1), MAJOR, 1)
    for y in range(TOP, H, 5 * PX):
        cv2.line(img, (0, y), (W - 1, y), MAJOR, 1)
    return img


GRID = grid()


def draw(signal, names):
    img = GRID.copy()
    idx = {n.upper(): i for i, n in enumerate(names)}
    px_per_sample = MM_S * PX / FS
    rows = LAYOUT + [["II"]]
    for r, leads in enumerate(rows):
        base = TOP + r * ROW_MM * PX + ROW_MM * PX // 2
        cal = np.array([[10, base], [18, base], [18, base - MM_MV * PX], [18 + 5 * PX, base - MM_MV * PX],
                        [18 + 5 * PX, base], [LEFT - 4, base]], np.int32)
        cv2.polylines(img, [cal], False, TRACE, 2, cv2.LINE_AA)
        span = 10 * FS // len(leads)
        for c, lead in enumerate(leads):
            seg = signal[c * span:(c + 1) * span, idx[lead]]
            x0 = LEFT + c * span * px_per_sample
            xs = x0 + np.arange(len(seg)) * px_per_sample
            ys = np.clip(base - seg * MM_MV * PX, 0, H - 1)
            pts = np.stack([xs, ys], 1).round().astype(np.int32)
            cv2.polylines(img, [pts], False, TRACE, 2, cv2.LINE_AA)
            if c:
                cv2.line(img, (int(x0), base - 3 * PX), (int(x0), base + 3 * PX), TRACE, 2)
            cv2.putText(img, LABEL.get(lead, lead), (int(x0) + 10, base - 12 * PX),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, TRACE, 2, cv2.LINE_AA)
    return img


def render(job):
    ecg_id, path, out = job
    signal, fields = wfdb.rdsamp(str(SRC / path))
    assert signal.shape == (5000, 12) and not np.isnan(signal).any(), (ecg_id, signal.shape)
    img = draw(signal, fields["sig_name"])
    assert img.shape == (H, W, 3)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return ecg_id


def records():
    diag = {}
    for r in csv.DictReader(open(SRC / "scp_statements.csv", encoding="utf-8")):
        code = r[""] if "" in r else next(iter(r.values()))
        if r["diagnostic"] in ("1", "1.0"):
            diag[code] = r["diagnostic_class"]
    out = []
    for r in csv.DictReader(open(SRC / "ptbxl_database.csv", encoding="utf-8")):
        sup = sorted({diag[k] for k in ast.literal_eval(r["scp_codes"]) if k in diag})
        if len(sup) == 1:
            fold = int(r["strat_fold"])
            split = next(s for s, f in FOLDS.items() if fold in f)
            out.append((int(r["ecg_id"]), r["filename_hr"], split, sup[0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--sample-dir")
    args = ap.parse_args()
    recs = records()
    counts = Counter((s, c) for _, _, s, c in recs)
    assert len(recs) == 16244 and sorted({c for *_, c in recs}) == sorted(CLASSES), len(recs)

    if args.sample_dir:   # a few images per class for a visual check, nothing else written
        picks = [next(r for r in recs if r[3] == c) for c in CLASSES][: args.limit or 5]
        for ecg_id, path, _, c in picks:
            render((ecg_id, path, Path(args.sample_dir) / f"{c}_{ecg_id:05d}.png"))
        print("samples in", args.sample_dir)
        return

    if OUT.exists():
        raise SystemExit(f"{OUT} exists; remove it first so no stale files enter the ZIP")
    jobs = [(i, p, OUT / s / c / f"{i:05d}.png") for i, p, s, c in recs]
    with ProcessPoolExecutor(8) as pool:
        for k, _ in enumerate(pool.map(render, jobs, chunksize=32), 1):
            if k % 2000 == 0:
                print(f"{k}/{len(jobs)}", flush=True)
    files = Counter((p.parent.parent.name, p.parent.name) for p in OUT.rglob("*.png"))
    assert files == counts, (files, counts)

    report = {"records": len(recs), "image_size": [W, H], "px_per_mm": PX,
              "split_by_class": {s: {c: counts[(s, c)] for c in CLASSES} for s in FOLDS},
              "split": {s: sum(counts[(s, c)] for c in CLASSES) for s in FOLDS}}
    (OUT / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(OUT.rglob("*.png")):
            z.write(f, f.relative_to(OUT))
    print(json.dumps(report, indent=1))
    print(f"wrote {ZIP.relative_to(ROOT)} ({ZIP.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
