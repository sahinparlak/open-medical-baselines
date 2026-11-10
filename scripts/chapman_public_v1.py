"""Build a public EVREN image version of the Chapman-Shaoxing / Ningbo 12-lead ECG database (PhysioNet
ecg-arrhythmia 1.0.0, CC BY 4.0; docs/datasets.md; plan docs/plan-chapman-public-v1.md).

Signals drawn exactly as the PTB-XL set (scripts/ptbxl_public_v1.py: 3 x 4 + lead II rhythm strip,
25 mm/s, 10 mm/mV, 2100 x 1360 PNG), so that models can be tested across the two sets.

Task: the 4 rhythm groups of Zheng et al. (Sci Data 2020): SB (sinus bradycardia), SR (sinus rhythm,
sinus irregularity), AFIB (atrial fibrillation, atrial flutter), GSVT (sinus tachycardia,
supraventricular tachycardia, atrial tachycardia, AVNRT, AVRT, sinus atrium to atrial wandering
rhythm), from the SNOMED codes in each header's #Dx line. Only records in exactly one group.
Records with a NaN sample or a flat lead (peak-to-peak < 0.01 mV) are left out.

Split: one record per patient (45,152 records, 45,152 names), so a seeded random 70/15/15 split
within each group is a patient-level split. After the split, records whose signal is identical to
another's (66 pairs, found by EVREN on import) keep one copy: the one in the latest split
(test > val > train), else the lowest record number; the others are left out
(annotations/chapman/duplicate_signals_removed.csv).

Run from the project root with the project venv (needs wfdb):
    .venv/bin/python -I scripts/chapman_public_v1.py
Writes data/chapman_public/v1/<split>/<group>/<record>.png, build_report.json (local only) and
data/upload/chapman_public_v1_cls.zip
"""

import hashlib
import json
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import wfdb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from ptbxl_public_v1 import H, W, draw  # noqa: E402

SRC = ROOT / "data" / "chapman_shaoxing" / "extracted" / \
    "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0" / "WFDBRecords"
OUT = ROOT / "data" / "chapman_public" / "v1"
ZIP = ROOT / "data" / "upload" / "chapman_public_v1_cls.zip"
GROUPS = {
    "SB": {"426177001"},
    "SR": {"426783006", "427393009"},
    "AFIB": {"164889003", "164890007"},
    "GSVT": {"427084000", "426761007", "713422000", "233896004", "233897008", "195101003"},
}
SHARES = {"train": 0.70, "val": 0.15, "test": 0.15}


def group_of(header):
    codes = set()
    for line in header.read_text().splitlines():
        if line.startswith("#Dx:"):
            codes = {c.strip() for c in line[4:].split(",") if c.strip()}
    g = [k for k, v in GROUPS.items() if codes & v]
    return g[0] if len(g) == 1 else None


def check(path):
    try:
        s, _ = wfdb.rdsamp(str(path))
    except Exception:            # noqa: BLE001 - an unreadable record is left out like a bad one
        return path, "unreadable", None
    if s.shape != (5000, 12):
        return path, "shape", None
    if np.isnan(s).any():
        return path, "nan", None
    if (np.ptp(s, axis=0) < 0.01).any():
        return path, "flat lead", None
    return path, None, hashlib.sha256(s.tobytes()).hexdigest()


def render(job):
    path, out = job
    s, f = wfdb.rdsamp(str(path))
    img = draw(s, f["sig_name"])
    assert img.shape == (H, W, 3)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def plan():
    headers = sorted(SRC.glob("*/*/*.hea"))
    assert len(headers) == 45152, len(headers)
    grouped = {h.with_suffix(""): g for h in headers if (g := group_of(h))}
    assert len(grouped) == 43365, len(grouped)

    with ProcessPoolExecutor(8) as pool:
        results = list(pool.map(check, list(grouped), chunksize=64))
    excluded = Counter(r for _, r, _ in results if r)
    good = [p for p, r, _ in results if r is None]
    digest = {p: d for p, r, d in results if r is None}

    rng = np.random.default_rng(0)
    split_of = {}
    for g in GROUPS:
        members = sorted(p for p in good if grouped[p] == g)
        order = rng.permutation(len(members))
        n_tr = round(SHARES["train"] * len(members))
        n_va = round(SHARES["val"] * len(members))
        for rank, i in enumerate(order):
            split_of[members[i]] = "train" if rank < n_tr else "val" if rank < n_tr + n_va else "test"

    same = defaultdict(list)
    for p in good:
        same[digest[p]].append(p)
    rank = {"train": 0, "val": 1, "test": 2}
    dropped = []
    for copies in same.values():
        keep = max(copies, key=lambda p: (rank[split_of[p]], -int(p.name[2:])))
        dropped += [(p, keep) for p in copies if p != keep]
    assert len(dropped) == 66, len(dropped)
    excluded["duplicate signal"] = len(dropped)
    gone = {p for p, _ in dropped}
    good = [p for p in good if p not in gone]
    return headers, grouped, excluded, good, split_of, dropped


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; remove it first so no stale files enter the ZIP")
    headers, grouped, excluded, good, split_of, _ = plan()

    jobs = [(p, OUT / split_of[p] / grouped[p] / f"{p.name}.png") for p in good]
    with ProcessPoolExecutor(8) as pool:
        for k, _ in enumerate(pool.map(render, jobs, chunksize=32), 1):
            if k % 5000 == 0:
                print(f"{k}/{len(jobs)}", flush=True)

    counts = Counter((split_of[p], grouped[p]) for p in good)
    files = Counter((f.parent.parent.name, f.parent.name) for f in OUT.rglob("*.png"))
    assert files == counts, (files, counts)
    report = {"records": len(headers), "single_group": len(grouped), "excluded": dict(excluded),
              "kept": len(good), "image_size": [W, H],
              "split_by_group": {s: {g: counts[(s, g)] for g in GROUPS} for s in SHARES},
              "split": {s: sum(counts[(s, g)] for g in GROUPS) for s in SHARES}}
    (OUT / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(OUT.rglob("*.png")):
            z.write(f, f.relative_to(OUT))
    print(json.dumps(report, indent=1))
    print(f"wrote {ZIP.relative_to(ROOT)} ({ZIP.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
