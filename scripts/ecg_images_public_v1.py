"""Build a public EVREN version of the ECG Images dataset of Cardiac Patients (Khan & Hussain,
Mendeley Data gwbz3fsgp8 v2, CC BY 4.0; entry in docs/datasets.md).

Source: 928 JPEG printouts (2213 x 1572) of 12-lead ECGs in four class folders: normal, MI,
abnormal heartbeat, history of MI. Downloaded to data/ecg_images/raw/<class>/ and checked
against the Mendeley SHA-256 hashes (data/ecg_images/jobs.tsv).

Changes made here:
- Exact copies dropped: only 491 images are unique (MI: 30 of 239 files; history of MI and
  normal: every image twice). The first file name in sort order is kept. No copy crosses classes.
- Cropped to the red ECG grid plus the traces that overflow it (rows 276-1537, cols 68-2176;
  the box is rows 283-1517, header text ends at row 275, footer text starts at row 1538). This
  removes the header (patient ID, sex) and the footer (exam date and time, filter settings,
  heart rate): identifiers and possible shortcuts. Saved as PNG (lossless; JPEG q95 came out larger).
- Split by report ID without reading it: images whose binarised ID-number field is pixel-identical
  share an ID (pairwise differences are 0 or >= 20 pixels, nothing between). 311 ID groups; 62
  of them hold images of more than one class. Whole groups go to one split; the split among
  2,000 seeded random group assignments whose per-class shares are closest to 70/15/15 is used.
  The IDs themselves are never decoded, printed or saved.

The build stops if a check fails: 928 files, 491 unique, class counts 30/233/86/142, red grid at
the same place in every image, no header or footer text inside the crop, 311 ID groups, every
class present in every split.

Run from the project root with the project venv:
    .venv/bin/python -I scripts/ecg_images_public_v1.py
Writes data/ecg_images_public/v1/<split>/<class>/*.png, build_report.json (local only) and
data/upload/ecg_images_public_v1_cls.zip
"""

import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "ecg_images" / "raw"
OUT = ROOT / "data" / "ecg_images_public" / "v1"
ZIP = ROOT / "data" / "upload" / "ecg_images_public_v1_cls.zip"
CLASSES = ["normal", "abnormal_heartbeat", "MI", "history_of_MI"]
PREFIX = {"normal": "Normal", "abnormal_heartbeat": "HB", "MI": "MI", "history_of_MI": "PMI"}
GRID = (283, 1518, 68, 2177)        # rows y0:y1, cols x0:x1 of the red grid box
CROP = (276, 1538, 68, 2177)        # grid plus traces that overflow it (header text ends at row
                                    # 275, footer text starts at row 1538 in every image)
ID_FIELD = (70, 113, 125, 272)      # rows, cols of the ID number in the header
SHARES = {"train": 0.70, "val": 0.15, "test": 0.15}


def red_box(rgb):
    r, g = rgb[..., 0].astype(int), rgb[..., 1].astype(int)
    red = (r > 200) & (g < 200) & (r - g > 40)
    ys, xs = np.where(red.mean(1) > 0.5)[0], np.where(red.mean(0) > 0.5)[0]
    return ys.min(), ys.max(), xs.min(), xs.max()


def choose_split(groups):
    """groups: {gid: Counter(class -> n)}. Return {gid: split}."""
    gids = sorted(groups)
    total = Counter()
    for c in groups.values():
        total.update(c)
    best, best_err = None, None
    for seed in range(2000):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(gids))
        assign, got = {}, {s: Counter() for s in SHARES}
        for i in order:  # greedy: put each group where its classes are most under target
            g = groups[gids[i]]
            def need(s):
                return sum(n * (SHARES[s] * total[c] - got[s][c]) / total[c] for c, n in g.items())
            s = max(SHARES, key=need)
            assign[gids[i]] = s
            got[s].update(g)
        err = sum((got[s][c] / total[c] - SHARES[s]) ** 2 for s in SHARES for c in CLASSES)
        if best_err is None or err < best_err:
            best, best_err = assign, err
    return best


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; remove it first so no stale files enter the ZIP")
    files = sorted(SRC.rglob("*.jpg"))
    assert len(files) == 928, len(files)

    seen, images, dropped = {}, [], 0
    for p in files:
        rgb = np.asarray(Image.open(p).convert("RGB"))
        assert rgb.shape == (1572, 2213, 3), (p, rgb.shape)
        h = hashlib.sha256(np.asarray(Image.open(p).convert("L")).tobytes()).hexdigest()
        if h in seen:
            assert seen[h].parent == p.parent, (p, seen[h])
            dropped += 1
            continue
        seen[h] = p
        y0, y1, x0, x1 = red_box(rgb)
        assert (y0, y1 + 1, x0, x1 + 1) == GRID, (p, y0, y1, x0, x1)
        dark = rgb.mean(2) < 128
        ink = dark[:, CROP[2]:CROP[3]].sum(1)
        assert CROP[1] + np.argmax(ink[CROP[1] - 20:] > 150) - 20 == CROP[1], p  # footer text row
        assert ink[CROP[0] - 1] == 0 or ink[CROP[0] - 1] < 40, p      # no header text line at the cut
        idf = dark[ID_FIELD[0]:ID_FIELD[1], ID_FIELD[2]:ID_FIELD[3]]
        assert idf.sum() > 300, p                                     # an ID is printed
        images.append({"path": p, "cls": p.parent.name, "gid": hashlib.sha1(idf.tobytes()).hexdigest()})

    per_class = Counter(m["cls"] for m in images)
    assert len(images) == 491 and dropped == 437, (len(images), dropped)
    assert per_class == {"MI": 30, "abnormal_heartbeat": 233, "history_of_MI": 86, "normal": 142}, per_class

    groups = defaultdict(Counter)
    for m in images:
        groups[m["gid"]][m["cls"]] += 1
    assert len(groups) == 311, len(groups)
    split_of = choose_split(groups)

    counts, gcount = Counter(), Counter()
    for m in images:
        s = split_of[m["gid"]]
        n = re.fullmatch(rf"{PREFIX[m['cls']]}\((\d+)\)\.jpg", m["path"].name).group(1)
        dst = OUT / s / m["cls"] / f"{PREFIX[m['cls']]}_{int(n):03d}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.open(m["path"]).convert("RGB").crop((CROP[2], CROP[0], CROP[3], CROP[1])).save(dst, optimize=True)
        counts[s, m["cls"]] += 1
    for g, s in split_of.items():
        gcount[s] += 1
    assert all(counts[s, c] > 0 for s in SHARES for c in CLASSES), counts

    report = {"source_files": 928, "unique_images": 491, "dropped_copies": dropped,
              "crop_rows_cols": CROP, "id_groups": len(groups),
              "mixed_class_groups": sum(len(c) > 1 for c in groups.values()),
              "groups_per_split": dict(gcount),
              "split": {s: {c: counts[s, c] for c in CLASSES} for s in SHARES}}
    (OUT / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")

    ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(OUT.rglob("*.png")):
            z.write(f, f.relative_to(OUT))
    print(json.dumps(report, indent=2))
    print(f"wrote {ZIP.relative_to(ROOT)} ({ZIP.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
