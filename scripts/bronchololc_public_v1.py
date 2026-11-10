"""Build a public EVREN version of BM-BronchoLC (flexible bronchoscopy; figshare 24243670, CC BY 4.0;
plan: docs/plan-bronchololc-public-v1.md).

208 patients (106 lung cancer, 102 not), one video each, frames at 1 per second, polygons of 11
airway landmarks and 7 lesions. Changes: the PNGs are stored with red and blue swapped, so the
channels are swapped back; everything outside the endoscopic field (the processor's side panel with
exam date and time, icons) is set to black; one blank frame (a plain "no signal" screen) is
dropped; a polygon with two classes becomes two lines; two source typos in class names are fixed. Split by patient, seeded, per group (lung cancer 74 / 16 /
16, non-lung cancer 72 / 15 / 15); patients that share an identical frame stay in one split and one
copy of the frame is dropped.

The field (a convex octagon) is found per frame: the longest run of columns, then of rows, in which
more than 35% of pixels are non-black (panel text is sparse); within that box, an opening with a
9 px square erases text strokes; the convex hull of the largest component, dilated by 2 px. Where
the field reaches under the panel, fragments of panel text stay (field labels, year digits). Some
outlines were drawn loosely past the field's edge; their points are counted, not moved.

The build stops if a check fails: 208 patients, 2,603 frames, every frame labelled, all coordinates
in [0, 1], no patient in two splits, no identical images.

Run from the project root with the project venv:
    .venv/bin/python -I scripts/bronchololc_public_v1.py
Writes data/bronchololc_public/ (images, labels, data.yaml, build_report.json) and
data/upload/bronchololc_public_v1_yoloseg.zip
"""

import hashlib
import json
import random
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "bronchololc" / "extract"
OUT = ROOT / "data" / "bronchololc_public"
ZIP = ROOT / "data" / "upload" / "bronchololc_public_v1_yoloseg.zip"
GROUPS = {"Lung_cancer": ("LC", 16, 16), "Non_lung_cancer": ("NLC", 15, 15)}  # prefix, val, test
RENAME = {"Muscosal erythema": "Mucosal erythema", "Anthrocosis": "Anthracosis"}
SEED = 0


def longest_run(flags):
    """(start, end) of the longest run of True values."""
    best, start = (0, 0, 0), None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        if not f and start is not None:
            best = max(best, (i - start, start, i))
            start = None
    return best[1], best[2]


def field_mask(bgr):
    on = (bgr.max(axis=2) > 12).astype(np.uint8)
    x0, x1 = longest_run(on.mean(axis=0) > 0.35)
    y0, y1 = longest_run(on[:, x0:x1].mean(axis=1) > 0.35)
    box = np.zeros_like(on)
    box[y0:y1, x0:x1] = on[y0:y1, x0:x1]
    box = cv2.morphologyEx(box, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, comp, stats, _ = cv2.connectedComponentsWithStats(box, connectivity=8)
    big = (comp == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    mask = np.zeros_like(on)
    cv2.fillConvexPoly(mask, cv2.convexHull(cv2.findNonZero(big)), 1)
    return cv2.dilate(mask, np.ones((5, 5), np.uint8))


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; remove it first so no stale files enter the ZIP")
    labels = json.loads((SRC / "Lung_cancer" / "labels.json").read_text())
    names = [RENAME.get(l["name"], l["name"]) for l in labels]
    cls = {l["id"]: i for i, l in enumerate(labels)}

    frames = []  # (group, study, image_id, path, annotations)
    for grp in GROUPS:
        objs = json.loads((SRC / grp / "objects.json").read_text())
        ann = json.loads((SRC / grp / "annotation.json").read_text())
        files = {p.stem: p for p in (SRC / grp / "imgs").rglob("*.png")}
        by_img = {}
        for a in ann:
            by_img.setdefault(a["object_id"], []).append(a)
        for o in objs:
            for v in o["videos"]:
                for im in v["images"]:
                    a = by_img[im["image_id"]]
                    assert len({x["study_id"] for x in a}) == 1
                    frames.append((grp, a[0]["study_id"], im["image_id"], files[im["image_id"]], a))

    blank = [f for f in frames if cv2.imread(str(f[3])).reshape(-1, 3).std(axis=0).max() < 8]
    assert len(blank) == 1, len(blank)
    frames = [f for f in frames if f not in blank]

    # Identical frames: keep the first, put the patients involved in one split unit.
    unit = {s: s for _, s, *_ in frames}
    def root(s):
        while unit[s] != s:
            s = unit[s]
        return s
    seen, dropped = {}, []
    for f in frames:
        h = hashlib.sha256(f[3].read_bytes()).hexdigest()
        if h in seen:
            unit[root(f[1])] = root(seen[h][1])
            dropped.append(f)
        else:
            seen[h] = f
    frames = [f for f in frames if f not in dropped]
    assert len(dropped) == 2 and len(frames) == 2603, (len(dropped), len(frames))

    split = {}
    rng = random.Random(SEED)
    for grp, (_, n_val, n_test) in GROUPS.items():
        units = {}
        for g, s, *_ in frames:
            if g == grp:
                units.setdefault(root(s), set()).add(s)
        order = rng.sample(sorted(units), len(units))
        done = Counter()
        for u in order:
            target = "test" if done["test"] < n_test else "val" if done["val"] < n_val else "train"
            for s in units[u]:
                split[s] = target
            done[target] += len(units[u])
    assert len(split) == 208

    for part in ("images", "labels"):
        for s in ("train", "val", "test"):
            (OUT / part / s).mkdir(parents=True)
    count, per_class, clipped, points, outside = Counter(), Counter(), 0, 0, 0
    index = Counter()
    for grp, study, image_id, path, anns in frames:
        s = split[study]
        bgr = cv2.imread(str(path))  # stored R/B-swapped: OpenCV's BGR order reads it back as RGB
        h, w = bgr.shape[:2]
        mask = field_mask(bgr)
        lines = []
        for a in anns:
            xy = np.array([[p["x"], p["y"]] for p in a["data"]], float)
            assert len(xy) >= 3
            inside = np.clip(xy, 0, [w - 1, h - 1]).astype(int)
            points += len(xy)
            outside += int((mask[inside[:, 1], inside[:, 0]] == 0).sum())  # loose outlines
            c = np.clip(xy, 0, [w, h])
            clipped += int((c != xy).any(axis=1).sum())
            norm = " ".join(f"{v:.6f}" for v in (c / [w, h]).ravel())
            for lid in a["label_ids"]:
                lines.append(f"{cls[lid]} {norm}\n")
                per_class[names[cls[lid]]] += 1
        out = bgr.copy()  # pixels in R,G,B order now, as PNG expects after the swap below
        out[mask == 0] = 0
        stem = f"{GROUPS[grp][0]}_{study[:8]}_{index[study]:03d}"
        index[study] += 1
        cv2.imwrite(str(OUT / "images" / s / f"{stem}.png"), out[..., ::-1])
        (OUT / "labels" / s / f"{stem}.txt").write_text("".join(lines))
        count[s] += 1
    (OUT / "data.yaml").write_text("names:\n" + "".join(f"  {i}: \"{n}\"\n" for i, n in enumerate(names)))

    pats = {s: {p.stem.rsplit("_", 1)[0] for p in (OUT / "images" / s).glob("*.png")} for s in count}
    assert not (pats["train"] & pats["val"] or pats["train"] & pats["test"] or pats["val"] & pats["test"])
    for s in ("train", "val", "test"):
        imgs = {p.stem for p in (OUT / "images" / s).glob("*.png")}
        labs = {p.stem for p in (OUT / "labels" / s).glob("*.txt") if p.stat().st_size}
        assert imgs == labs and len(imgs) == count[s], s
    hashes = Counter(hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT / "images").rglob("*.png"))
    assert max(hashes.values()) == 1, "duplicate images"

    report = {"patients": {s: len(pats[s]) for s in ("train", "val", "test")},
              "frames": {s: count[s] for s in ("train", "val", "test")},
              "polygon_lines": sum(per_class.values()), "clipped_points": clipped,
              "points_outside_field": f"{outside} of {points}",
              "dropped_identical": [f"{f[0]}/{f[2]}" for f in dropped],
              "dropped_blank": [f"{f[0]}/{f[2]}" for f in blank],
              "per_class": {n: per_class[n] for n in names}}
    (OUT / "build_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED) as zo:
        zo.write(OUT / "data.yaml", "data.yaml")
        for p in sorted((OUT / "images").rglob("*.png")) + sorted((OUT / "labels").rglob("*.txt")):
            zo.write(p, p.relative_to(OUT))
    with zipfile.ZipFile(ZIP) as zi:
        assert zi.testzip() is None
        n = len(zi.namelist())
    print(f"zip {ZIP.stat().st_size / 1e9:.2f} GB, {n} entries -> {ZIP.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
