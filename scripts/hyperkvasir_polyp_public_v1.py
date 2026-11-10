"""Build a public EVREN version of the HyperKvasir segmented polyp images (OSF mh9sj, CC BY 4.0;
docs/datasets.md, HyperKvasir entry).

1,000 colonoscopy images (JPEG, 333 sizes) with binary polyp masks (JPEG, threshold 127). Images
are copied byte for byte; masks become YOLO polygons (one class, polyp; outer contour per part,
holes joined by keyholes, as in dsad_to_yolo_seg.py) and are checked by drawing them again (Dice
against the source mask).

Split (no patient or exam ids in the source): near-duplicate groups from
annotations/hyperkvasir/near_duplicate_pairs.csv (pairs decided "same" or "same_uncertain";
connected pairs form one group) stay in one split. Groups are assigned 70/15/15 by a seeded search
over 2,000 random greedy orders, keeping the assignment closest to the target counts in each
polyp-size tertile (mask area as a fraction of the image).

The build stops if a check fails: 1,000 images and masks with matching sizes, 951 groups, no group
in two splits, every image with at least one polygon, median polygon Dice >= 0.97.

Run from the project root with the project venv:
    .venv/bin/python -I scripts/hyperkvasir_polyp_public_v1.py
Writes data/hyperkvasir_polyp_public/v1/ (images, labels, data.yaml, build_report.json) and
data/upload/hyperkvasir_polyp_public_v1_yoloseg.zip
"""

import csv
import json
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

SRC = ROOT / "data" / "hyperkvasir" / "raw" / "extracted" / "segmented-images"
PAIRS = ROOT / "annotations" / "hyperkvasir" / "near_duplicate_pairs.csv"
OUT = ROOT / "data" / "hyperkvasir_polyp_public" / "v1"
ZIP = ROOT / "data" / "upload" / "hyperkvasir_polyp_public_v1_yoloseg.zip"
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
TRIES = 2000


def groups_from_pairs(names):
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in csv.DictReader(open(PAIRS, encoding="utf-8")):
        if r["decision"] in ("same", "same_uncertain"):
            parent[find(r["a"])] = find(r["b"])
    groups = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    return sorted(groups.values(), key=lambda g: sorted(g)[0])


def assign(groups, stratum):
    """Seeded greedy search: each group goes to the split furthest below its target in the group's
    size tertiles."""
    n_strata = max(stratum.values()) + 1
    target = np.array([[sum(stratum[n] == s for g in groups for n in g) * f for s in range(n_strata)]
                       for f in SPLITS.values()])
    rng = np.random.default_rng(0)
    best, best_cost = None, np.inf
    for _ in range(TRIES):
        have = np.zeros_like(target)
        out = {}
        for gi in rng.permutation(len(groups)):
            vec = np.bincount([stratum[n] for n in groups[gi]], minlength=n_strata)
            k = int(np.argmax(((target - have) / target) @ vec))
            have[k] += vec
            out[gi] = k
        cost = ((have - target) ** 2).sum()
        if cost < best_cost:
            best, best_cost = out, cost
    return best


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; remove it first so no stale files enter the ZIP")
    names = sorted(p.stem for p in (SRC / "images").glob("*.jpg"))
    assert len(names) == 1000 and names == sorted(p.stem for p in (SRC / "masks").glob("*.jpg"))

    masks, area = {}, {}
    for n in names:
        img = cv2.imread(str(SRC / "images" / f"{n}.jpg"))
        m = cv2.imread(str(SRC / "masks" / f"{n}.jpg"), cv2.IMREAD_GRAYSCALE) > 127
        assert img.shape[:2] == m.shape and m.any(), n
        masks[n], area[n] = m, m.mean()
    cuts = np.quantile(list(area.values()), [1 / 3, 2 / 3])
    stratum = {n: int(np.searchsorted(cuts, a)) for n, a in area.items()}

    groups = groups_from_pairs(names)
    assert len(groups) == 951, len(groups)
    split_of = {}
    for gi, k in assign(groups, stratum).items():
        for n in groups[gi]:
            split_of[n] = list(SPLITS)[k]

    for part in ("images", "labels"):
        for s in SPLITS:
            (OUT / part / s).mkdir(parents=True)
    counts, by_size, fidelity, polys = Counter(), Counter(), [], 0
    for n in names:
        s, m = split_of[n], masks[n]
        h, w = m.shape
        shutil.copyfile(SRC / "images" / f"{n}.jpg", OUT / "images" / s / f"{n}.jpg")
        ps = mask_to_polygons(m)
        assert ps, f"{n}: no polygon"
        redraw = np.zeros((h, w), np.uint8)
        lines = []
        for p in ps:
            cv2.fillPoly(redraw, [p.astype(np.int32)], 1)
            lines.append("0 " + " ".join(f"{v:.6f}" for v in (p / np.array([w, h])).ravel()))
        polys += len(ps)
        r = redraw.astype(bool)
        fidelity.append(2 * (r & m).sum() / (r.sum() + m.sum()))
        (OUT / "labels" / s / f"{n}.txt").write_text("\n".join(lines) + "\n")
        counts[s] += 1
        by_size[f"{s}/{['small', 'medium', 'large'][stratum[n]]}"] += 1

    for g in groups:
        assert len({split_of[n] for n in g}) == 1, g
    fid = {"median": round(float(np.median(fidelity)), 4), "min": round(float(np.min(fidelity)), 4)}
    assert fid["median"] >= 0.97, fid
    group_split = Counter(split_of[g[0]] for g in groups)
    multi_split = Counter(split_of[g[0]] for g in groups if len(g) > 1)

    (OUT / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: polyp\n")
    report = {"images": len(names), "groups": len(groups),
              "multi_image_groups": sum(len(g) > 1 for g in groups),
              "split": dict(counts), "groups_per_split": dict(group_split),
              "multi_image_groups_per_split": dict(multi_split),
              "size_tertile_cuts_area_fraction": [round(float(c), 4) for c in cuts],
              "split_by_size": dict(sorted(by_size.items())), "polygons": polys, "polygon_dice": fid,
              "split_of": split_of}
    (OUT / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")

    ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(OUT.rglob("*")):
            if f.is_file() and f.name != "build_report.json":
                z.write(f, f.relative_to(OUT))
    print(json.dumps({k: v for k, v in report.items() if k != "split_of"}, indent=2))
    print(f"wrote {ZIP.relative_to(ROOT)} ({ZIP.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
