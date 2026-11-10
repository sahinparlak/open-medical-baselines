"""Build a public EVREN version of the HyperKvasir labelled images (OSF mh9sj, CC BY 4.0;
docs/datasets.md, HyperKvasir labelled images entry).

10,662 JPEG endoscopy images in 23 class folders (image-labels.csv agrees with the folders; no
byte-identical copies). Images are copied byte for byte; class names are the source folder names.

Split (no patient or exam ids in the source; the authors' official 2-/5-fold splits are image-level
random, so near-identical frames fall in both folds):
- The 1,000 polyp images that are also the segmented polyp set keep their split from
  data/hyperkvasir_polyp_public/v1/build_report.json, so the two EVREN sets agree.
- Groups: pairs in annotations/hyperkvasir_labeled/near_duplicate_pairs.csv (DINOv2 similarity
  >= 0.95, stool classes left out; scripts/hyperkvasir_labeled_near_duplicates.py) plus the polyp
  set's reviewed "same"/"same_uncertain" pairs; a pair the polyp review called "different" is not
  joined. Connected pairs form one group.
- The stool classes (bbps-0-1, bbps-2-3, impacted-stool) cannot be grouped by similarity: each
  image is its own group, so frames of one exam may fall in different splits there.
- Groups go 70/15/15 by a seeded search over 2,000 random greedy orders, keeping the assignment
  closest to the per-class target counts (pre-assigned polyp images counted).

The build stops if a check fails: 10,662 images, 23 classes, no group in two splits, no polyp image
off its polyp-set split, every class in every split.

Run from the project root with the project venv:
    .venv/bin/python -I scripts/hyperkvasir_labeled_public_v1.py
Writes data/hyperkvasir_labeled_public/v1/<split>/<class>/*.jpg, build_report.json (local only)
and data/upload/hyperkvasir_labeled_public_v1_cls.zip
"""

import csv
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "hyperkvasir_labeled" / "extracted" / "labeled-images"
PAIRS = ROOT / "annotations" / "hyperkvasir_labeled" / "near_duplicate_pairs.csv"
POLYP_PAIRS = ROOT / "annotations" / "hyperkvasir" / "near_duplicate_pairs.csv"
POLYP_REPORT = ROOT / "data" / "hyperkvasir_polyp_public" / "v1" / "build_report.json"
OUT = ROOT / "data" / "hyperkvasir_labeled_public" / "v1"
ZIP = ROOT / "data" / "upload" / "hyperkvasir_labeled_public_v1_cls.zip"
SPLITS = ["train", "val", "test"]
SHARES = np.array([0.70, 0.15, 0.15])
TRIES = 2000


def groups_from_pairs(names, links, blocked):
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in links:
        if frozenset((a, b)) not in blocked:
            parent[find(a)] = find(b)
    groups = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    return sorted(groups.values(), key=lambda g: sorted(g)[0])


def assign(groups, cls_of, classes, fixed):
    """Groups holding a polyp-set image take its split; the rest go, in a seeded random order, to
    the split furthest below its per-class target. Keeps the order with the smallest squared error."""
    ci = {c: k for k, c in enumerate(classes)}
    total = np.bincount([ci[cls_of[n]] for g in groups for n in g], minlength=len(classes))
    target = SHARES[:, None] * total[None, :]
    base = np.zeros_like(target)
    free, out = [], {}
    for gi, g in enumerate(groups):
        vec = np.bincount([ci[cls_of[n]] for n in g], minlength=len(classes))
        s = {fixed[n] for n in g if n in fixed}
        assert len(s) <= 1, f"group with polyp images from two splits: {g}"
        if s:
            k = SPLITS.index(s.pop())
            base[k] += vec
            out[gi] = k
        else:
            free.append((gi, vec))
    rng = np.random.default_rng(0)
    best, best_cost = None, np.inf
    for _ in range(TRIES):
        have = base.copy()
        trial = {}
        for t in rng.permutation(len(free)):
            gi, vec = free[t]
            k = int(np.argmax(((target - have) / np.maximum(target, 1e-9)) @ vec))
            have[k] += vec
            trial[gi] = k
        cost = (((have - target) / np.maximum(total, 1)) ** 2).sum()
        if cost < best_cost:
            best, best_cost = trial, cost
    out.update(best)
    return out


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; remove it first so no stale files enter the ZIP")
    rows = list(csv.DictReader(open(SRC / "image-labels.csv", encoding="utf-8")))
    cls_of = {r["Video file"]: r["Finding"] for r in rows}
    paths = {p.stem: p for p in SRC.rglob("*.jpg")}
    names = sorted(cls_of)
    classes = sorted(set(cls_of.values()))
    assert len(names) == 10662 and set(names) == set(paths) and len(classes) == 23
    assert all(paths[n].parent.name == cls_of[n] for n in names)

    fixed = json.loads(POLYP_REPORT.read_text())["split_of"]
    assert len(fixed) == 1000 and all(cls_of[n] == "polyps" for n in fixed)
    links = [(r["a"], r["b"]) for r in csv.DictReader(open(PAIRS, encoding="utf-8"))]
    polyp_rows = list(csv.DictReader(open(POLYP_PAIRS, encoding="utf-8")))
    links += [(r["a"], r["b"]) for r in polyp_rows if r["decision"] in ("same", "same_uncertain")]
    blocked = {frozenset((r["a"], r["b"])) for r in polyp_rows if r["decision"] == "different"}
    groups = groups_from_pairs(names, links, blocked)

    split_of = {}
    for gi, k in assign(groups, cls_of, classes, fixed).items():
        for n in groups[gi]:
            split_of[n] = SPLITS[k]
    for g in groups:
        assert len({split_of[n] for n in g}) == 1, g
    assert all(split_of[n] == s for n, s in fixed.items())
    by_class = Counter((cls_of[n], split_of[n]) for n in names)
    assert all(by_class[(c, s)] > 0 for c in classes for s in SPLITS), by_class

    for n in names:
        dst = OUT / split_of[n] / cls_of[n]
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(paths[n], dst / f"{n}.jpg")

    sizes = Counter(len(g) for g in groups)
    report = {
        "images": len(names), "classes": len(classes), "groups": len(groups),
        "multi_image_groups": sum(len(g) > 1 for g in groups),
        "images_in_multi_image_groups": sum(len(g) for g in groups if len(g) > 1),
        "largest_group": max(sizes), "links_used": len(links) - sum(frozenset(l) in blocked for l in links),
        "split": dict(Counter(split_of.values())),
        "split_by_class": {c: [by_class[(c, s)] for s in SPLITS] for c in classes},
        "split_of": split_of,
    }
    (OUT / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")

    ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(OUT.rglob("*.jpg")):
            z.write(f, f.relative_to(OUT))
    print(json.dumps({k: v for k, v in report.items() if k != "split_of"}, indent=1))
    print(f"wrote {ZIP.relative_to(ROOT)} ({ZIP.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
