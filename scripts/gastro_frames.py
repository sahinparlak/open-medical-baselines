"""Laparoscopic gastrectomy frames (Zhang et al., Mil Med Res 2026; figshare 30058882, CC BY 4.0).

Reads the Roboflow COCO export straight from the zip (docs/datasets.md), groups frames by
surgery and maps the dataset's classes to ours. Shared by scripts/evaluate_gastrectomy.py and
scripts/build_anatomy_v5.py.

Grouping: file names follow four styles. Two carry the recording day ("YYYY-MM-DD_hhmmss_VIDnnn"
and "video-YYYYMMDDhhmmss"); frames of one day form a group. The two "V<nnn>_..." styles are
merged into one group, the "<n>-OR<n>-" style is one group: 18 groups for the paper's 18
surgeries. A day holding two surgeries would only make a group larger, never split a surgery.
Groups are numbered g01..g18 in sorted order of their keys; the keys (dates) stay in memory and
are not written anywhere.

Class mapping: six classes are ours (stomach, colon, intestine -> small_intestine, liver,
spleen, pancreas). Omentum, gallbladder, ligament, vessels, blood and instruments are not in our
class list and count as background for us. Abdominal wall is ours but not labelled here.

Run from the project root to (re)write the surgery split (seeded, see make_split):
    .venv/bin/python scripts/gastro_frames.py
Writes splits/gastrectomy_surgeries_v1.csv (group, split, images) and extracts the frames to
data/gastrectomy_mmr2026/frames/<group>_<nnnn>.jpg.
"""

import csv
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ZIP = ROOT / "data" / "gastrectomy_mmr2026" / "gastrectomy_frames.zip"
SPLIT = ROOT / "splits" / "gastrectomy_surgeries_v1.csv"
FRAMES = ROOT / "data" / "gastrectomy_mmr2026" / "frames"
W, H = 1920, 1080
MAP = {"stomach": "stomach", "colon": "colon", "intestine": "small_intestine", "liver": "liver",
       "spleen": "spleen", "pancreas": "pancreas"}
OURS = list(MAP.values())


def group_key(file_name):
    if m := re.match(r"(\d{4}-\d{2}-\d{2})_\d{6}_", file_name):
        return "day " + m.group(1)
    if m := re.match(r"video-(\d{4})(\d{2})(\d{2})\d{6}_", file_name):
        return f"day {m.group(1)}-{m.group(2)}-{m.group(3)}"
    if re.match(r"V\d+_", file_name):
        return "prefix V"
    if re.match(r"\d+-OR\d+-_", file_name):
        return "prefix OR"
    raise ValueError(f"unknown file name style: {file_name[:12]}...")


def base_name(file_name):
    return re.sub(r"\.rf\.[0-9a-f]+\.jpg$", "", file_name)


def load():
    """Frames as dicts: id, file (zip member), group (g01..), polygons {our class: [Nx2 arrays]},
    other (count of polygons of classes that are not ours), named {dataset class: [Nx2 arrays]}
    for all classes (instruments, omentum, ...). Roboflow duplicates (same base
    name) keep their first copy; frames without any annotation are dropped."""
    z = zipfile.ZipFile(ZIP)
    d = json.loads(z.read("train/_annotations.coco.json"))
    names = {c["id"]: c["name"] for c in d["categories"]}
    keys = sorted({group_key(i["file_name"]) for i in d["images"]})
    gid = {k: f"g{n + 1:02d}" for n, k in enumerate(keys)}
    anns = defaultdict(list)
    for a in d["annotations"]:
        anns[a["image_id"]].append(a)
    frames, seen = [], set()
    for img in sorted(d["images"], key=lambda i: i["file_name"]):
        b = base_name(img["file_name"])
        if b in seen or not anns[img["id"]]:
            continue
        seen.add(b)
        assert (img["width"], img["height"]) == (W, H)
        polys, other, named = defaultdict(list), 0, defaultdict(list)
        for a in anns[img["id"]]:
            name = names[a["category_id"]]
            ours = MAP.get(name)
            other += ours is None
            for seg in a["segmentation"]:
                if len(seg) >= 6:
                    p = np.array(seg, dtype=np.float32).reshape(-1, 2)
                    named[name].append(p)
                    if ours:
                        polys[ours].append(p)
        frames.append({"id": img["id"], "file": "train/" + img["file_name"], "group": gid[group_key(img["file_name"])],
                       "polygons": dict(polys), "other": other, "named": dict(named)})
    k = Counter()
    for f in frames:  # neutral names without dates: <group>_<running number>.jpg
        k[f["group"]] += 1
        f["name"] = f"{f['group']}_{k[f['group']]:04d}"
        f["path"] = FRAMES / f"{f['name']}.jpg"
    return frames


def extract(frames):
    """Write the frames' JPEG bytes unchanged under their neutral names (no EXIF in the export)."""
    FRAMES.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(ZIP)
    n = 0
    for f in frames:
        if not f["path"].exists():
            f["path"].write_bytes(z.read(f["file"]))
            n += 1
    return n


def masks(frame):
    """Ground-truth masks {our class: HxW bool} of one frame (absent classes are not keys)."""
    return _fill(frame["polygons"].items())


def named_masks(frame, classes):
    """Ground-truth masks of dataset classes by their dataset name (e.g. omentum, gallbladder)."""
    return _fill((c, frame["named"][c]) for c in classes if c in frame["named"])


def _fill(items):
    out = {}
    for cls, polys in items:
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [np.round(p).astype(np.int32) for p in polys], 1)
        if m.any():
            out[cls] = m.astype(bool)
    return out


def image(frame, z=None):
    z = z or zipfile.ZipFile(ZIP)
    return cv2.imdecode(np.frombuffer(z.read(frame["file"]), np.uint8), cv2.IMREAD_COLOR)


def splits():
    return {r["group"]: r["split"] for r in csv.DictReader(open(SPLIT))}


def make_split(frames, seed=0, n_val=3, n_test=3):
    """Seeded random choice of val and test groups; redrawn until every one of our six classes
    appears in at least 20 frames of val and of test, so both can score every class."""
    groups = sorted({f["group"] for f in frames})
    per = defaultdict(Counter)
    for f in frames:
        for cls in f["polygons"]:
            per[f["group"]][cls] += 1
    rng = random.Random(seed)
    for _ in range(10000):
        g = groups[:]
        rng.shuffle(g)
        val, test = sorted(g[:n_val]), sorted(g[n_val:n_val + n_test])
        ok = all(sum(per[x][c] for x in part) >= 20 for part in (val, test) for c in OURS)
        if ok:
            return {x: ("val" if x in val else "test" if x in test else "train") for x in groups}
    raise RuntimeError("no split satisfies the class rule")


def main():
    frames = load()
    print(f"extracted {extract(frames)} new frames to {FRAMES.relative_to(ROOT)}")
    split = make_split(frames)
    count = Counter(f["group"] for f in frames)
    SPLIT.parent.mkdir(exist_ok=True)
    with open(SPLIT, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["group", "split", "images"])
        for g in sorted(split):
            wr.writerow([g, split[g], count[g]])
    by = defaultdict(Counter)
    for f in frames:
        s = split[f["group"]]
        by[s]["frames"] += 1
        for cls in f["polygons"]:
            by[s][cls] += 1
    print(f"{len(frames)} frames in {len(split)} groups")
    for s in ("train", "val", "test"):
        gs = [g for g in sorted(split) if split[g] == s]
        print(f"{s:5s} {len(gs):2d} groups {by[s]['frames']:5d} frames  " + "  ".join(f"{c} {by[s][c]}" for c in OURS))
    print(f"wrote {SPLIT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
