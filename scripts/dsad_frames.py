"""DSAD single-organ folders: weak labels and frame grouping.

In a single-organ folder (<organ>/<surgery>/imageNN.png) only that organ is segmented
(merged/maskNN.png); weak_labels.csv lists which structures are visible in each frame. The
same frame can sit in several organ folders, so frames are grouped by pixel content and every
expert mask of a frame is kept. The stomach folder is skipped: its frames are exactly the
multilabel frames (docs/findings-dsad.md).

Shared by scripts/build_anatomy_v2.py and scripts/evaluate_segmentation.py.
"""

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DSAD = ROOT / "data" / "dsad"

# Column order of weak_labels.csv, inferred from the always-present column of each folder.
WEAK_COLUMNS = ["abdominal_wall", "colon", "inferior_mesenteric_artery", "intestinal_veins",
                "liver", "pancreas", "small_intestine", "spleen", "stomach", "ureter", None,
                "vesicular_glands"]


def load_weak(folder):
    weak = {}
    path = folder / "weak_labels.csv"
    if path.exists():
        for row in csv.reader(open(path)):
            # weak_labels.csv names frames "images000.png", files are "image00.png": match by number
            idx = int(row[0].replace("images", "").replace(".png", ""))
            weak[idx] = {WEAK_COLUMNS[i] for i, v in enumerate(row[1:13]) if v == "1" and WEAK_COLUMNS[i]}
    return weak


def collect_frames(surgeries, require_weak=True):
    """Group single-organ frames of the given surgeries by pixel content: {(surgery, md5): frame}.
    A frame has its expert masks per organ, the structures visible by weak labels, and the
    visible structures that no expert mask covers ("missing").
    DSAD ships no weak labels for spleen/14 (val). Such frames raise unless require_weak is
    False; then they get weak_known=False and can only be scored on their expert masks."""
    frames = {}
    for organ_dir in sorted(p for p in DSAD.iterdir() if p.is_dir() and p.name not in ("multilabel", "stomach")):
        organ = organ_dir.name
        for surgery_dir in sorted(organ_dir.iterdir()):
            if not surgery_dir.is_dir() or surgery_dir.name not in surgeries:
                continue
            weak = load_weak(surgery_dir)
            for image in sorted(surgery_dir.glob("image*.png")):
                idx = image.stem.replace("image", "")
                key = (surgery_dir.name, hashlib.md5(image.read_bytes()).hexdigest())
                f = frames.setdefault(key, {"surgery": surgery_dir.name, "image": image, "experts": {}, "visible": set(),
                                            "weak_known": False, "name": f"s{surgery_dir.name}_{organ}_{idx}"})
                f["experts"][organ] = surgery_dir / "merged" / f"mask{idx}.png"
                if int(idx) in weak:
                    f["visible"] |= weak[int(idx)]
                    f["weak_known"] = True
                elif require_weak:
                    raise RuntimeError(f"no weak label for {image}")
    for f in frames.values():
        f["missing"] = f["visible"] - set(f["experts"])
    return frames
