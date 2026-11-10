"""Build a public EVREN version of VIP20K (Kaggle prohuge/vip20k v5, CC BY 4.0; docs/datasets.md,
VIP20K entry; plan docs/plan-vip20k-public-v1.md).

Source: 23 laparoscopic pancreatic operations (15 LDP, 8 LPD), frames 1920 x 1080 JPEG at 1 fps,
binary masks (0/1 PNG, `<frame>_mask.png`) of the SMV, PV and SV merged into one class.

Changes made here:
- Frames without a mask (131: 130 in LPD Case 04 segment 10, pic-09050 in LPD Case 02 segment 02),
  the one mask without a frame (LPD Case 02 segment 02 pic-00000_mask.png; not paired by guess)
  and the 177 "no object" frames (no case id) are left out: 18,872 frames remain.
- Images copied byte for byte, renamed `<LDP|LPD>-<case>_s<segment>_<frame>.jpg` (frame names
  repeat across operations).
- Masks → YOLO polygons (one class, vein; outer contour per part, holes joined by keyholes, specks
  < 25 px dropped, as in dsad_to_yolo_seg.py), checked by drawing them again (Dice against the mask).
- Split by operation (plan decision 2): val LDP 06, LDP 10, LPD 07, LPD 08; test LDP 01, LDP 12,
  LPD 05, LPD 06; train the other 15.

The build stops if a check fails: 23 operations, 18,873 masks with exactly the one orphan above,
every mask 0/1 and the size of its
image, no byte-identical image in two splits, median polygon Dice >= 0.97.

Run from the project root with the project venv:
    .venv/bin/python -I scripts/vip20k_public_v1.py
Writes data/vip20k_public/v1/ (images, labels, data.yaml, build_report.json) and
data/upload/vip20k_public_v1_yoloseg.zip
"""

import hashlib
import json
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from dsad_to_yolo_seg import mask_to_polygons  # noqa: E402

SRC = ROOT / "data" / "vip20k" / "extracted" / "VIP20K" / "VIP20K"
OUT = ROOT / "data" / "vip20k_public" / "v1"
ZIP = ROOT / "data" / "upload" / "vip20k_public_v1_yoloseg.zip"
VAL = {"LDP-06", "LDP-10", "LPD-07", "LPD-08"}
TEST = {"LDP-01", "LDP-12", "LPD-05", "LPD-06"}


def case_id(mask_path):
    proc = mask_path.parts[-5].split("-")[0]           # LDP-PUMCH -> LDP
    return f"{proc}-{mask_path.parts[-4].split()[-1]}"  # Case 01 -> 01


def convert(mask_path):
    case = case_id(mask_path)
    split = "val" if case in VAL else "test" if case in TEST else "train"
    seg = mask_path.parts[-3].replace("segment_", "s")
    stem = mask_path.name[: -len("_mask.png")]
    image = mask_path.parent.parent / "Image" / f"{stem}.jpg"
    name = f"{case}_{seg}_{stem}"
    raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    img = cv2.imread(str(image))
    values = tuple(int(v) for v in np.unique(raw))
    if raw.ndim != 2 or not set(values) <= {0, 1} or raw.shape != img.shape[:2]:
        return {"name": name, "error": f"mask values {values[:5]} shape {raw.shape} vs {img.shape}"}
    m = raw == 1
    h, w = m.shape
    polys = mask_to_polygons(m) if m.any() else []
    redraw = np.zeros((h, w), np.uint8)
    lines = []
    for p in polys:
        cv2.fillPoly(redraw, [p.astype(np.int32)], 1)
        lines.append("0 " + " ".join(f"{v:.6f}" for v in (p / np.array([w, h])).ravel()))
    dice = None
    if m.any():
        r = redraw.astype(bool)
        dice = float(2 * (r & m).sum() / (r.sum() + m.sum()))
    shutil.copyfile(image, OUT / "images" / split / f"{name}.jpg")
    (OUT / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return {"name": name, "case": case, "split": split, "polygons": len(polys), "dice": dice,
            "area": float(m.mean()), "md5": hashlib.md5(image.read_bytes()).hexdigest()}


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; remove it first so no stale files enter the ZIP")
    all_masks = sorted(SRC.glob("*-PUMCH/Case */segment_*/Mask/*_mask.png"))
    assert len(all_masks) == 18873, len(all_masks)
    masks = [m for m in all_masks if (m.parent.parent / "Image" / f"{m.name[:-9]}.jpg").exists()]
    orphans = [str(m.relative_to(SRC)) for m in all_masks if m not in set(masks)]
    assert orphans == ["LPD-PUMCH/Case 02/segment_02/Mask/pic-00000_mask.png"], orphans
    cases = {case_id(p) for p in masks}
    assert len(cases) == 23 and VAL | TEST <= cases and not VAL & TEST, sorted(cases)
    for part in ("images", "labels"):
        for s in ("train", "val", "test"):
            (OUT / part / s).mkdir(parents=True)

    with ProcessPoolExecutor(8) as pool:
        rows = list(pool.map(convert, masks, chunksize=32))
    errors = [r for r in rows if "error" in r]
    assert not errors, errors[:5]

    by_md5 = defaultdict(set)
    for r in rows:
        by_md5[r["md5"]].add(r["split"])
    dup_groups = [s for s in by_md5.values() if len(s) > 1]
    assert not dup_groups, f"{len(dup_groups)} byte-identical images across splits"
    dice = [r["dice"] for r in rows if r["dice"] is not None]
    fid = {"median": round(float(np.median(dice)), 4), "p01": round(float(np.quantile(dice, 0.01)), 4),
           "min": round(float(np.min(dice)), 4)}
    assert fid["median"] >= 0.97, fid

    (OUT / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: vein\n")
    split_frames = Counter(r["split"] for r in rows)
    report = {
        "frames": len(rows), "operations": len(cases),
        "split_frames": dict(split_frames),
        "split_operations": {s: sorted({r["case"] for r in rows if r["split"] == s})
                             for s in ("train", "val", "test")},
        "empty_masks": sum(r["dice"] is None for r in rows),
        "polygons": sum(r["polygons"] for r in rows),
        "polygon_dice": fid,
        "byte_identical_extra_copies": len(rows) - len(by_md5),
        "vein_area_fraction_median": round(float(np.median([r["area"] for r in rows])), 4),
        "frames_per_operation": dict(sorted(Counter(r["case"] for r in rows).items())),
    }
    (OUT / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")

    ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(OUT.rglob("*")):
            if f.is_file() and f.name != "build_report.json":
                z.write(f, f.relative_to(OUT))
    print(json.dumps(report, indent=1))
    print(f"wrote {ZIP.relative_to(ROOT)} ({ZIP.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
