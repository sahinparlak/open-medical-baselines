"""Find near-duplicate candidates among the HyperKvasir labelled images (docs/datasets.md).

The source has no patient or exam ids, so frames of the same exam can only be linked by image
similarity. Each image is embedded with frozen DINOv2 (dinov2_vitb14_reg, Apache 2.0): short side
224 px, centre crop 224 x 224, ImageNet normalization, normalized CLS token. Pairs with cosine
similarity >= 0.95 are written out, across classes too (one exam can hold a polyp and its dyed
lifting); scripts/hyperkvasir_labeled_public_v1.py joins them into groups.

Calibration (2026-11-09): side-by-side review of 24 random pairs at 0.95-0.97 found 10 clearly the
same exam and 14 uncertain oesophageal views (grouped to be safe); of 24 at 0.93-0.95, 1-3 looked
like the same exam (left ungrouped: 3,774 such pairs). The stool classes (bbps-0-1, bbps-2-3,
impacted-stool) are left out: 84-97 % of their images have a neighbour above 0.95 because
stool-covered views look alike across patients, so similarity cannot separate exams there.

Run from the project root with the project venv:
    .venv/bin/python -I scripts/hyperkvasir_labeled_near_duplicates.py [--device mps]
Writes data/embeddings/hyperkvasir_labeled_dinov2_vitb14_reg.npz and
annotations/hyperkvasir_labeled/near_duplicate_pairs.csv
"""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "hyperkvasir_labeled" / "extracted" / "labeled-images"
EMB = ROOT / "data" / "embeddings" / "hyperkvasir_labeled_dinov2_vitb14_reg.npz"
PAIRS = ROOT / "annotations" / "hyperkvasir_labeled" / "near_duplicate_pairs.csv"
MIN_SIM = 0.95
STOOL = {"bbps-0-1", "bbps-2-3", "impacted-stool"}


def embed(device):
    rows = list(csv.DictReader(open(SRC / "image-labels.csv", encoding="utf-8")))
    paths = {p.stem: p for p in SRC.rglob("*.jpg")}
    names = sorted(r["Video file"] for r in rows)
    finding = {r["Video file"]: r["Finding"] for r in rows}
    tf = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg", trust_repo=True)
    model.eval().to(device)
    out = []
    with torch.inference_mode():
        for i in range(0, len(names), 32):
            batch = torch.stack([tf(Image.open(paths[n]).convert("RGB")) for n in names[i:i + 32]])
            out.append(model.forward_features(batch.to(device))["x_norm_clstoken"].float().cpu().numpy())
            print(f"\r{min(i + 32, len(names))}/{len(names)} images", end="", flush=True)
    print()
    EMB.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EMB, name=np.array(names), finding=np.array([finding[n] for n in names]),
                        cls=np.concatenate(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    if not EMB.exists():
        embed(args.device)
    d = np.load(EMB)
    names, finding = d["name"], d["finding"]
    x = d["cls"] / np.linalg.norm(d["cls"], axis=1, keepdims=True)
    sim = x @ x.T
    stool = np.isin(finding, list(STOOL))
    sim[stool, :] = 0
    sim[:, stool] = 0
    i, j = np.where(np.triu(sim, 1) >= MIN_SIM)
    order = np.argsort(-sim[i, j])
    PAIRS.parent.mkdir(parents=True, exist_ok=True)
    with open(PAIRS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "finding_a", "finding_b", "sim"])
        for k in order:
            a, b = i[k], j[k]
            w.writerow([names[a], names[b], finding[a], finding[b], f"{sim[a, b]:.4f}"])
    print(f"{len(order)} pairs with similarity >= {MIN_SIM} -> {PAIRS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
