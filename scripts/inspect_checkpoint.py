"""Check an EVREN checkpoint before evaluating it.

Prints whether it is the final file (at the end of training Ultralytics strips the optimizer
and EMA and sets epoch to -1), its class names, the train_args that differ from a reference
checkpoint (normally the run it was derived from), the best epoch (highest fitness =
0.1 mAP50 + 0.9 mAP50-95, box + mask, box only for detection; classification: top-1 + top-5; since the issue F fix
the EVREN model page shows this epoch) and a sparse training curve. Losses are sums of all
four terms from the checkpoint; EVREN's UI val_loss omits the seg term and is not comparable
with its train loss. A best epoch inside the close_mosaic phase explains mosaic, mixup and
copy_paste = 0.0 in train_args (issue G).

Run from the project root with the project venv:
    .venv/bin/python scripts/inspect_checkpoint.py models/<run>/best.pt [models/<reference>/best.pt]
"""

import hashlib
import sys
from pathlib import Path

import torch

SETTINGS = ("model", "epochs", "patience", "batch", "imgsz", "optimizer", "lr0", "lrf", "cos_lr", "seed",
            "close_mosaic", "mosaic", "mixup", "dropout", "label_smoothing")


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def main(path, ref_path=None):
    ck = load(path)
    ta = ck.get("train_args", {})
    print(f"{path}  md5 {hashlib.md5(Path(path).read_bytes()).hexdigest()}  saved {ck.get('date')}  "
          f"Ultralytics {ck.get('version')}")
    print(f"final file: {ck.get('epoch') == -1 and ck.get('optimizer') is None} "
          f"(epoch {ck.get('epoch')}, optimizer {'none' if ck.get('optimizer') is None else 'present'})")
    print("classes:", ck["model"].names)
    print("settings:", ", ".join(f"{k}={ta.get(k)!r}" for k in SETTINGS))
    if ref_path:
        ref = load(ref_path)
        rta = ref.get("train_args", {})
        print(f"\nclass names equal to the reference: {ck['model'].names == ref['model'].names}")
        print(f"train_args differing from {ref_path} (new vs reference; data and project are job paths):")
        for k in sorted(set(ta) | set(rta)):
            if ta.get(k) != rta.get(k):
                print(f"  {k}: {ta.get(k)!r} vs {rta.get(k)!r}")

    r = ck.get("train_results")
    if not r:
        return

    def col(key, i):
        return float(str(r[key][i]).strip())

    ep = [int(col("epoch", i)) for i in range(len(r["epoch"]))]
    if "metrics/accuracy_top1" in r:  # classification: fitness = (top-1 + top-5) / 2
        fit = [col("metrics/accuracy_top1", i) + col("metrics/accuracy_top5", i) for i in range(len(ep))]
        # Ultralytics re-saves best.pt on ties (fitness == best), so the file holds the last best epoch
        best = max(range(len(ep)), key=lambda i: (fit[i], i))
        print(f"\nepochs rows {ep[-1]} of {ta.get('epochs')}; best (fitness, last of ties) E{ep[best]}")
        print(f"{'ep':>4}{'top-1':>8}{'train L':>9}{'val L':>7}{'lr':>10}")
        for i in sorted(set(range(9, len(ep), 10)) | {best, len(ep) - 1}):
            print(f"{ep[i]:>4}{col('metrics/accuracy_top1', i):>8.3f}{col('train/loss', i):>9.3f}"
                  f"{col('val/loss', i):>7.3f}{col('lr/pg0', i):>10.2e}{'  <- best' if i == best else ''}")
        return

    seg = "metrics/mAP50(M)" in r  # detection checkpoints have box metrics and no seg loss

    def loss(part, i):
        return sum(col(f"{part}/{t}_loss", i) for t in ("box", "seg", "cls", "dfl") if f"{part}/{t}_loss" in r)

    def mask(key, i):
        return col(key, i) if seg else float("nan")

    fit = [0.1 * col("metrics/mAP50(B)", i) + 0.9 * col("metrics/mAP50-95(B)", i)
           + (0.1 * col("metrics/mAP50(M)", i) + 0.9 * col("metrics/mAP50-95(M)", i) if seg else 0)
           for i in range(len(ep))]
    best = max(range(len(ep)), key=lambda i: fit[i])
    top50 = max(range(len(ep)), key=lambda i: col("metrics/mAP50(B)", i))
    print(f"\nepochs run {ep[-1]} of {ta.get('epochs')}; best (fitness) E{ep[best]}; highest box mAP50 "
          f"E{ep[top50]} ({col('metrics/mAP50(B)', top50):.3f}; the epoch the EVREN page showed before the "
          f"issue F fix of 2026-09-29)")
    if ta.get("close_mosaic") and ep[best] > ta.get("epochs", 0) - ta["close_mosaic"]:
        print(f"note: best epoch is in the last {ta['close_mosaic']} (close_mosaic): Ultralytics then overwrites "
              "mosaic, mixup and copy_paste with 0.0 in train_args; training used the dialog values before that "
              "(EVREN answer to issue G)")
    print(f"{'ep':>4}{'mAP50 B':>9}{'5095 B':>8}{'mAP50 M':>9}{'5095 M':>8}{'train L':>9}{'val L':>7}{'lr':>10}")
    for i in sorted(set(range(9, len(ep), 10)) | {best, len(ep) - 1}):
        print(f"{ep[i]:>4}{col('metrics/mAP50(B)', i):>9.3f}{col('metrics/mAP50-95(B)', i):>8.3f}"
              f"{mask('metrics/mAP50(M)', i):>9.3f}{mask('metrics/mAP50-95(M)', i):>8.3f}{loss('train', i):>9.2f}"
              f"{loss('val', i):>7.2f}{col('lr/pg0', i):>10.2e}{'  <- best' if i == best else ''}")

if __name__ == "__main__":
    main(*sys.argv[1:3])
