# Open medical datasets and baselines on EVREN

Scripts that built the public medical datasets and evaluated the open baseline models published by
Şahin Parlak on [EVREN](https://evren.ssyz.org.tr), Türkiye's national AI platform. The public
collection: [Sağlık: Açık Veri Setleri ve Baseline Modeller](https://evren.ssyz.org.tr/collections/01a1b752-1527-7231-8d73-3e4174e74810).

Every dataset was taken from its original source with its license read there, split by patient,
operation or near-duplicate group, and documented on its EVREN page ("Kaynak ve türetim" lists
the source, license, changes and split). Every published model has a full model card on EVREN.
**None of the models is a clinical tool.**

This repository holds code only: no images, signals, masks or model weights.

## Datasets, models and scripts

EVREN pages are under `https://evren.ssyz.org.tr/datasets/u97ff17445ca0/<slug>` and
`https://evren.ssyz.org.tr/models/u97ff17445ca0/<run>`.

| Dataset (EVREN slug) | Build script | Published model | Evaluation |
|---|---|---|---|
| `vip20k-laparoscopic-pancreatic-surgery-vein-segmentation` | `vip20k_public_v1.py` | `vip20k-yolo26m-seg-1280-e60-seed1` | `evaluate_vip20k.py`, `external_pv_vip20k.py` |
| `lapgc-kvad-30-gastric-cancer-key-vessel-segmentation` | `lapgc_to_yolo_seg.py`, `lapgc_public_v1.py` | `lapgc-public-yolo26m-1280-e60-seed1` | `evaluate_vessels.py` |
| `laparoscopic-gastrectomy-organ-and-instrument-segmentation` | `gastrectomy_public_v1.py` (`gastro_frames.py`) | `gastrectomy-yolo26m-seg-1280-e60` | `evaluate_gastrectomy_public.py` |
| `hyperkvasir-colonoscopy-polyp-segmentation` | `hyperkvasir_polyp_public_v1.py` | `hyperkvasir-yolo26m-seg-640-e100` | `evaluate_hyperkvasir.py` |
| `hyperkvasir-gi-endoscopy-23-class-classification` | `hyperkvasir_labeled_near_duplicates.py`, `hyperkvasir_labeled_public_v1.py` | — | — |
| `bm-broncholc-bronchoscopy-landmark-and-lesion-segmentation` | `bronchololc_public_v1.py` | — | — |
| `hc18-fetal-head-circumference-ultrasound-segmentation` | `hc18_public_v1.py` | `hc18-yolo26m-seg-640-e100` | `evaluate_hc18.py` |
| `focus-fetal-four-chamber-ultrasound-cardiac-and-thoracic-segmentation` | `focus_public_v1.py` | `focus-yolo26m-seg-640-e300-nooverlap` | `evaluate_focus.py` |
| `psfhs-intrapartum-ultrasound-pubic-symphysis-and-fetal-head-segmentation` | `psfhs_public_v1.py` | `psfhs-yolo26m-seg-512-e100-seed1` | `evaluate_psfhs.py` |
| `pfus1-pelvic-floor-ultrasound-segmentation` | `pfus1_public_v1.py` | `pfus1-yolo26m-1024-e60-seed1` | `evaluate_pfus1.py` |
| `ecg-images-of-cardiac-patients-12-lead-classification` (and `...-square-1024`) | `ecg_images_public_v1.py` | `ecg-yolo26m-cls-1024-e100-seed1` | `evaluate_ecg.py` |
| `ptb-xl-12-lead-ecg-superclass-classification-rendered` | `ptbxl_public_v1.py` | — | — |
| `chapman-shaoxing-12-lead-ecg-rhythm-classification-rendered` | `chapman_public_v1.py` (draws with `ptbxl_public_v1.py`) | — | — |
| `arcade-coronary-artery-segments-syntax`, `arcade-coronary-stenosis-segmentation` | `arcade_public_v1.py` | `arcade-syntax-yolo26m-512-e100-seed1`, `arcade-stenosis-yolo26m-512-e100` | `evaluate_arcade.py` |
| `modus-urine-sediment-detection-40x`, `...-10x` | `modus_public_v1.py` | `modus40x-yolo26m-det-1280-e100` | `evaluate_modus.py` |
| `visem-tracking-sperm-detection` | `visem_public_v1.py` | `visem-yolo26m-det-640-e30-seed1` | `evaluate_modus.py` |

Not in this repository: `grazpedwri-dx-wrist-fracture-detection` (built in a separate project) and
`dsad-anatomy-v6` (its build depends on intermediate models that are not public).

Shared modules: `dsad_to_yolo_seg.py` (mask-to-polygon conversion), `evaluate_segmentation.py`
(prediction with time-out handling, Dice), `dsad_frames.py`, `inspect_checkpoint.py` (best epoch and
training settings of a checkpoint), `download_parallel.py` (resumable parallel range downloads).

Files used by the builds: `annotations/` (reviewed near-duplicate pairs of the two HyperKvasir
sets, the 66 duplicate-signal records removed from Chapman-Shaoxing), `splits/` (surgery splits of
DSAD and the gastrectomy set), `results/dsad_label_audit.csv` (DSAD inter-annotator Dice).

## Use

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -I scripts/<script>.py --help
```

Each script's docstring states its input folder under `data/`, its output and how to run it. Download
the source data from the original source named on the dataset's EVREN page and place it where the
docstring expects it. Run one prediction job at a time.

Docstrings refer to the author's working notes (`docs/...`); those notes are not part of this
repository. The same facts are on the EVREN dataset and model pages.

## Licenses

- Code: MIT (see `LICENSE`).
- Data: each dataset keeps the license of its original source (CC BY 4.0 or CC0 for the sets
  above); cite the original authors as given on the EVREN pages.
- Models trained with Ultralytics YOLO are AGPL-3.0.

## Not for clinical use

These datasets and models are for research and education. They are not validated for diagnosis,
treatment or intraoperative guidance.
