# FedChest

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/TMLR-2025-8a2be2.svg)](https://openreview.net/forum?id=GtdYFLsblb)

Code that builds FedChest, a federated benchmark for multi-label chest X-ray classification assembled from NIH ChestX-ray14, CheXpert and PadChest. FedChest was introduced in [An Architecture Built for Federated Learning: Addressing Data Heterogeneity through Adaptive Normalization-Free Feature Recalibration](https://openreview.net/forum?id=GtdYFLsblb) (TMLR, 2025).

The repository contains code only. The images and labels of the three source datasets are not redistributed here, so each of them has to be obtained from its provider under its own terms of use.

## Which FedChest is this?

Two builds of FedChest exist, and they are not interchangeable.

| | Paper build (February 2024) | This repository (rebuild of January 2026) |
|---|---|---|
| Used for | every FedChest result in the TMLR paper | later work |
| Images | 205,504 | 255,672 |
| Labels | 8: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, No Finding, Pneumonia, Pneumothorax | 10: the same eight, plus Lung Lesion and Lung Opacity |
| Splits | by patient, with 15,000 validation images per client and 10,000 test images (7,797 and 7,203 for the two CheXpert clients) | by patient, with 15% of the patients in validation and 10% in test |
| Code | [`fl_bench/chest/src/chest_preparation.py`](https://github.com/siomvas/ANFR/blob/main/fl_bench/chest/src/chest_preparation.py) in the ANFR repository | this repository |

The rebuild keeps almost every image of the paper build and adds about 52,000 more, most of which carry none of the eight original labels. The splits of the two builds were drawn independently, so most test images of the paper build sit in the training split of the rebuild. Results obtained on one build must therefore not be compared with, or mixed with, results obtained on the other. To reproduce the paper, use the script in the ANFR repository.

## Clients

CheXpert is about twice the size of the other two datasets after preprocessing, so it is divided into two clients at the median patient age, which is 62 years in this build and 63 in the paper build.

| Client | Source | Train | Validation | Test |
|---|---|---|---|---|
| `cxr14` | NIH ChestX-ray14 | 47,685 | 9,349 | 6,349 |
| `padchest` | PadChest | 45,136 | 8,958 | 6,087 |
| `cxp_young` | CheXpert, age below 62 | 49,468 | 8,658 | 6,490 |
| `cxp_old` | CheXpert, age 62 and above | 51,348 | 9,212 | 6,932 |

The counts are those of the run of January 2026 with seed 42.

## What the pipeline does

1. Keeps frontal views only (AP and PA).
2. Maps the labels of each source to the common set. CheXpert's uncertain and missing labels become negative. ChestX-ray14's Mass and Nodule become Lung Lesion, and its Infiltration and Fibrosis become Lung Opacity. PadChest's free-text labels are matched through the alias lists in `configs/preprocessing.yaml`.
3. Drops every image that carries none of the common labels.
4. Removes duplicates, meaning images of the same patient with the same view and the same common labels.
5. Skips the PadChest images known to be corrupted.
6. Crops 5% from each edge, resizes to 224 by 224 pixels and applies contrast-limited adaptive histogram equalization (clip limit 2.0, 8 by 8 tiles).
7. Splits by patient, so that no patient appears in two splits of a client. CheXpert patients with images on both sides of the age threshold are assigned to `cxp_young`.

## Building the dataset

Request access to the three sources first:

- NIH ChestX-ray14: https://nihcc.app.box.com/v/ChestXray-NIHCC
- CheXpert: https://stanfordmlgroup.github.io/competitions/chexpert/
- PadChest: https://bimcv.cipf.es/bimcv-projects/padchest/

```bash
git clone https://github.com/siomvas/FedChest.git
cd FedChest
pip install -e .

# 1. Check that the source folders have the expected layout
python scripts/01_download_sources.py --check \
    --cxr14-dir /path/to/CXR14 \
    --chexpert-dir /path/to/CheXpert-v1.0 \
    --padchest-dir /path/to/PadChest

# 2. Preprocess the images and write the split CSVs
python scripts/02_preprocess_all.py \
    --cxr14-dir /path/to/CXR14 \
    --chexpert-dir /path/to/CheXpert-v1.0 \
    --padchest-dir /path/to/PadChest \
    --output-dir ./data/processed

# 3. Verify the splits
python scripts/03_generate_splits.py --data-dir ./data/processed --verify

# 4. Write the datalists that the loaders read
python scripts/04_create_datalists.py \
    --data-root ./data/processed/images \
    --csv-dir ./data/processed/csv \
    --output-dir ./data/datalists \
    --config configs/splits.yaml

# 5. Validate the result
python scripts/05_validate_dataset.py \
    --data-root ./data/processed/images \
    --datalist-dir ./data/datalists \
    --config configs/splits.yaml
```

Every script documents its arguments under `--help`.

## Loading a client

```python
from fedchest import ChestDataModule

dm = ChestDataModule(data_dir="/path/to/fedchest", client_idx="client_cxr14", batch_size=64)
dm.setup()
```

`data_dir` is the folder that holds `datalists/`. The loaders need `monai`, which `pip install -e ".[training]"` adds.

## Status

The label set is still being revised. `configs/preprocessing.yaml` defines the ten labels above, while `configs/splits.yaml` and `scripts/05_validate_dataset.py` still carry earlier lists, so steps 4 and 5 need those lists aligned with the preprocessing configuration before they are run on its output.

## Repository layout

| Path | Contents |
|---|---|
| `configs/` | preprocessing parameters, label aliases and split settings |
| `scripts/` | the five numbered steps above |
| `src/fedchest/preprocessing/` | one preprocessor per source dataset and the shared image transforms |
| `src/fedchest/dataloaders/` | PyTorch data modules for federated and centralized training |
| `src/fedchest/evaluation/` | validation loop and metrics |
| `CXR14_pre_process.ipynb`, `PC_pre_process.ipynb` | the notebooks the pipeline grew out of |

## Citation

If you use FedChest, please cite the paper that introduced it:

```bibtex
@article{siomos2025anfr,
  title   = {An Architecture Built for Federated Learning: Addressing Data Heterogeneity through Adaptive Normalization-Free Feature Recalibration},
  author  = {Siomos, Vasilis and Passerat-Palmbach, Jonathan and Tarroni, Giacomo},
  journal = {Transactions on Machine Learning Research},
  year    = {2025},
  url     = {https://openreview.net/forum?id=GtdYFLsblb}
}
```

and the three source datasets:

```bibtex
@inproceedings{wang2017chestx,
  title     = {ChestX-ray8: Hospital-scale chest X-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases},
  author    = {Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and Bagheri, Mohammadhadi and Summers, Ronald M.},
  booktitle = {CVPR},
  year      = {2017}
}

@inproceedings{irvin2019chexpert,
  title     = {CheXpert: A large chest radiograph dataset with uncertainty labels and expert comparison},
  author    = {Irvin, Jeremy and Rajpurkar, Pranav and Ko, Michael and others},
  booktitle = {AAAI},
  year      = {2019}
}

@article{bustos2020padchest,
  title   = {PadChest: A large chest x-ray image dataset with multi-label annotated reports},
  author  = {Bustos, Aurelia and Pertusa, Antonio and Salinas, Jose-Maria and de la Iglesia-Vay{\'a}, Maria},
  journal = {Medical Image Analysis},
  year    = {2020}
}
```

## License

The code is released under the Apache 2.0 license (see `LICENSE`). The source datasets keep their own licenses and terms of use, which apply to anything built from them.

## Contact

Please open a GitHub issue.
