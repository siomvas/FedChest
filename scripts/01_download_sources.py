#!/usr/bin/env python3
"""
Script 01: Download source datasets.

This script provides instructions and utilities for downloading the source
datasets required to build FedChest.

NOTE: All three datasets require registration/approval before download.
This script cannot automatically download them.

Required datasets:
1. NIH ChestX-ray14 (CXR14)
   - URL: https://nihcc.app.box.com/v/ChestXray-NIHCC
   - Size: ~45 GB
   - Files needed:
     - images/ (all PNG files)
     - Data_Entry_2017_v2020.csv
     - train_val_list.txt
     - test_list.txt

2. CheXpert
   - URL: https://stanfordmlgroup.github.io/competitions/chexpert/
   - Size: ~11 GB (small version) or ~439 GB (full)
   - Requires Stanford AIMI registration
   - Files needed:
     - CheXpert-v1.0/train/ (all images)
     - train.csv

3. PadChest
   - URL: https://bimcv.cipf.es/bimcv-projects/padchest/
   - Size: ~1 TB (full) or subset
   - Requires BIMCV registration
   - Files needed:
     - images/ (PNG files)
     - PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv

Usage:
    python scripts/01_download_sources.py --check --data-dir /path/to/data

After downloading, your directory structure should look like:

    /path/to/data/
    ├── cxr14/
    │   ├── images/
    │   │   ├── 00000001_000.png
    │   │   └── ...
    │   ├── Data_Entry_2017_v2020.csv
    │   ├── train_val_list.txt
    │   └── test_list.txt
    ├── chexpert/
    │   ├── CheXpert-v1.0/
    │   │   └── train/
    │   │       ├── patient00001/
    │   │       └── ...
    │   └── train.csv
    └── padchest/
        ├── images/
        │   ├── 216840111366964012373310883942009152114636712_00-102-146.png
        │   └── ...
        └── PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Expected files for each dataset
EXPECTED_FILES: Dict[str, List[Tuple[str, bool]]] = {
    "cxr14": [
        ("Data_Entry_2017_v2020.csv", True),
        ("train_val_list.txt", False),  # Optional, can reconstruct
        ("test_list.txt", False),
        ("images", True),  # Directory
    ],
    "chexpert": [
        ("train.csv", True),
        ("CheXpert-v1.0/train", True),  # Directory
    ],
    "padchest": [
        ("PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv", True),
        ("images", True),  # Directory
    ],
}

# Alternative filenames that are also acceptable
ALTERNATIVE_FILES: Dict[str, Dict[str, str]] = {
    "padchest": {
        "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv": "padchest_labels.csv",
    },
}


def check_dataset(data_dir: Path, dataset_name: str) -> Tuple[bool, List[str]]:
    """
    Check if a dataset is properly downloaded.
    
    Args:
        data_dir: Root data directory
        dataset_name: Name of dataset (cxr14, chexpert, padchest)
    
    Returns:
        Tuple of (all_required_present, list_of_missing_files)
    """
    dataset_dir = data_dir / dataset_name
    missing = []
    
    if not dataset_dir.exists():
        return False, [f"{dataset_name}/ directory not found"]
    
    expected = EXPECTED_FILES.get(dataset_name, [])
    alternatives = ALTERNATIVE_FILES.get(dataset_name, {})
    
    for filename, required in expected:
        filepath = dataset_dir / filename
        
        # Check for alternative filenames
        alt_filename = alternatives.get(filename)
        alt_filepath = dataset_dir / alt_filename if alt_filename else None
        
        exists = filepath.exists() or (alt_filepath and alt_filepath.exists())
        
        if not exists and required:
            missing.append(filename)
    
    return len(missing) == 0, missing


def check_all_datasets(data_dir: Path) -> Dict[str, Tuple[bool, List[str]]]:
    """
    Check all required datasets.
    
    Args:
        data_dir: Root data directory
    
    Returns:
        Dict mapping dataset name to (ready, missing_files)
    """
    results = {}
    
    for dataset_name in EXPECTED_FILES.keys():
        ready, missing = check_dataset(data_dir, dataset_name)
        results[dataset_name] = (ready, missing)
    
    return results


def print_download_instructions():
    """Print instructions for downloading each dataset."""
    
    instructions = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                     FedChest Dataset Download Instructions                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

All three datasets require registration before download. Please follow the
instructions below for each dataset.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. NIH ChestX-ray14 (CXR14)
   ─────────────────────────
   URL: https://nihcc.app.box.com/v/ChestXray-NIHCC
   
   Steps:
   a) Create a Box account (if you don't have one)
   b) Download all files from the images/ folder
   c) Download Data_Entry_2017_v2020.csv
   d) Download train_val_list.txt and test_list.txt
   
   Place files in: <data_dir>/cxr14/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2. CheXpert
   ─────────
   URL: https://stanfordmlgroup.github.io/competitions/chexpert/
   
   Steps:
   a) Register at Stanford AIMI: https://aimi.stanford.edu/chexpert-chest-x-rays
   b) Accept the data use agreement
   c) Download CheXpert-v1.0-small.zip (or full version)
   d) Extract the archive
   
   Place files in: <data_dir>/chexpert/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3. PadChest
   ─────────
   URL: https://bimcv.cipf.es/bimcv-projects/padchest/
   
   Steps:
   a) Register at BIMCV
   b) Request access to the PadChest dataset
   c) Download images and labels CSV
   
   Place files in: <data_dir>/padchest/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

After downloading, run this script with --check to verify:

    python scripts/01_download_sources.py --check --data-dir /path/to/data

"""
    print(instructions)


def main():
    parser = argparse.ArgumentParser(
        description="Download and verify source datasets for FedChest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if datasets are properly downloaded",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data/raw"),
        help="Root directory for raw datasets",
    )
    parser.add_argument(
        "--instructions",
        action="store_true",
        help="Print download instructions",
    )
    args = parser.parse_args()
    
    if args.instructions or not args.check:
        print_download_instructions()
    
    if args.check:
        logger.info(f"Checking datasets in: {args.data_dir}")
        logger.info("=" * 60)
        
        results = check_all_datasets(args.data_dir)
        all_ready = True
        
        for dataset_name, (ready, missing) in results.items():
            status = "✓ Ready" if ready else "✗ Missing files"
            logger.info(f"{dataset_name}: {status}")
            
            if missing:
                for filename in missing:
                    logger.info(f"  - Missing: {filename}")
                all_ready = False
        
        logger.info("=" * 60)
        
        if all_ready:
            logger.info("✓ All datasets are ready for preprocessing!")
            logger.info("Run: python scripts/02_preprocess_all.py --data-dir " + str(args.data_dir))
        else:
            logger.warning("Some datasets are missing. See download instructions above.")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
