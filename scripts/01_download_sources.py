#!/usr/bin/env python3
"""
Script 01: Download and Verify Source Datasets.

This script checks for the presence of the three required source datasets.
Since these datasets can be large and stored in different locations on an HPC 
or local machine, this script allows you to specify the path to each dataset 
individually.

NOTE: All three datasets require registration/approval before download.

Required datasets and expected structures:

1. NIH ChestX-ray14 (CXR14)
   - Argument: --cxr14-dir /path/to/cxr14
   - Expected contents inside directory:
     - images/ (folder containing PNGs)
     - Data_Entry_2017_v2020.csv

2. CheXpert
   - Argument: --chexpert-dir /path/to/CheXpert-v1.0
   - Expected contents inside directory:
     - train/ (folder containing patient subfolders)
     - train.csv

3. PadChest
   - Argument: --padchest-dir /path/to/padchest
   - Expected contents inside directory:
     - images/ (folder containing PNGs)
     - PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv

Usage:
    python scripts/01_download_sources.py \
        --check \
        --cxr14-dir /data/public/cxr14 \
        --chexpert-dir /path/to/archive/CheXpert-v1.0 \
        --padchest-dir /scratch/datasets/padchest
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Expected files relative to the specific dataset directory provided by the user
EXPECTED_FILES: Dict[str, List[Tuple[str, bool]]] = {
    "cxr14": [
        ("Data_Entry_2017_v2020.csv", True),
        ("train_val_list.txt", False),  # Optional
        ("test_list.txt", False),       # Optional
        ("images", True),               # Directory
    ],
    "chexpert": [
        ("train.csv", True),
        ("train", True),                # Directory (contains patient folders)
        ("valid.csv", False),           # Optional
        ("valid", False),               # Optional
    ],
    "padchest": [
        ("PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv", True),
        ("images", True),               # Directory
    ],
}

# Alternative filenames that are also acceptable
ALTERNATIVE_FILES: Dict[str, Dict[str, str]] = {
    "padchest": {
        "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv": "padchest_labels.csv",
    },
}


def check_dataset(dataset_path: Path, dataset_name: str) -> Tuple[bool, List[str]]:
    """
    Check if a specific dataset is valid at the given path.
    
    Args:
        dataset_path: Path to the specific dataset root
        dataset_name: Name of dataset (cxr14, chexpert, padchest) for looking up rules
    
    Returns:
        Tuple of (all_required_present, list_of_missing_files)
    """
    missing = []
    
    if not dataset_path.exists():
        return False, [f"Directory not found: {dataset_path}"]
    
    expected = EXPECTED_FILES.get(dataset_name, [])
    alternatives = ALTERNATIVE_FILES.get(dataset_name, {})
    
    for filename, required in expected:
        filepath = dataset_path / filename
        
        # Check for alternative filenames
        alt_filename = alternatives.get(filename)
        alt_filepath = dataset_path / alt_filename if alt_filename else None
        
        exists = filepath.exists() or (alt_filepath and alt_filepath.exists())
        
        if not exists and required:
            missing.append(filename)
    
    return len(missing) == 0, missing


def print_download_instructions():
    """Print instructions for downloading each dataset."""
    
    instructions = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                     FedChest Dataset Download Instructions                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. NIH ChestX-ray14 (CXR14)
   URL: https://nihcc.app.box.com/v/ChestXray-NIHCC
   Required: images/ folder and Data_Entry_2017_v2020.csv

2. CheXpert
   URL: https://stanfordmlgroup.github.io/competitions/chexpert/
   Required: CheXpert-v1.0-small.zip (or full). Unzip it.
   Point the script to the unzipped folder (containing train/ and train.csv).

3. PadChest
   URL: https://bimcv.cipf.es/bimcv-projects/padchest/
   Required: images/ folder and labels CSV.

Verify your paths using the --check flag with specific directory arguments.
"""
    print(instructions)


def main():
    parser = argparse.ArgumentParser(
        description="Verify source datasets for FedChest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Arguments for each specific dataset path
    parser.add_argument(
        "--cxr14-dir",
        type=Path,
        help="Path to the NIH ChestX-ray14 dataset root",
    )
    parser.add_argument(
        "--chexpert-dir",
        type=Path,
        help="Path to the CheXpert dataset root (e.g., .../CheXpert-v1.0)",
    )
    parser.add_argument(
        "--padchest-dir",
        type=Path,
        help="Path to the PadChest dataset root",
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if datasets are properly downloaded at the specified paths",
    )
    parser.add_argument(
        "--instructions",
        action="store_true",
        help="Print download instructions",
    )
    
    args = parser.parse_args()
    
    # If no arguments provided, just show help
    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    if args.instructions:
        print_download_instructions()
        return 0
    
    if args.check:
        logger.info("Checking provided dataset paths...")
        logger.info("=" * 60)
        
        # Map the cli args to the internal names
        # We only check datasets that the user explicitly provided paths for
        datasets_to_check = {
            "cxr14": args.cxr14_dir,
            "chexpert": args.chexpert_dir,
            "padchest": args.padchest_dir
        }
        
        all_passed = True
        checked_any = False

        for name, path in datasets_to_check.items():
            if path is None:
                # User did not provide this path
                logger.warning(f"{name}: Path not provided (skipping check)")
                continue
            
            checked_any = True
            ready, missing = check_dataset(path, name)
            status = "✓ Ready" if ready else "✗ Missing files"
            
            # Print status with the path for clarity
            logger.info(f"{name} ({path}): {status}")
            
            if missing:
                all_passed = False
                for filename in missing:
                    logger.info(f"  - Missing: {filename}")
        
        logger.info("=" * 60)
        
        if not checked_any:
            logger.error("No dataset paths provided to check! Please use --cxr14-dir, --chexpert-dir, or --padchest-dir.")
            return 1

        if all_passed:
            logger.info("✓ All checked datasets are valid.")
            logger.info("You can now proceed to preprocessing using these specific paths.")
        else:
            logger.warning("Some datasets are missing required files.")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())