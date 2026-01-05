#!/usr/bin/env python3
"""
Script 02: Preprocess Source Datasets.

Runs the preprocessing pipeline for all source datasets.
This script expects the specific paths to each dataset to be provided via arguments,
matching the flexibility of the download script.

Preprocessing steps for each dataset:
1. Load and parse metadata/labels (mapping to 10 Federated Classes)
2. Filter to common diseases
3. Remove duplicates and invalid files
4. Crop borders (5% from each edge)
5. Resize to 224x224
6. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
7. Create patient-level train/val/test splits
8. Save processed images and CSV files

Usage:
    # Process all datasets (requires all paths):
    python scripts/02_preprocess_all.py \
        --cxr14-dir /path/to/cxr14 \
        --chexpert-dir /path/to/CheXpert-v1.0 \
        --padchest-dir /path/to/padchest \
        --output-dir ./data/processed

    # Process only specific datasets:
    python scripts/02_preprocess_all.py \
        --datasets cxr14 \
        --cxr14-dir /path/to/cxr14 \
        --output-dir ./data/processed
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fedchest.preprocessing import (
    CXR14Preprocessor,
    CheXpertPreprocessor,
    PadChestPreprocessor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("preprocessing.log"),
    ],
)
logger = logging.getLogger(__name__)


DATASET_PREPROCESSORS = {
    "cxr14": CXR14Preprocessor,
    "chexpert": CheXpertPreprocessor,
    "padchest": PadChestPreprocessor,
}

def preprocess_dataset(
    dataset_name: str,
    source_dir: Path,
    output_dir: Path,
    config_path: Optional[Path],
    stage: str,
    seed: int,
) -> bool:
    """
    Preprocess a single dataset using its specific source path.
    
    Args:
        dataset_name: Name of dataset (cxr14, chexpert, padchest)
        source_dir: The specific root directory for this dataset
        output_dir: Directory for processed output
        config_path: Path to preprocessing config
        seed: Random seed for splits
    
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {dataset_name.upper()}")
    logger.info(f"Source: {source_dir}")
    logger.info(f"{'='*60}")
    
    preprocessor_class = DATASET_PREPROCESSORS.get(dataset_name)
    if not preprocessor_class:
        logger.error(f"Unknown dataset: {dataset_name}")
        return False
    
    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return False
    
    try:
        # Initialize the specific preprocessor
        preprocessor = preprocessor_class(
            source_dir=source_dir,
            output_dir=output_dir,
            config_path=config_path,
        )
        
        # Run the pipeline
        results = preprocessor.run(seed=seed, stage=stage)
        
        logger.info(f"\n{dataset_name.upper()} preprocessing complete!")
        logger.info(f"Output files: {results}")
        
        return True
        
    except Exception as e:
        logger.exception(f"Error processing {dataset_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess FedChest source datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Dataset specific paths
    parser.add_argument(
        "--cxr14-dir",
        type=Path,
        default=Path("/path/to/archive/CXR14"),
        help="Path to the NIH ChestX-ray14 dataset root",
    )
    parser.add_argument(
        "--chexpert-dir",
        type=Path,
        default=Path("/path/to/archive/CheXpert-v1.0"),
        help="Path to the CheXpert dataset root",
    )
    parser.add_argument(
        "--padchest-dir",
        type=Path,
        default=Path("/path/to/sharedscratch/PadChest"),
        help="Path to the PadChest dataset root",
    )
    
    # General arguments
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./data/processed"),
        help="Directory for processed output",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("./configs/preprocessing.yaml"),
        help="Path to preprocessing configuration file",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_PREPROCESSORS.keys()),
        default=list(DATASET_PREPROCESSORS.keys()),
        help="Datasets to process (default: all)",
    )

    parser.add_argument(
        "--stage",
        type=str,
        choices=["prepare", "process", "all"],
        default="all",
        help="Preprocessing stage to run",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without processing",
    )
    args = parser.parse_args()
    
    # Map dataset names to their provided paths
    path_map: Dict[str, Optional[Path]] = {
        "cxr14": args.cxr14_dir,
        "chexpert": args.chexpert_dir,
        "padchest": args.padchest_dir,
    }

    # Validate that we have paths for the requested datasets
    missing_paths = []
    for ds in args.datasets:
        if path_map[ds] is None:
            missing_paths.append(ds)
    
    if missing_paths:
        logger.error("Missing paths for requested datasets:")
        for ds in missing_paths:
            logger.error(f"  --{ds}-dir is required")
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("FedChest Preprocessing Pipeline")
    logger.info("=" * 60)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Datasets to process: {args.datasets}")
    
    if args.dry_run:
        logger.info("\n[DRY RUN] Would process the following:")
        for dataset in args.datasets:
            path = path_map[dataset]
            status = "✓ Found" if path.exists() else "✗ Missing"
            logger.info(f"  - {dataset}: {path} ({status})")
        return 0
    
    # Process each dataset
    results = {}
    for dataset_name in args.datasets:
        # Get the specific path for this dataset
        source_path = path_map[dataset_name]
        
        success = preprocess_dataset(
            dataset_name=dataset_name,
            source_dir=source_path,
            output_dir=args.output_dir,
            config_path=args.config if args.config.exists() else None,
            stage=args.stage,
            seed=args.seed,
        )
        results[dataset_name] = success
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("PREPROCESSING SUMMARY")
    logger.info("=" * 60)
    
    all_success = True
    for dataset_name, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        logger.info(f"  {dataset_name}: {status}")
        if not success:
            all_success = False
    
    if all_success:
        logger.info("\n✓ All datasets preprocessed successfully!")
        logger.info(f"\nProcessed data saved to: {args.output_dir}")
    else:
        logger.error("\n✗ Some datasets failed to process. Check logs above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())