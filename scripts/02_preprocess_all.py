"""

Runs the preprocessing pipeline for all source datasets:
- CXR14: NIH ChestX-ray14
- CheXpert: Stanford CheXpert (split into cxp_young and cxp_old)
- PadChest: BIMCV PadChest

Preprocessing steps for each dataset:
1. Load and parse metadata/labels
2. Filter to common diseases (9 classes)
3. Remove duplicates
4. Crop borders (5% from each edge)
5. Resize to 224x224
6. Apply CLAHE
7. Create patient-level train/val/test splits
8. Save processed images and CSV files

Usage:
    python scripts/02_preprocess_all.py --raw-dir ./data/raw --output-dir ./data/processed

    # Process only specific datasets:
    python scripts/02_preprocess_all.py --datasets cxr14 chexpert

    # Use custom config:
    python scripts/02_preprocess_all.py --config ./configs/preprocessing.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import yaml

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
    raw_dir: Path,
    output_dir: Path,
    config_path: Optional[Path],
    seed: int,
) -> bool:
    """
    Preprocess a single dataset.
    
    Args:
        dataset_name: Name of dataset (cxr14, chexpert, padchest)
        raw_dir: Directory containing raw dataset
        output_dir: Directory for processed output
        config_path: Path to preprocessing config
        seed: Random seed for splits
    
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {dataset_name.upper()}")
    logger.info(f"{'='*60}")
    
    preprocessor_class = DATASET_PREPROCESSORS.get(dataset_name)
    if not preprocessor_class:
        logger.error(f"Unknown dataset: {dataset_name}")
        return False
    
    source_dir = raw_dir / dataset_name
    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        return False
    
    try:
        preprocessor = preprocessor_class(
            source_dir=source_dir,
            output_dir=output_dir,
            config_path=config_path,
        )
        
        results = preprocessor.run(seed=seed)
        
        logger.info(f"\n{dataset_name.upper()} preprocessing complete!")
        logger.info(f"Output files: {results}")
        
        return True
        
    except Exception as e:
        logger.exception(f"Error processing {dataset_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess all FedChest source datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("./data/raw"),
        help="Directory containing raw source datasets",
    )
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val/test splits (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without processing",
    )
    args = parser.parse_args()
    
    # Validate paths
    if not args.raw_dir.exists():
        logger.error(f"Raw data directory not found: {args.raw_dir}")
        logger.error("Run 01_download_sources.py first to download datasets")
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("FedChest Preprocessing Pipeline")
    logger.info("=" * 60)
    logger.info(f"Raw data directory: {args.raw_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Datasets to process: {args.datasets}")
    
    if args.dry_run:
        logger.info("\n[DRY RUN] Would process the following datasets:")
        for dataset in args.datasets:
            source_dir = args.raw_dir / dataset
            status = "✓ Found" if source_dir.exists() else "✗ Missing"
            logger.info(f"  - {dataset}: {status}")
        return 0
    
    # Process each dataset
    results = {}
    for dataset_name in args.datasets:
        success = preprocess_dataset(
            dataset_name=dataset_name,
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            config_path=args.config if args.config.exists() else None,
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
        logger.info("\nNext step: Run script 03 to generate datalist splits")
        logger.info(f"  python scripts/03_generate_splits.py --data-dir {args.output_dir}")
    else:
        logger.error("\n✗ Some datasets failed to process. Check logs above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
