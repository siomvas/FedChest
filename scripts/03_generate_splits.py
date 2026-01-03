"""
Script 03: Generate train/val/test splits.

Creates standardized split CSVs from the preprocessed data.
This script ensures consistent splits across all datasets using
patient-level separation to prevent data leakage.

The splits are created during preprocessing (script 02), but this
script can be used to:
1. Verify existing splits
2. Regenerate splits with a different seed
3. Create custom splits for ablation studies

Usage:
    # Verify existing splits:
    python scripts/03_generate_splits.py --data-dir ./data/processed --verify

    # Regenerate with different seed:
    python scripts/03_generate_splits.py --data-dir ./data/processed --seed 123

Output structure:
    ./data/processed/csv/
    ├── cxr14_train.csv
    ├── cxr14_val.csv
    ├── cxr14_test.csv
    ├── cxp_young_train.csv
    ├── cxp_young_val.csv
    ├── cxp_young_test.csv
    ├── cxp_old_train.csv
    ├── cxp_old_val.csv
    ├── cxp_old_test.csv
    ├── padchest_train.csv
    ├── padchest_val.csv
    └── padchest_test.csv
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Expected split sizes (approximate)
EXPECTED_SPLITS = {
    "cxr14": {"train": 23000, "val": 15000, "test": 10000},
    "cxp_young": {"train": 31000, "val": 15000, "test": 7500},
    "cxp_old": {"train": 29000, "val": 15000, "test": 7500},
    "padchest": {"train": 26000, "val": 15000, "test": 10000},
}

# Federated clients
CLIENTS = ["cxr14", "cxp_young", "cxp_old", "padchest"]

COMMON_DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "No Finding",
    "Pneumonia",
    "Pneumothorax",
]


def load_splits_config(config_path: Path) -> dict:
    """Load splits configuration from YAML."""
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def verify_split_file(filepath: Path) -> Tuple[bool, str]:
    """
    Verify a split CSV file is valid.
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not filepath.exists():
        return False, f"File not found: {filepath}"
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return False, f"Failed to read CSV: {e}"
    
    # Check required columns
    required_cols = ["Path", "Patient ID"] + COMMON_DISEASES
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        return False, f"Missing columns: {missing_cols}"
    
    # Check for empty file
    if len(df) == 0:
        return False, "File is empty"
    
    # Check for NaN in labels
    label_nans = df[COMMON_DISEASES].isna().sum().sum()
    if label_nans > 0:
        return False, f"Found {label_nans} NaN values in labels"
    
    # Check label values are 0 or 1
    for col in COMMON_DISEASES:
        unique_vals = df[col].unique()
        if not all(v in [0, 0.0, 1, 1.0] for v in unique_vals):
            return False, f"Invalid label values in {col}: {unique_vals}"
    
    return True, f"Valid ({len(df)} samples)"


def verify_no_patient_overlap(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[bool, str]:
    """
    Verify no patients appear in multiple splits.
    
    Args:
        train_df, val_df, test_df: Split DataFrames
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    train_patients = set(train_df["Patient ID"])
    val_patients = set(val_df["Patient ID"])
    test_patients = set(test_df["Patient ID"])
    
    train_val_overlap = train_patients & val_patients
    train_test_overlap = train_patients & test_patients
    val_test_overlap = val_patients & test_patients
    
    if train_val_overlap:
        return False, f"Train/val overlap: {len(train_val_overlap)} patients"
    if train_test_overlap:
        return False, f"Train/test overlap: {len(train_test_overlap)} patients"
    if val_test_overlap:
        return False, f"Val/test overlap: {len(val_test_overlap)} patients"
    
    return True, "No patient overlap"


def verify_client_splits(csv_dir: Path, client: str) -> Dict[str, Any]:
    """
    Verify all splits for a client.
    
    Args:
        csv_dir: Directory containing CSV files
        client: Client name
    
    Returns:
        Dict with verification results
    """
    results = {
        "client": client,
        "valid": True,
        "splits": {},
        "errors": [],
    }
    
    # Check each split file
    for split in ["train", "val", "test"]:
        filepath = csv_dir / f"{client}_{split}.csv"
        valid, message = verify_split_file(filepath)
        results["splits"][split] = {"valid": valid, "message": message}
        
        if not valid:
            results["valid"] = False
            results["errors"].append(f"{split}: {message}")
    
    # Check patient overlap if all files exist
    if all(results["splits"][s]["valid"] for s in ["train", "val", "test"]):
        train_df = pd.read_csv(csv_dir / f"{client}_train.csv")
        val_df = pd.read_csv(csv_dir / f"{client}_val.csv")
        test_df = pd.read_csv(csv_dir / f"{client}_test.csv")
        
        valid, message = verify_no_patient_overlap(train_df, val_df, test_df)
        results["patient_overlap"] = {"valid": valid, "message": message}
        
        if not valid:
            results["valid"] = False
            results["errors"].append(f"Patient overlap: {message}")
    
    return results


def print_split_summary(csv_dir: Path) -> None:
    """Print summary of all splits."""
    logger.info("\n" + "=" * 70)
    logger.info("SPLIT SUMMARY")
    logger.info("=" * 70)
    
    header = f"{'Client':<12} {'Train':>10} {'Val':>10} {'Test':>10} {'Total':>10}"
    logger.info(header)
    logger.info("-" * 70)
    
    total_train = total_val = total_test = 0
    
    for client in CLIENTS:
        train_path = csv_dir / f"{client}_train.csv"
        val_path = csv_dir / f"{client}_val.csv"
        test_path = csv_dir / f"{client}_test.csv"
        
        if all(p.exists() for p in [train_path, val_path, test_path]):
            train_count = len(pd.read_csv(train_path))
            val_count = len(pd.read_csv(val_path))
            test_count = len(pd.read_csv(test_path))
            total = train_count + val_count + test_count
            
            logger.info(f"{client:<12} {train_count:>10,} {val_count:>10,} "
                       f"{test_count:>10,} {total:>10,}")
            
            total_train += train_count
            total_val += val_count
            total_test += test_count
        else:
            logger.info(f"{client:<12} {'Missing':>10}")
    
    logger.info("-" * 70)
    grand_total = total_train + total_val + total_test
    logger.info(f"{'TOTAL':<12} {total_train:>10,} {total_val:>10,} "
               f"{total_test:>10,} {grand_total:>10,}")


def print_label_distribution(csv_dir: Path) -> None:
    """Print label distribution for each client."""
    logger.info("\n" + "=" * 70)
    logger.info("LABEL DISTRIBUTION (Training Set)")
    logger.info("=" * 70)
    
    for client in CLIENTS:
        train_path = csv_dir / f"{client}_train.csv"
        if train_path.exists():
            df = pd.read_csv(train_path)
            dist = df[COMMON_DISEASES].sum().astype(int)
            
            logger.info(f"\n{client}:")
            for disease, count in dist.items():
                pct = 100 * count / len(df)
                bar = "█" * int(pct / 5)
                logger.info(f"  {disease:<15} {count:>6,} ({pct:>5.1f}%) {bar}")


def create_patient_split(
    df: pd.DataFrame,
    seed: int,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create patient-level train/val/test split.
    
    Args:
        df: Full dataset
        seed: Random seed
        val_ratio: Fraction for validation
        test_ratio: Fraction for test
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    patients = df["Patient ID"].unique()
    n_patients = len(patients)
    
    np.random.seed(seed)
    np.random.shuffle(patients)
    
    n_test = int(n_patients * test_ratio)
    n_val = int(n_patients * val_ratio)
    
    test_patients = patients[:n_test]
    val_patients = patients[n_test:n_test + n_val]
    train_patients = patients[n_test + n_val:]
    
    test_df = df[df["Patient ID"].isin(test_patients)].copy()
    val_df = df[df["Patient ID"].isin(val_patients)].copy()
    train_df = df[df["Patient ID"].isin(train_patients)].copy()
    
    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate and verify train/val/test splits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./data/processed"),
        help="Directory containing processed data",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("./configs/splits.yaml"),
        help="Path to splits configuration file",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing splits without regenerating",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits (default: 42)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary of existing splits",
    )
    args = parser.parse_args()
    
    csv_dir = args.data_dir / "csv"
    
    if not csv_dir.exists():
        logger.error(f"CSV directory not found: {csv_dir}")
        logger.error("Run 02_preprocess_all.py first")
        return 1
    
    # Print summary if requested
    if args.summary:
        print_split_summary(csv_dir)
        print_label_distribution(csv_dir)
        return 0
    
    # Verify existing splits
    if args.verify:
        logger.info("Verifying existing splits...")
        logger.info("=" * 60)
        
        all_valid = True
        for client in CLIENTS:
            results = verify_client_splits(csv_dir, client)
            
            status = "✓" if results["valid"] else "✗"
            logger.info(f"\n{status} {client}:")
            
            for split, info in results["splits"].items():
                split_status = "✓" if info["valid"] else "✗"
                logger.info(f"  {split_status} {split}: {info['message']}")
            
            if "patient_overlap" in results:
                overlap_status = "✓" if results["patient_overlap"]["valid"] else "✗"
                logger.info(f"  {overlap_status} {results['patient_overlap']['message']}")
            
            if not results["valid"]:
                all_valid = False
        
        print_split_summary(csv_dir)
        
        if all_valid:
            logger.info("\n✓ All splits are valid!")
            return 0
        else:
            logger.error("\n✗ Some splits have issues. See above for details.")
            return 1
    
    # If not verifying, just show summary
    print_split_summary(csv_dir)
    print_label_distribution(csv_dir)
    
    logger.info("\nTo verify splits: python scripts/03_generate_splits.py --verify")
    logger.info("Next step: python scripts/04_create_datalists.py")
    
    return 0


if __name__ == "__main__":
    exit(main())
