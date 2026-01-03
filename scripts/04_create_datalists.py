"""
Create datalist JSON files for FedChest.

This script generates the JSON datalist files used by the DataLoaders.
It creates per-client datalists and a centralized datalist.

Usage:
    python scripts/04_create_datalists.py \\
        --data-root /path/to/processed/images \\
        --csv-dir /path/to/csv/files \\
        --output-dir /path/to/output/datalists \\
        --config configs/splits.yaml
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict:
    """Load splits configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_datalist_entry(row: pd.Series, path_column: str, label_columns: List[str]) -> Dict:
    """
    Create a single datalist entry with relative path.
    """
    return {
        "image": str(row[path_column]),  # Store relative path
        "label": row[label_columns].values.tolist(),
    }


def create_client_datalist(
    train_csv: Path,
    val_csv: Path,
    test_csv: Optional[Path],
    path_column: str,
    label_columns: List[str],
) -> tuple[Dict, Optional[Dict], Dict]:
    """
    Create datalist dicts for a single client.
    
    Returns:
        Tuple of (train_val_datalist, test_datalist, summary)
    """
    # Load CSVs
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    # Create train/val datalist
    train_val_datalist = {
        "training": [],
        "validation": [],
    }
    
    for _, row in train_df.iterrows():
        entry = create_datalist_entry(row, path_column, label_columns)
        train_val_datalist["training"].append(entry)
    
    for _, row in val_df.iterrows():
        entry = create_datalist_entry(row, path_column, label_columns)
        train_val_datalist["validation"].append(entry)
    
    # Create test datalist if test CSV exists
    test_datalist = None
    if test_csv and test_csv.exists():
        test_df = pd.read_csv(test_csv)
        test_datalist = {"testing": []}
        
        for _, row in test_df.iterrows():
            entry = create_datalist_entry(row, path_column, label_columns)
            test_datalist["testing"].append(entry)
    
    # Create summary with class counts
    summary = train_df[label_columns].sum().to_dict()
    summary["samples"] = len(train_df)
    
    return train_val_datalist, test_datalist, summary


def merge_datalists(
    datalists: List[Dict],
    keys: List[str],
) -> Dict:
    """
    Merge multiple datalists.
    """
    merged = {key: [] for key in keys}
    
    for datalist in datalists:
        for key in keys:
            if key in datalist:
                merged[key].extend(datalist[key])
    
    return merged


def main():
    parser = argparse.ArgumentParser(description="Create FedChest datalist JSON files")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root directory containing processed images",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        required=True,
        help="Directory containing train/val/test CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for datalist JSON files",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/splits.yaml"),
        help="Path to splits configuration",
    )
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = load_config(args.config)
    
    # Define clients and their CSV file patterns
    # Label columns (consistent order from config)
    label_columns = config["common_diseases"]
    
    clients = {
        "cxr14": {
            "train": args.csv_dir / "cxr14_train.csv",
            "val": args.csv_dir / "cxr14_val.csv",
            "test": args.csv_dir / "cxr14_test.csv",
            "path_column": "Path",
        },
        "cxp_young": {
            "train": args.csv_dir / "cxp_young_train.csv",
            "val": args.csv_dir / "cxp_young_val.csv",
            "test": args.csv_dir / "cxp_young_test.csv",
            "path_column": "Path",
        },
        "cxp_old": {
            "train": args.csv_dir / "cxp_old_train.csv",
            "val": args.csv_dir / "cxp_old_val.csv",
            "test": args.csv_dir / "cxp_old_test.csv",
            "path_column": "Path",
        },
        "padchest": {
            "train": args.csv_dir / "padchest_train.csv",
            "val": args.csv_dir / "padchest_val.csv",
            "test": args.csv_dir / "padchest_test.csv",
            "path_column": "Path",
        },
    }
    
    # Process each client
    all_train_val = []
    all_test = []
    
    for client_id, client_config in clients.items():
        logger.info(f"Processing client: {client_id}")
        
        train_val, test, summary = create_client_datalist(
            train_csv=client_config["train"],
            val_csv=client_config["val"],
            test_csv=client_config.get("test"),
            path_column=client_config["path_column"],
            label_columns=label_columns,
        )
        
        # Save client datalists
        train_val_path = args.output_dir / f"client_{client_id}.json"
        with open(train_val_path, "w") as f:
            json.dump(train_val, f, indent=4)
        logger.info(f"  Saved: {train_val_path}")
        logger.info(f"    Training samples: {len(train_val['training'])}")
        logger.info(f"    Validation samples: {len(train_val['validation'])}")
        
        if test:
            test_path = args.output_dir / f"client_{client_id}_test.json"
            with open(test_path, "w") as f:
                json.dump(test, f, indent=4)
            logger.info(f"  Saved: {test_path}")
            logger.info(f"    Test samples: {len(test['testing'])}")
            all_test.append(test)
        
        # Save summary
        summary_path = args.output_dir / f"{client_id}_summary.txt"
        with open(summary_path, "w") as f:
            json.dump(summary, f)
        logger.info(f"  Saved: {summary_path}")
        
        all_train_val.append(train_val)
    
    # Create centralized datalists
    logger.info("Creating centralized datalists...")
    
    # Merge training and validation
    centralized_train_val = merge_datalists(all_train_val, ["training", "validation"])
    centralized_path = args.output_dir / "client_All.json"
    with open(centralized_path, "w") as f:
        json.dump(centralized_train_val, f, indent=4)
    logger.info(f"Saved: {centralized_path}")
    logger.info(f"  Training samples: {len(centralized_train_val['training'])}")
    logger.info(f"  Validation samples: {len(centralized_train_val['validation'])}")
    
    centralized_test = merge_datalists(all_test, ["testing"])
    centralized_test_path = args.output_dir / "client_All_test.json"
    with open(centralized_test_path, "w") as f:
        json.dump(centralized_test, f, indent=4)
    logger.info(f"Saved: {centralized_test_path}")
    logger.info(f"  Test samples: {len(centralized_test['testing'])}")
    
    # Validation check
    expected_test_clients = 4
    if len(all_test) != expected_test_clients:
        logger.warning(
            f"Expected {expected_test_clients} test sets, got {len(all_test)}. "
            "Some clients may be missing test data."
        )
    
    logger.info("Datalist generation complete!")


if __name__ == "__main__":
    main()
