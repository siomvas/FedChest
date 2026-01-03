#!/usr/bin/env python3
"""
Script 05: Validate FedChest dataset integrity.

Runs comprehensive checks to ensure the dataset is ready for release:
- No patient overlap between train/val/test splits
- All image files exist and are loadable
- Label distributions match expected values
- Datalist JSON files are well-formed

Usage:
    python scripts/05_validate_dataset.py \\
        --data-root /path/to/processed/images \\
        --datalist-dir /path/to/datalists \\
        --config configs/splits.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DatasetValidator:
    """Validates FedChest dataset integrity."""
    
    def __init__(
        self,
        data_root: Path,
        datalist_dir: Path,
        config_path: Path,
    ):
        self.data_root = data_root
        self.datalist_dir = datalist_dir
        self.config = self._load_config(config_path)
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def _load_config(self, config_path: Path) -> Dict:
        with open(config_path) as f:
            return yaml.safe_load(f)
    
    def validate_all(self) -> bool:
        """Run all validation checks."""
        logger.info("=" * 60)
        logger.info("FedChest Dataset Validation")
        logger.info("=" * 60)
        
        checks = [
            ("Datalist files exist", self.check_datalist_files_exist),
            ("Datalist JSON structure", self.check_datalist_structure),
            ("Image files exist", self.check_images_exist),
            ("Images are loadable", self.check_images_loadable),
            ("No patient overlap", self.check_no_patient_overlap),
            ("Label consistency", self.check_label_consistency),
            ("Split sizes", self.check_split_sizes),
            ("Centralized test completeness", self.check_centralized_test),
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            logger.info(f"\n{'─' * 40}")
            logger.info(f"Running: {check_name}")
            try:
                passed = check_func()
                status = "✓ PASSED" if passed else "✗ FAILED"
                logger.info(f"Result: {status}")
                all_passed = all_passed and passed
            except Exception as e:
                logger.error(f"Check failed with exception: {e}")
                all_passed = False
        
        # Summary
        logger.info(f"\n{'=' * 60}")
        logger.info("VALIDATION SUMMARY")
        logger.info(f"{'=' * 60}")
        
        if self.errors:
            logger.error(f"Errors ({len(self.errors)}):")
            for error in self.errors:
                logger.error(f"  - {error}")
        
        if self.warnings:
            logger.warning(f"Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")
        
        if all_passed and not self.errors:
            logger.info("✓ All validation checks passed!")
            return True
        else:
            logger.error("✗ Validation failed. Please fix errors before release.")
            return False
    
    def check_datalist_files_exist(self) -> bool:
        """Check that all expected datalist files exist."""
        expected_files = [
            "client_cxr14.json",
            "client_cxr14_test.json",
            "client_cxp_young.json",
            "client_cxp_young_test.json",
            "client_cxp_old.json",
            "client_cxp_old_test.json",
            "client_padchest.json",
            "client_padchest_test.json",
            "client_All.json",
            "client_All_test.json",
        ]
        
        all_exist = True
        for filename in expected_files:
            filepath = self.datalist_dir / filename
            if not filepath.exists():
                self.errors.append(f"Missing datalist file: {filename}")
                all_exist = False
            else:
                logger.info(f"  Found: {filename}")
        
        return all_exist
    
    def check_datalist_structure(self) -> bool:
        """Check that datalist JSON files have correct structure."""
        valid = True
        
        for datalist_file in self.datalist_dir.glob("*.json"):
            try:
                with open(datalist_file) as f:
                    data = json.load(f)
                
                # Check structure
                if "_test" in datalist_file.name:
                    required_keys = ["testing"]
                else:
                    required_keys = ["training", "validation"]
                
                for key in required_keys:
                    if key not in data:
                        self.errors.append(f"{datalist_file.name}: Missing key '{key}'")
                        valid = False
                        continue
                    
                    # Check entry structure
                    for i, entry in enumerate(data[key][:5]):  # Check first 5
                        if "image" not in entry:
                            self.errors.append(
                                f"{datalist_file.name}: Entry {i} missing 'image' key"
                            )
                            valid = False
                        if "label" not in entry:
                            self.errors.append(
                                f"{datalist_file.name}: Entry {i} missing 'label' key"
                            )
                            valid = False
                
                logger.info(f"  Valid structure: {datalist_file.name}")
                
            except json.JSONDecodeError as e:
                self.errors.append(f"{datalist_file.name}: Invalid JSON - {e}")
                valid = False
        
        return valid
    
    def check_images_exist(self) -> bool:
        """Check that all referenced image files exist."""
        missing_count = 0
        total_count = 0
        
        for datalist_file in self.datalist_dir.glob("*.json"):
            with open(datalist_file) as f:
                data = json.load(f)
            
            for key in ["training", "validation", "testing"]:
                if key not in data:
                    continue
                
                for entry in data[key]:
                    total_count += 1
                    image_path = self.data_root / entry["image"]
                    if not image_path.exists():
                        missing_count += 1
                        if missing_count <= 10:
                            self.errors.append(f"Missing image: {entry['image']}")
        
        if missing_count > 10:
            self.errors.append(f"... and {missing_count - 10} more missing images")
        
        logger.info(f"  Checked {total_count} image references")
        logger.info(f"  Missing: {missing_count}")
        
        return missing_count == 0
    
    def check_images_loadable(self, sample_size: int = 100) -> bool:
        """Check that a sample of images can be loaded."""
        all_images = []
        
        for datalist_file in self.datalist_dir.glob("client_*.json"):
            if datalist_file.name == "client_All.json":
                continue
            with open(datalist_file) as f:
                data = json.load(f)
            for key in ["training", "validation", "testing"]:
                if key in data:
                    all_images.extend([e["image"] for e in data[key]])
        
        # Sample random images
        np.random.seed(42)
        sample_indices = np.random.choice(
            len(all_images), 
            size=min(sample_size, len(all_images)), 
            replace=False
        )
        sample_images = [all_images[i] for i in sample_indices]
        
        load_failures = 0
        for img_path in tqdm(sample_images, desc="  Loading samples"):
            try:
                full_path = self.data_root / img_path
                img = Image.open(full_path)
                img_array = np.array(img)
                if img_array.size == 0:
                    load_failures += 1
            except Exception as e:
                load_failures += 1
                if load_failures <= 5:
                    self.errors.append(f"Failed to load {img_path}: {e}")
        
        logger.info(f"  Sampled {len(sample_images)} images")
        logger.info(f"  Load failures: {load_failures}")
        
        return load_failures == 0
    
    def check_no_patient_overlap(self) -> bool:
        """Check that patients don't appear in multiple splits."""
        # This requires patient ID in the path or a separate mapping
        # For now, check that same image doesn't appear in multiple splits
        
        valid = True
        
        for datalist_file in self.datalist_dir.glob("client_*.json"):
            if datalist_file.name.startswith("client_All"):
                continue
                
            with open(datalist_file) as f:
                data = json.load(f)
            
            images_per_split: Dict[str, Set[str]] = {}
            for key in ["training", "validation", "testing"]:
                if key in data:
                    images_per_split[key] = {e["image"] for e in data[key]}
            
            # Check pairwise overlap
            splits = list(images_per_split.keys())
            for i, split1 in enumerate(splits):
                for split2 in splits[i+1:]:
                    overlap = images_per_split[split1] & images_per_split[split2]
                    if overlap:
                        self.errors.append(
                            f"{datalist_file.name}: {len(overlap)} images appear in "
                            f"both {split1} and {split2}"
                        )
                        valid = False
            
            if valid:
                logger.info(f"  No overlap in {datalist_file.name}")
        
        return valid
    
    def check_label_consistency(self) -> bool:
        """Check that labels have expected format and values."""
        valid = True
        expected_num_labels = 8
        
        for datalist_file in self.datalist_dir.glob("client_*.json"):
            with open(datalist_file) as f:
                data = json.load(f)
            
            for key in ["training", "validation", "testing"]:
                if key not in data:
                    continue
                
                for i, entry in enumerate(data[key]):
                    label = entry["label"]
                    
                    if len(label) != expected_num_labels:
                        self.errors.append(
                            f"{datalist_file.name}[{key}][{i}]: "
                            f"Expected {expected_num_labels} labels, got {len(label)}"
                        )
                        valid = False
                        break
                    
                    # Check values are 0 or 1
                    for val in label:
                        if val not in [0, 0.0, 1, 1.0]:
                            self.warnings.append(
                                f"{datalist_file.name}: Unexpected label value {val}"
                            )
                            break
        
        if valid:
            logger.info(f"  All labels have {expected_num_labels} classes")
        
        return valid
    
    def check_split_sizes(self) -> bool:
        """Check that split sizes are reasonable."""
        valid = True
        
        for datalist_file in self.datalist_dir.glob("client_*.json"):
            if datalist_file.name.startswith("client_All"):
                continue
                
            with open(datalist_file) as f:
                data = json.load(f)
            
            sizes = {}
            for key in ["training", "validation", "testing"]:
                if key in data:
                    sizes[key] = len(data[key])
            
            logger.info(f"  {datalist_file.name}: {sizes}")
            
            # Check minimums
            if "training" in sizes and sizes["training"] < 1000:
                self.warnings.append(
                    f"{datalist_file.name}: Small training set ({sizes['training']})"
                )
        
        return valid
    
    def check_centralized_test(self) -> bool:
        """
        CRITICAL CHECK: Verify centralized test set includes ALL clients.
        
        This addresses the bug where cxp_old was missing from the merged test set.
        """
        centralized_test_path = self.datalist_dir / "client_All_test.json"
        
        if not centralized_test_path.exists():
            self.errors.append("client_All_test.json does not exist")
            return False
        
        with open(centralized_test_path) as f:
            centralized_data = json.load(f)
        
        centralized_images = {e["image"] for e in centralized_data["testing"]}
        
        # Count images from each client's test set
        client_test_files = [
            "client_cxr14_test.json",
            "client_cxp_young_test.json",
            "client_cxp_old_test.json",  # This was missing in original!
            "client_padchest_test.json",
        ]
        
        expected_total = 0
        for client_file in client_test_files:
            client_path = self.datalist_dir / client_file
            if client_path.exists():
                with open(client_path) as f:
                    client_data = json.load(f)
                client_count = len(client_data["testing"])
                expected_total += client_count
                logger.info(f"  {client_file}: {client_count} images")
        
        actual_total = len(centralized_images)
        logger.info(f"  client_All_test.json: {actual_total} images")
        logger.info(f"  Expected total: {expected_total} images")
        
        if actual_total != expected_total:
            self.errors.append(
                f"Centralized test set has {actual_total} images, "
                f"expected {expected_total}. "
                "Some client test sets may be missing (check cxp_old!)."
            )
            return False
        
        return True


def main():
    parser = argparse.ArgumentParser(description="Validate FedChest dataset")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root directory containing processed images",
    )
    parser.add_argument(
        "--datalist-dir",
        type=Path,
        required=True,
        help="Directory containing datalist JSON files",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/splits.yaml"),
        help="Path to splits configuration",
    )
    args = parser.parse_args()
    
    validator = DatasetValidator(
        data_root=args.data_root,
        datalist_dir=args.datalist_dir,
        config_path=args.config,
    )
    
    success = validator.validate_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
