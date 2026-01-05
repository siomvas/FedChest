"""
PadChest dataset preprocessor.

Source: https://bimcv.cipf.es/bimcv-projects/padchest/

Original dataset structure:
- images/: Contains PNG images
- PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv: Metadata with labels

Labels are stored as string representations of Python lists.
Multiple label aliases exist for the same condition.

Known corrupted images that must be skipped:
- 216840111366964013076187734852011291090445391_00-196-188.png
- 216840111366964012373310883942009170084120009_00-097-074.png
- 216840111366964012373310883942009180082307973_00-097-011.png
- 216840111366964012819207061112010281134410801_00-129-131.png
- 216840111366964012989926673512011083134050913_00-168-009.png
- 216840111366964012819207061112010306085429121_04-020-102.png
- 216840111366964012819207061112010307142602253_04-014-084.png
"""

import logging
import shutil
import ast
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from fedchest.preprocessing.base import BasePreprocessor
from fedchest.preprocessing.image_transforms import preprocess_chest_xray

logger = logging.getLogger(__name__)


# Known corrupted images to skip
CORRUPTED_IMAGES: Set[str] = {
    "216840111366964013076187734852011291090445391_00-196-188.png",
    "216840111366964012373310883942009170084120009_00-097-074.png",
    "216840111366964012373310883942009180082307973_00-097-011.png",
    "216840111366964012819207061112010281134410801_00-129-131.png",
    "216840111366964012989926673512011083134050913_00-168-009.png",
    "216840111366964012819207061112010306085429121_04-020-102.png",
    "216840111366964012819207061112010307142602253_04-014-084.png",
}


class PadChestPreprocessor(BasePreprocessor):
    """
    Preprocessor for PadChest dataset.
    
    Special handling:
    - Labels stored as string lists, need parsing
    - Multiple label aliases for each disease
    - Known corrupted images must be skipped
    - Filter to AP/PA views only
    """
    
    DATASET_NAME = "padchest"
    
    def __init__(
        self,
        source_dir: Path,
        output_dir: Path,
        config_path: Optional[Path] = None,
    ):
        super().__init__(source_dir, output_dir, config_path)
        
        # Expected source files
        self.metadata_file = self.source_dir / "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"
        self.images_source_dir = self.source_dir / "images"
        
        # Alternative metadata filename
        if not self.metadata_file.exists():
            alt_file = self.source_dir / "padchest_labels.csv"
            if alt_file.exists():
                self.metadata_file = alt_file
        
        self.interim_dir = self.output_dir / "interim_splits"
        self.interim_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_label_map(self) -> dict:
        """Load label mapping from config."""
        alias_config = self.config["padchest"]["label_aliases"]
        if not alias_config:
            raise ValueError("No label aliases found in config for PadChest.")
        
        source_to_target = defaultdict(list)
        for target, sources in alias_config.items():
            for source in sources:
                source_to_target[source.lower().strip()].append(target)

        return source_to_target
    
    def _parse_label_string(self, label_str: str) -> List[str]:
        """
        Parse label string from PadChest format.
        
        Labels are stored as "['label1', 'label2']" strings.
        
        Args:
            label_str: String representation of label list
        
        Returns:
            List of label strings (lowercase, stripped)
        """
        if pd.isna(label_str) or label_str.strip() == "":
            return []
        
        try: 
            parsed = ast.literal_eval(label_str)
            if isinstance(parsed, list):
                return [l.lower().strip() for l in parsed]
            return []
        except (ValueError, SyntaxError):
            logger.warning(f"Failed to parse label string: {label_str}")
            return []
    
    def _map_to_common_disease(self, labels: List[str]) -> dict:
        """
        Map raw labels to common disease columns.
        
        Uses LABEL_ALIASES to map various terms to standard disease names.
        
        Args:
            labels: List of raw label strings
        
        Returns:
            Dict mapping disease name to 1.0 or 0.0
        """
        result = {disease: 0.0 for disease in self.common_diseases}
        changes = {}
        for label in labels:
            label_lower = label.lower().strip()
            
            # Check each disease's aliases
            for disease, aliases in LABEL_ALIASES.items():
                if label_lower in aliases:
                    changes[label_lower] = changes.get(label_lower, 0) + 1
                    # Map to standard name
                    if disease == "no_finding":
                        result["No Finding"] = 1.0
                    else:
                        # Capitalize first letter
                        standard_name = disease.replace("_", " ").title()
                        if standard_name in result:
                            result[standard_name] = 1.0
        print(f"Mapping labels {labels} to changes: {changes}")
        return result
    
    def load_and_clean_metadata(self) -> pd.DataFrame:
        """
        Load, parse, clean, apply mappings, and filter PadChest metadata.
        
        Returns:
            DataFrame with standardized columns
        """
        logger.info(f"Loading PadChest metadata from {self.metadata_file}")
        
        # Load CSV
        df = pd.read_csv(
            self.metadata_file,
            usecols=["ImageID", "PatientID", "Projection", "Labels"],
        )
        df.rename(columns={
            "ImageID": "Path",
            "PatientID": "Patient ID",
            "Labels": "RawLabels"
        }, inplace=True)
        
        logger.info(f"Loaded {len(df)} rows")
        
        # Filter to AP/PA projections only
        valid_projections = ["PA", "AP", "AP_horizontal"]
        df = df[df["Projection"].isin(valid_projections)].copy()
        logger.info(f"After filtering projections: {len(df)} rows")
        
        # Remove known corrupted images
        corrupted_count = df["Path"].isin(CORRUPTED_IMAGES).sum()
        if corrupted_count > 0:
            logger.info(f"Removing {corrupted_count} known corrupted images")
            df = df[~df["Path"].isin(CORRUPTED_IMAGES)]

        # Remove rows with NA labels
        na_count = df["RawLabels"].isna().sum()
        if na_count > 0:
            logger.info(f"Removing {na_count} rows with NA labels")
            df = df.dropna(subset=["RawLabels"])
        
        # Parse labels and create disease columns
        source_to_target = self._load_label_map()

        logger.info("Parsing labels...")
        df["LabelList"] = df["RawLabels"].apply(self._parse_label_string)

        for disease in self.common_diseases:
            df[disease] = 0.0

        audit_log = defaultdict(set)

        disease_index = {disease: idx for idx, disease in enumerate(self.common_diseases)}

        n_rows = len(df)
        n_classes = len(self.common_diseases)
        label_matrix = np.zeros((n_rows, n_classes), dtype=np.float32)

        for i, labels in enumerate(tqdm(df["LabelList"], total=n_rows, desc="Mapping labels")):
            for source_label in labels:
                if source_label in source_to_target:
                    target_labels = source_to_target[source_label]

                    # log for supervision, track *unique* mappings
                    audit_log[source_label].update(target_labels)

                    for target_label in target_labels:
                        if target_label in disease_index:
                            label_idx = disease_index[target_label]
                            label_matrix[i, label_idx] = 1.0
    
        # Assign back to dataframe
        for name, idx in disease_index.items():
            df[name] = label_matrix[:, idx]
        
        logger.info("=" * 40)
        logger.info("MAPPING AUDIT LOG (Unique Mappings Applied)")
        logger.info("=" * 40)
        for source, mapped_targets in sorted(audit_log.items()):
            logger.info(f"Source: '{source}' -> Mapped to: {sorted(list(mapped_targets))}")
        
        # Also verify the counts match (number of mappings applied matches the config)
        total_mappings = sum(len(v) for v in audit_log.values())
        logger.info(f"Total unique source labels mapped: {len(audit_log)}, vs the config mappings: {len(source_to_target)}")

        logger.info("=" * 40)
        
        df.drop(columns=["RawLabels"], inplace=True)
        
        # Filter to common diseases (keep only rows with at least one)
        df = self.filter_common_diseases(df)
        
        # Remove duplicates
        dup_subset = ["Patient ID", "Projection"] + self.common_diseases
        df = self.remove_duplicates(df, dup_subset)
        
        # Validate
        self.validate_labels(df)
        
        # Log distribution
        dist = self.get_label_distribution(df)
        logger.info(f"Label distribution: {dist}")
        
        return df
    
    def prepare_splits(self, seed: int = 42, test_ratio: float=0.1, val_ratio: float=0.15):
        """
        Prepare interim splits without image processing.
        
        PadChest has a large number of patients, so we will create splits
        at the patient level to avoid data leakage.
        
        Args:
            seed: Random seed for reproducibility
            test_ratio: Fraction of patients for test set
            val_ratio: Fraction of patients for val set
        """
        logger.info("Preparing interim splits for PadChest...")
        
        # Load and clean metadata
        df = self.load_and_clean_metadata()
        
        # Get unique patients and shuffle
        patients = df["Patient ID"].unique()
        np.random.seed(seed)
        np.random.shuffle(patients)
        
        # Split patients
        test_patients = int(len(patients) * test_ratio)
        val_patients = int(len(patients) * val_ratio)

        test_patient_ids = patients[:test_patients]
        val_patient_ids = patients[test_patients:test_patients + val_patients]
        train_patient_ids = patients[test_patients + val_patients:]
        
        # Create splits
        test_df = df[df["Patient ID"].isin(test_patient_ids)].copy()
        val_df = df[df["Patient ID"].isin(val_patient_ids)].copy()
        train_df = df[df["Patient ID"].isin(train_patient_ids)].copy()
        
        logger.info(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        # Save interim CSVs
        train_path = self.interim_dir / "padchest_train.csv"
        val_path = self.interim_dir / "padchest_val.csv"
        test_path = self.interim_dir / "padchest_test.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        return True
    
    def process_images(self):
        """
        Preprocess all images in the dataset.
        """
        logger.info("=" * 40)
        logger.info("STAGE 2: Processing Images")
        logger.info("=" * 40)

        splits = list(self.interim_dir.glob("padchest_*.csv"))
        if not splits:
            logger.error("No interim split CSVs for PadChest found. Please run prepare_splits() first.")
            return {}
        
        results = {}
        
        output_subdir = self.images_dir / self.DATASET_NAME
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        for split_path in splits:
            split_name = split_path.stem
            logger.info(f"Processing split: {split_name}")
            
            df = pd.read_csv(split_path)
            valid_indices = []
        
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
                source_path = self.images_source_dir / row["Path"]
                output_path = self.images_dir / self.DATASET_NAME / row["Path"]
            
                if preprocess_chest_xray(
                    image_path=source_path,
                    output_path=output_path,
                    size=self.image_size,
                    crop_percent=self.crop_percent,
                    clip_limit=self.clip_limit,
                    tile_grid_size=self.tile_grid_size,
                ):
                    valid_indices.append(idx)
                else:
                    logger.warning(f"Failed to process image: {source_path}")
            
            df_final = df.loc[valid_indices].copy()
            df_final["Path"] = df_final["Path"].apply(lambda p: f"{self.DATASET_NAME}/{p}")
        
            self.save_csv(df_final, f"{split_name}.csv")
            results[split_name] = f"{split_name}.csv"

        return results
    
    def run(self, seed: int = 42, stage: str = "all") -> dict:
        """
        Run the preprocessing pipeline.
        
        Args:
            seed: Random seed for splits
            stage: Stage to run ("prepare", "process", or "all")
        
        Returns:
            Dict with paths to output files
        """
        logger.info("=" * 60)
        logger.info("PadChest Preprocessing Pipeline")
        logger.info("=" * 60)
        
        if stage in ("prepare", "all"):
            success = self.prepare_splits(seed=seed)
            if not success:
                logger.error("Failed to prepare splits. Aborting.")
                return {}
        if stage in ("process", "all"):
            return self.process_images()
        
        logger.info("PadChest preprocessing complete!")

        return {}
