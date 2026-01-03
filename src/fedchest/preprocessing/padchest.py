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

# Label aliases - multiple terms map to the same common disease
# load from configs/preprocessing.yaml
with open(Path(__file__).parent.parent.parent / "configs" / "preprocessing.yaml", "r") as f:
    PREPROCESSING_CONFIGS = yaml.safe_load(f)
LABEL_ALIASES = PREPROCESSING_CONFIGS["padchest"]["label_aliases"]


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
    
    def _parse_label_string(self, label_str: str) -> List[str]:
        """
        Parse label string from PadChest format.
        
        Labels are stored as "['label1', 'label2']" strings.
        
        Args:
            label_str: String representation of label list
        
        Returns:
            List of label strings (lowercase, stripped)
        """
        if pd.isna(label_str):
            return []
        
        # Remove brackets and quotes, split by comma
        cleaned = label_str.replace("[", "").replace("]", "").replace("'", "")
        labels = [l.strip().lower() for l in cleaned.split(",")]
        
        return [l for l in labels if l]  # Remove empty strings
    
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
    
    def load_metadata(self) -> pd.DataFrame:
        """
        Load and parse PadChest metadata.
        
        Returns:
            DataFrame with standardized columns
        """
        logger.info(f"Loading PadChest metadata from {self.metadata_file}")
        
        # Load CSV
        df = pd.read_csv(
            self.metadata_file,
            usecols=["ImageID", "PatientID", "Projection", "Labels"],
        )
        
        logger.info(f"Loaded {len(df)} rows")
        
        # Filter to AP/PA projections only
        valid_projections = ["PA", "AP", "AP_horizontal"]
        df = df[df["Projection"].isin(valid_projections)].copy()
        logger.info(f"After filtering projections: {len(df)} rows")
        
        # Remove rows with NA labels
        na_count = df["Labels"].isna().sum()
        if na_count > 0:
            logger.info(f"Removing {na_count} rows with NA labels")
            df = df.dropna(subset=["Labels"])
        
        # Parse labels and create disease columns
        logger.info("Parsing labels...")
        disease_data = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing labels"):
            raw_labels = self._parse_label_string(row["Labels"])
            disease_dict = self._map_to_common_disease(raw_labels)
            disease_data.append(disease_dict)
        
        disease_df = pd.DataFrame(disease_data)
        df = pd.concat([df.reset_index(drop=True), disease_df], axis=1)
        df.drop(columns=["Labels"], inplace=True)
        
        # Filter to common diseases (keep only rows with at least one)
        df = self.filter_common_diseases(df)
        
        # Remove known corrupted images
        corrupted_count = df["ImageID"].isin(CORRUPTED_IMAGES).sum()
        if corrupted_count > 0:
            logger.info(f"Removing {corrupted_count} known corrupted images")
            df = df[~df["ImageID"].isin(CORRUPTED_IMAGES)]
        
        # Remove duplicates
        dup_subset = ["PatientID", "Projection"] + [self.common_diseases[i] for i in range(len(self.common_diseases)) if self.common_diseases[i] in df.columns]
        df = self.remove_duplicates(df, dup_subset)
        
        # Standardize column names
        df = df.rename(columns={
            "ImageID": "Path",
            "PatientID": "Patient ID",
        })
        
        # Ensure correct column order
        cols = ["Path", "Patient ID", "Projection"] + self.common_diseases
        df = df[[c for c in cols if c in df.columns]]
        
        # Validate
        self.validate_labels(df)
        
        # Log distribution
        dist = self.get_label_distribution(df)
        logger.info(f"Label distribution: {dist}")
        
        return df
    
    def get_source_image_path(self, row: pd.Series) -> Path:
        """Get path to source image for a row."""
        return self.images_source_dir / row["Path"]
    
    def get_output_image_path(self, row: pd.Series) -> Path:
        """Get path for processed output image."""
        return self.images_dir / self.DATASET_NAME / row["Path"]
    
    def preprocess_images(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess all images in the dataset.
        
        Args:
            df: DataFrame with Path column
        
        Returns:
            DataFrame with updated paths (failed images removed)
        """
        logger.info(f"Preprocessing {len(df)} images...")
        
        # Create output subdirectory
        output_subdir = self.images_dir / self.DATASET_NAME
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        failed_indices = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
            source_path = self.get_source_image_path(row)
            output_path = self.get_output_image_path(row)
            
            success = preprocess_chest_xray(
                image_path=source_path,
                output_path=output_path,
                size=self.image_size,
                crop_percent=self.crop_percent,
                clip_limit=self.clip_limit,
                tile_grid_size=self.tile_grid_size,
            )
            
            if not success:
                failed_indices.append(idx)
        
        # Remove failed images
        if failed_indices:
            logger.warning(f"Failed to process {len(failed_indices)} images")
            df = df.drop(index=failed_indices)
        
        # Update paths to be relative
        df["Path"] = df["Path"].apply(lambda p: f"{self.DATASET_NAME}/{p}")
        
        logger.info(f"Successfully processed {len(df)} images")
        return df
    
    def create_splits(
        self,
        df: pd.DataFrame,
        seed: int = 42,
        test_patients: int = 8471,
        val_patients: int = 12920,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/val/test splits at the patient level.
        
        Args:
            df: Full dataset DataFrame
            seed: Random seed for reproducibility
            test_patients: Number of patients for test set (~10k images)
            val_patients: Number of patients for val set (~15k images)
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Creating train/val/test splits...")
        
        # Get unique patients and shuffle
        patients = df["Patient ID"].unique()
        np.random.seed(seed)
        np.random.shuffle(patients)
        
        # Split patients
        test_patient_ids = patients[:test_patients]
        remaining_patients = patients[test_patients:]
        
        # Shuffle remaining for val/train split
        np.random.seed(seed)
        np.random.shuffle(remaining_patients)
        val_patient_ids = remaining_patients[:val_patients]
        train_patient_ids = remaining_patients[val_patients:]
        
        # Create splits
        test_df = df[df["Patient ID"].isin(test_patient_ids)].copy()
        val_df = df[df["Patient ID"].isin(val_patient_ids)].copy()
        train_df = df[df["Patient ID"].isin(train_patient_ids)].copy()
        
        logger.info(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def run(self, seed: int = 42) -> dict:
        """
        Run the full preprocessing pipeline.
        
        Args:
            seed: Random seed for splits
        
        Returns:
            Dict with paths to output files
        """
        logger.info("=" * 60)
        logger.info("PadChest Preprocessing Pipeline")
        logger.info("=" * 60)
        
        # Load and clean metadata
        df = self.load_metadata()
        
        # Preprocess images
        df = self.preprocess_images(df)
        
        # Create splits
        train_df, val_df, test_df = self.create_splits(df, seed=seed)
        
        # Save CSVs
        train_path = self.save_csv(train_df, f"{self.DATASET_NAME}_train.csv")
        val_path = self.save_csv(val_df, f"{self.DATASET_NAME}_val.csv")
        test_path = self.save_csv(test_df, f"{self.DATASET_NAME}_test.csv")
        
        # Save full processed dataset
        full_path = self.save_csv(df, f"{self.DATASET_NAME}_processed.csv")
        
        logger.info("PadChest preprocessing complete!")
        
        return {
            "train": train_path,
            "val": val_path,
            "test": test_path,
            "full": full_path,
        }
