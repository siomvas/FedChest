"""
CheXpert dataset preprocessor.

Source: https://stanfordmlgroup.github.io/competitions/chexpert/

Original dataset structure:
- CheXpert-v1.0/train/: Training images organized by patient/study
- train.csv: Metadata with labels

Labels have uncertainty (-1) which we treat as negative (0) per original paper.
Dataset is split into two clients based on patient age (threshold: 63 years).

"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from fedchest.preprocessing.base import BasePreprocessor
from fedchest.preprocessing.image_transforms import preprocess_chest_xray

logger = logging.getLogger(__name__)


class CheXpertPreprocessor(BasePreprocessor):
    """
    Preprocessor for CheXpert dataset.
    
    Special handling:
    - Uncertainty labels (-1) converted to 0
    - Split into young (<63) and old (>=63) age groups
    - Patients with images in both age groups assigned to young client
    - Lateral views filtered out (frontal only)
    """
    
    DATASET_NAME = "chexpert"
    AGE_THRESHOLD = 63  # Median age for split
    
    def __init__(
        self,
        source_dir: Path,
        output_dir: Path,
        config_path: Optional[Path] = None,
    ):
        super().__init__(source_dir, output_dir, config_path)
        
        # Expected source files
        self.metadata_file = self.source_dir / "train.csv"
        self.images_source_dir = self.source_dir / "CheXpert-v1.0" / "train"
    
    def load_metadata(self) -> pd.DataFrame:
        """
        Load and parse CheXpert metadata.
        
        Returns:
            DataFrame with standardized columns
        """
        logger.info(f"Loading CheXpert metadata from {self.metadata_file}")
        
        # Load CSV
        df = pd.read_csv(self.metadata_file)
        
        logger.info(f"Loaded {len(df)} rows")
        
        # Drop unnecessary columns
        cols_to_drop = ["Sex", "Support Devices"]
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
        
        # Filter to frontal views only
        if "Frontal/Lateral" in df.columns:
            df = df[df["Frontal/Lateral"] != "Lateral"].copy()
            df.drop(columns=["Frontal/Lateral"], inplace=True)
            logger.info(f"After filtering lateral views: {len(df)} rows")
        
        # Handle uncertainty labels: -1 -> 0 (per CheXpert paper)
        label_cols = [c for c in df.columns if c not in ["Path", "Age", "AP/PA"]]
        df[label_cols] = df[label_cols].fillna(0.0).replace(-1.0, 0.0)
        
        # Rename Pleural Effusion to Effusion for consistency
        if "Pleural Effusion" in df.columns:
            df.rename(columns={"Pleural Effusion": "Effusion"}, inplace=True)
        
        # Filter to common diseases
        df = self.filter_common_diseases(df)
        
        # Drop uncommon disease columns
        uncommon = ["Lung Opacity", "Enlarged Cardiomediastinum", "Pleural Other", "Fracture"]
        df.drop(columns=[c for c in uncommon if c in df.columns], inplace=True)
        
        # Extract patient ID from path
        # Path format: patient00001/study1/view1_frontal.jpg
        df["Patient ID"] = df["Path"].str.extract(r"patient(\d+)")[0]
        
        # Simplify path for output (remove CheXpert directory structure)
        df["Path"] = df["Path"].str.replace("/", "_")
        df["Path"] = df["Path"].str.split("_").str[2:].str.join("_")
        
        # Remove duplicates
        dup_subset = ["Patient ID", "AP/PA"] + [c for c in self.common_diseases if c in df.columns]
        df = self.remove_duplicates(df, dup_subset)
        
        # Standardize column names
        df = self.standardize_columns(df)
        
        # Validate
        self.validate_labels(df)
        
        # Log distribution
        dist = self.get_label_distribution(df)
        logger.info(f"Label distribution: {dist}")
        
        return df
    
    def load_with_age(self) -> pd.DataFrame:
        """
        Load metadata including age column for splitting.
        
        Returns:
            DataFrame with Age column preserved
        """
        logger.info(f"Loading CheXpert metadata with age from {self.metadata_file}")
        
        # Load with Age column
        df_original = pd.read_csv(self.metadata_file, usecols=["Path", "Age"])
        df_original["Path"] = df_original["Path"].str.replace("CheXpert-v1.0/train/", "")
        df_original["Path"] = df_original["Path"].str.replace("/", "_")
        
        # Get processed metadata
        df = self.load_metadata()
        
        # Merge age back in
        df = df.merge(df_original, on="Path", how="left")
        
        return df
    
    def get_source_image_path(self, row: pd.Series) -> Path:
        """Get path to source image for a row."""
        # Convert simplified path back to original structure
        path_parts = row["Path"].replace(".jpg", "").split("_")
        patient = path_parts[0]
        study = path_parts[1]
        view = "_".join(path_parts[2:]) + ".jpg"
        
        return self.images_source_dir / patient / study / view
    
    def get_output_image_path(self, row: pd.Series, client: str) -> Path:
        """Get path for processed output image."""
        return self.images_dir / client / row["Path"]
    
    def preprocess_images(
        self,
        df: pd.DataFrame,
        client: str,
    ) -> pd.DataFrame:
        """
        Preprocess all images for a specific client.
        
        Args:
            df: DataFrame with Path column
            client: Client name (cxp_young or cxp_old)
        
        Returns:
            DataFrame with updated paths (failed images removed)
        """
        logger.info(f"Preprocessing {len(df)} images for {client}...")
        
        # Create output subdirectory
        output_subdir = self.images_dir / client
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        failed_indices = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {client}"):
            source_path = self.get_source_image_path(row)
            output_path = self.get_output_image_path(row, client)
            
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
        df["Path"] = df["Path"].apply(lambda p: f"{client}/{p}")
        
        logger.info(f"Successfully processed {len(df)} images for {client}")
        return df
    
    def split_by_age(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset by patient age.
        
        Patients with images spanning the age threshold are assigned
        to the young group to maintain patient-level separation.
        
        Args:
            df: DataFrame with Age column
        
        Returns:
            Tuple of (young_df, old_df)
        """
        logger.info(f"Splitting by age (threshold: {self.AGE_THRESHOLD})...")
        
        # Initial split by age
        young_df = df[df["Age"] < self.AGE_THRESHOLD].copy()
        old_df = df[df["Age"] >= self.AGE_THRESHOLD].copy()
        
        # Find patients that appear in both groups
        young_patients = set(young_df["Patient ID"])
        old_patients = set(old_df["Patient ID"])
        common_patients = young_patients & old_patients
        
        if common_patients:
            logger.info(f"Found {len(common_patients)} patients spanning age threshold")
            
            # Move all images from common patients to young group
            common_old_images = old_df[old_df["Patient ID"].isin(common_patients)]
            young_df = pd.concat([young_df, common_old_images], ignore_index=True)
            old_df = old_df[~old_df["Patient ID"].isin(common_patients)]
            
            logger.info(f"Moved {len(common_old_images)} images to young group")
        
        # Verify no overlap
        assert len(set(young_df["Patient ID"]) & set(old_df["Patient ID"])) == 0
        
        logger.info(f"Split result: young={len(young_df)}, old={len(old_df)}")
        
        return young_df, old_df
    
    def create_test_split(
        self,
        df: pd.DataFrame,
        seed: int = 42,
        test_patients: int = 7356,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create test split from full dataset (before age split).
        
        Args:
            df: Full dataset with Age column
            seed: Random seed
            test_patients: Number of patients for test set
        
        Returns:
            Tuple of (train_val_df, test_df)
        """
        logger.info("Creating test split...")
        
        patients = df["Patient ID"].unique()
        np.random.seed(seed)
        np.random.shuffle(patients)
        
        test_patient_ids = patients[:test_patients]
        
        test_df = df[df["Patient ID"].isin(test_patient_ids)].copy()
        train_val_df = df[~df["Patient ID"].isin(test_patient_ids)].copy()
        
        logger.info(f"Test split: train_val={len(train_val_df)}, test={len(test_df)}")
        
        return train_val_df, test_df
    
    def create_val_split(
        self,
        df: pd.DataFrame,
        seed: int = 42,
        val_patients: int = 7489,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create validation split from a subset.
        
        Args:
            df: Dataset subset (young or old)
            seed: Random seed
            val_patients: Number of patients for validation
        
        Returns:
            Tuple of (train_df, val_df)
        """
        patients = df["Patient ID"].unique()
        np.random.seed(seed)
        np.random.shuffle(patients)
        
        val_patient_ids = patients[:val_patients]
        
        val_df = df[df["Patient ID"].isin(val_patient_ids)].copy()
        train_df = df[~df["Patient ID"].isin(val_patient_ids)].copy()
        
        return train_df, val_df
    
    def run(self, seed: int = 42) -> dict:
        """
        Run the full preprocessing pipeline.
        
        Args:
            seed: Random seed for splits (standardized to 42, was 44)
        
        Returns:
            Dict with paths to output files for both clients
        """
        logger.info("=" * 60)
        logger.info("CheXpert Preprocessing Pipeline")
        logger.info("=" * 60)
        
        # Load metadata with age
        df = self.load_with_age()
        
        # Create test split first (before age split)
        train_val_df, test_df = self.create_test_split(df, seed=seed)
        
        # Split train_val by age
        young_train_val, old_train_val = self.split_by_age(train_val_df)
        
        # Split test by age (same threshold)
        young_test, old_test = self.split_by_age(test_df)
        
        # Create train/val splits for each age group
        # Young: ~7489 patients for val
        young_train, young_val = self.create_val_split(young_train_val, seed=seed, val_patients=7489)
        # Old: ~7264 patients for val
        old_train, old_val = self.create_val_split(old_train_val, seed=seed, val_patients=7264)
        
        # Preprocess images for each client
        logger.info("\n--- Processing cxp_young ---")
        young_train = self.preprocess_images(young_train, "cxp_young")
        young_val = self.preprocess_images(young_val, "cxp_young")
        young_test = self.preprocess_images(young_test, "cxp_young")
        
        logger.info("\n--- Processing cxp_old ---")
        old_train = self.preprocess_images(old_train, "cxp_old")
        old_val = self.preprocess_images(old_val, "cxp_old")
        old_test = self.preprocess_images(old_test, "cxp_old")
        
        # Drop Age column before saving
        for df_split in [young_train, young_val, young_test, old_train, old_val, old_test]:
            if "Age" in df_split.columns:
                df_split.drop(columns=["Age"], inplace=True)
        
        # Save CSVs
        results = {}
        
        # Young client
        results["cxp_young"] = {
            "train": self.save_csv(young_train, "cxp_young_train.csv"),
            "val": self.save_csv(young_val, "cxp_young_val.csv"),
            "test": self.save_csv(young_test, "cxp_young_test.csv"),
        }
        
        # Old client
        results["cxp_old"] = {
            "train": self.save_csv(old_train, "cxp_old_train.csv"),
            "val": self.save_csv(old_val, "cxp_old_val.csv"),
            "test": self.save_csv(old_test, "cxp_old_test.csv"),
        }
        
        logger.info("CheXpert preprocessing complete!")
        
        return results
