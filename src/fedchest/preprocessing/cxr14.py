"""
NIH ChestX-ray14 (CXR14) dataset preprocessor.

Source: https://nihcc.app.box.com/v/ChestXray-NIHCC

Original dataset structure:
- images/: Contains all PNG images (1024x1024)
- Data_Entry_2017_v2020.csv: Metadata with labels
- train_val_list.txt: Official train/val image list
- test_list.txt: Official test image list

Labels are pipe-separated strings (e.g., "Atelectasis|Effusion")
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from fedchest.preprocessing.base import BasePreprocessor
from fedchest.preprocessing.image_transforms import preprocess_chest_xray

logger = logging.getLogger(__name__)

with open(Path(__file__).parent.parent.parent / "configs" / "preprocessing.yaml", "r") as f:
    PREPROCESSING_CONFIGS = yaml.safe_load(f)
LABEL_ALIASES = PREPROCESSING_CONFIGS["cxr14"]["label_aliases"]


class CXR14Preprocessor(BasePreprocessor):
    """
    Preprocessor for NIH ChestX-ray14 dataset.
    
    Handles:
    - Loading pipe-separated labels from CSV
    - Filtering to common diseases
    - Removing duplicate patient/projection/label combinations
    - Preprocessing images with CLAHE
    """
    
    DATASET_NAME = "cxr14"
    
    def __init__(
        self,
        source_dir: Path,
        output_dir: Path,
        config_path: Optional[Path] = None,
    ):
        super().__init__(source_dir, output_dir, config_path)
        
        # Expected source files
        self.metadata_file = self.source_dir / "Data_Entry_2017_v2020.csv"
        self.images_source_dir = self.source_dir / "images"
    
    def load_metadata(self) -> pd.DataFrame:
        """
        Load and parse CXR14 metadata.
        
        Returns:
            DataFrame with standardized columns
        """
        logger.info(f"Loading CXR14 metadata from {self.metadata_file}")
        
        # Load CSV
        df = pd.read_csv(
            self.metadata_file,
            usecols=["Image Index", "Patient ID", "View Position", "Finding Labels"],
        )
        
        logger.info(f"Loaded {len(df)} rows")
        
        # Parse pipe-separated labels into one-hot encoding
        labels = df["Finding Labels"].str.get_dummies(sep="|").astype(np.float32)
        df = pd.concat([df, labels], axis=1)
        df.drop(columns=["Finding Labels"], inplace=True)
        
        # Filter to common diseases
        df = self.filter_common_diseases(df)
        
        # Drop columns that aren't common diseases
        extra_cols = [c for c in df.columns 
                     if c not in ["Image Index", "Patient ID", "View Position"] + self.common_diseases]
        if extra_cols:
            logger.info(f"Dropping non-common disease columns: {extra_cols}")
            df.drop(columns=extra_cols, inplace=True)
        
        # Remove duplicates (same patient, projection, and labels)
        dup_subset = ["Patient ID", "View Position"] + self.common_diseases
        df = self.remove_duplicates(df, dup_subset)
        
        # Standardize column names
        df = self.standardize_columns(df)
        
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
        # Keep original filename
        return self.images_dir / self.DATASET_NAME / row["Path"]
    
    def preprocess_images(self, df: pd.DataFrame, num_workers: int = 1) -> pd.DataFrame:
        """
        Preprocess all images in the dataset.
        
        Args:
            df: DataFrame with Path column
            num_workers: Number of parallel workers (not implemented yet)
        
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
        
        # Update paths to be relative to images directory
        df["Path"] = df["Path"].apply(lambda p: f"{self.DATASET_NAME}/{p}")
        
        logger.info(f"Successfully processed {len(df)} images")
        return df
    
    def create_splits(
        self,
        df: pd.DataFrame,
        seed: int = 42,
        test_patients: int = 5711,
        val_patients: int = 8470,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/val/test splits at the patient level.
        
        IMPORTANT: All images from a patient go to the same split
        to prevent data leakage.
        
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
        val_patient_ids = patients[test_patients:test_patients + val_patients]
        train_patient_ids = patients[test_patients + val_patients:]
        
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
        logger.info("CXR14 Preprocessing Pipeline")
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
        
        logger.info("CXR14 preprocessing complete!")
        
        return {
            "train": train_path,
            "val": val_path,
            "test": test_path,
            "full": full_path,
        }
