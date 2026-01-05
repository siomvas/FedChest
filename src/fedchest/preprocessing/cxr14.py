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
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from fedchest.preprocessing.base import BasePreprocessor
from fedchest.preprocessing.image_transforms import preprocess_chest_xray

logger = logging.getLogger(__name__)

ALIASES = {
    "Mass": "Lung Lesion",
    "Nodule": "Lung Lesion",
    "Infiltration": "Lung Opacity",
    "Fibrosis": "Lung Opacity",
}

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

        self.metadata_file = self.source_dir / "Data_Entry_2017_v2020.csv"
        self.images_source_dir = self.source_dir / "images"

        self.interim_dir = self.output_dir / "interim_splits"
        self.interim_dir.mkdir(parents=True, exist_ok=True)
    
    def load_and_clean_metadata(self) -> pd.DataFrame:
        """
        Load, parse, clean, apply mappings, and filter CXR14 metadata.
        
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
        # Make extra column for the merged labels (Lung Lesion, Lung Opacity)
        for original, alias in ALIASES.items():
            if original in df.columns:
                if alias not in df.columns:
                    df[alias] = 0.0
                df[alias] = np.maximum(df[alias], df[original])
        
        df.drop(columns=["Finding Labels"], inplace=True)
        
        # Filter to rows with common diseases
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
    
    def prepare_splits(self, seed: int = 42, test_ratio: float=0.1, val_ratio: float=0.15):
        """
        Prepare interim splits without image processing.
        
        Args:
            seed: Random seed for reproducibility
            test_ratio: Proportion of data for test set
            val_ratio: Proportion of data for validation set
        """
        logger.info("=" * 40)
        logger.info("STAGE 1: Preparing Splits (Metadata Only)")
        logger.info("=" * 40)

        df = self.load_and_clean_metadata()
        
        # Create splits
        patients = df["Patient ID"].unique()
        np.random.seed(seed)
        np.random.shuffle(patients)

        test_patients = int(len(patients) * test_ratio)
        val_patients = int(len(patients) * val_ratio)
        
        # Split patients
        test_patient_ids = patients[:test_patients]
        val_patient_ids = patients[test_patients:test_patients + val_patients]
        train_patient_ids = patients[test_patients + val_patients:]
        
        # Create splits
        test_df = df[df["Patient ID"].isin(test_patient_ids)].copy()
        val_df = df[df["Patient ID"].isin(val_patient_ids)].copy()
        train_df = df[df["Patient ID"].isin(train_patient_ids)].copy()
        
        logger.info(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

        # Save interim CSVs
        train_path = self.interim_dir / "cxr14_train.csv"
        val_path = self.interim_dir / "cxr14_val.csv"
        test_path = self.interim_dir / "cxr14_test.csv"
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        return True
    
    def process_images(self):
        """
        Process all images in the dataset.
        """
        logger.info("=" * 40)
        logger.info("STAGE 2: Processing Images")
        logger.info("=" * 40)

        splits = list(self.interim_dir.glob("cxr14_*.csv"))
        if not splits:
            logger.error("No interim split CSVs for CXR14 found. Please run prepare_splits() first.")
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
        logger.info("CXR14 Preprocessing Pipeline")
        logger.info("=" * 60)
        
        if stage in ["prepare", "all"]:
            success = self.prepare_splits(seed=seed)
            if not success:
                logger.error("Failed to prepare splits. Aborting.")
                return {}
        if stage in ["process", "all"]:
            return self.process_images()
        
        logger.info("CXR14 preprocessing complete!")
        return {}
