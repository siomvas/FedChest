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
import shutil
from pathlib import Path
from typing import Optional

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
    
    def __init__(
        self,
        source_dir: Path,
        output_dir: Path,
        config_path: Optional[Path] = None,
    ):
        super().__init__(source_dir, output_dir, config_path)

        self.metadata_file = self.source_dir / "train.csv"
        self.images_source_dir = self.source_dir / "train"

        self.interim_dir = self.output_dir / "interim_splits"
        self.interim_dir.mkdir(parents=True, exist_ok=True)
    
    def load_and_clean_metadata(self) -> pd.DataFrame:
        """
        Load, clean labels, and filter common diseases.
        
        Returns:
            DataFrame with Age and Patient ID.
        """
        logger.info(f"Loading CheXpert metadata from {self.metadata_file}")
        
        # Load CSV (with age)
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
        
        # Drop uncommon disease columns
        uncommon = ["Enlarged Cardiomediastinum", "Pleural Other", "Fracture"]
        df.drop(columns=[c for c in uncommon if c in df.columns], inplace=True)

        # Filter to common diseases
        df = self.filter_common_diseases(df)
        
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

        # Ensure Age is present (might have been dropped by standardize if not in config)
        if "Age" not in df.columns:
            # Re-merge Age from original if lost
            df_age = pd.read_csv(self.metadata_file, usecols=["Path", "Age"])
            df_age["Path"] = df_age["Path"].str.replace("CheXpert-v1.0/train/", "", regex=False)
            df_age["Path"] = df_age["Path"].str.replace("/", "_")
            df = df.merge(df_age[["Path", "Age"]], on="Path", how="inner")
        
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

        median_age = df["Age"].median()
        logger.info(f"Median age in dataset: {median_age}")

        patients = df["Patient ID"].unique()
        np.random.seed(seed)
        np.random.shuffle(patients)

        test_size = int(len(patients) * test_ratio)
        test_patients = patients[:test_size]
        train_val_patients = patients[test_size:]
        df_test = df[df["Patient ID"].isin(test_patients)].copy()
        df_train_val = df[df["Patient ID"].isin(train_val_patients)].copy()

        logger.info(f"Global Split: Test Patients={len(test_patients)}, Train/Val Patients={len(train_val_patients)}")

        # Split by age
        # Patients straddling the boundary go to Young to prevent leakage
        young_df = df_train_val[df_train_val["Age"] < median_age].copy()
        old_df = df_train_val[df_train_val["Age"] >= median_age].copy()
        young_patients = set(young_df["Patient ID"])
        old_patients = set(old_df["Patient ID"])
        common_patients = young_patients & old_patients
        if common_patients:
            logger.info(f"Found {len(common_patients)} patients spanning age threshold")
            common_old_images = old_df[old_df["Patient ID"].isin(common_patients)]
            young_df = pd.concat([young_df, common_old_images], ignore_index=True)
            old_df = old_df[~old_df["Patient ID"].isin(common_patients)]
            logger.info(f"Moved {len(common_old_images)} images to young group")
        
        young_test = df_test[df_test["Age"] < median_age].copy()
        old_test = df_test[df_test["Age"] >= median_age].copy()

        def create_client_split(client_df, client_name):
            patients = client_df["Patient ID"].unique()
            np.random.seed(seed)
            np.random.shuffle(patients)
            val_size = int(len(patients) * val_ratio)

            val_patients = patients[:val_size]
            train_patients = patients[val_size:]

            val_df = client_df[client_df["Patient ID"].isin(val_patients)].copy()
            train_df = client_df[client_df["Patient ID"].isin(train_patients)].copy()

            logger.info(f"{client_name} Split: Train Patients={len(train_patients)}, Val Patients={len(val_patients)}")
            
            return train_df, val_df
        
        young_train, young_val = create_client_split(young_df, "cxp_young")
        old_train, old_val = create_client_split(old_df, "cxp_old")

        # Save interim CSVs
        splits = {
            "cxp_young_train": young_train,
            "cxp_young_val": young_val,
            "cxp_young_test": young_test,
            "cxp_old_train": old_train,
            "cxp_old_val": old_val,
            "cxp_old_test": old_test,
        }

        for split_name, split_df in splits.items():
            split_file = self.interim_dir / f"{split_name}.csv"
            split_df.to_csv(split_file, index=False)
            logger.info(f"Saved {split_name} with {len(split_df)} rows to {split_file}")
        
        return True
    
    def process_images(self):
        """
        Process all images.
        """
        logger.info("=" * 40)
        logger.info("STAGE 2: Processing Images")
        logger.info("=" * 40)

        splits = list(self.interim_dir.glob("cxp*.csv"))
        if not splits:
            logger.error("No interim split CSVs found. Please run prepare_splits() first.")
            return {}
        
        results = {}

        for split_file in splits:
            split_name = split_file.stem
            client = "cxp_young" if "young" in split_name else "cxp_old"
            logger.info(f"\n--- Processing {split_name} for {client} ---")
            df = pd.read_csv(split_file)

            # create output subdirectory
            client_output_dir = self.images_dir / client
            client_output_dir.mkdir(parents=True, exist_ok=True)

            valid_indices = []
        
            for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {client}"):
                # Reconstruct source path
                # Path in CSV is "patient001_study1_view1.jpg"
                # Actual source is "train/patient001/study1/view1.jpg"
                parts = row["Path"].split("_")
                patient, study = parts[0], parts[1]
                filename = "_".join(parts[2:])
                
                source_path = self.images_source_dir / patient / study / filename
                output_path = client_output_dir / row["Path"]
            
                if preprocess_chest_xray(
                    image_path=source_path,
                    output_path=output_path,
                    size=self.image_size,
                    crop_percent=self.crop_percent,
                    clip_limit=self.clip_limit,
                    tile_grid_size=self.tile_grid_size,
                ):
                    valid_indices.append(idx)
        
            # Remove failed images
            df_final = df.loc[valid_indices].copy()
        
            # Update paths to be relative
            df_final["Path"] = df_final["Path"].apply(lambda p: f"{client}/{p}")

            if "Age" in df_final.columns:
                df_final.drop(columns=["Age"], inplace=True)
            
            final_csv_name = f"{split_name}.csv"
            self.save_csv(df_final, final_csv_name)
            results[split_name] = final_csv_name
        
        # shutil.rmtree(self.interim_dir)
        
        return results
    
    def run(self, seed: int = 42, stage: str = "all") -> dict:
        """
        Run the preprocessing pipeline.
        
        Args:
            seed: Random seed for splits
            stage: Stage to run ("prepare", "process", or "all")
        
        Returns:
            Dict with paths to output files for both clients
        """
        logger.info("=" * 60)
        logger.info("CheXpert Preprocessing Pipeline")
        logger.info("=" * 60)
        
        if stage in ["prepare", "all"]:
            success = self.prepare_splits(seed=seed)
            if not success:
                logger.error("Failed to prepare splits.")
                return {}
        if stage in ["process", "all"]:
            return self.process_images()
        
        logger.info("CheXpert preprocessing complete!")
        return {}
