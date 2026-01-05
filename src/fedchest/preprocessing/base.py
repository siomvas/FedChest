"""
Base preprocessor class for FedChest datasets.

Provides common functionality for all dataset preprocessors.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

class BasePreprocessor(ABC):
    """
    Abstract base class for dataset preprocessors.
    
    Each dataset (CXR14, CheXpert, PadChest) has its own preprocessor
    that inherits from this class and implements dataset-specific logic.
    """
    
    def __init__(
        self,
        source_dir: Path,
        output_dir: Path,
        config_path: Optional[Path] = None,
    ):
        """
        Args:
            source_dir: Directory containing raw dataset files
            output_dir: Directory to save processed files
            config_path: Path to preprocessing config YAML
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.config = self._load_config(config_path)
        
        # Create output directories
        self.images_dir = self.output_dir / "images"
        self.csv_dir = self.output_dir / "csv"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self, config_path: Optional[Path]) -> Dict:
        """Load preprocessing configuration."""
        if config_path and config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
    
    @property
    def image_size(self) -> Tuple[int, int]:
        """Target image size."""
        size = self.config["image"]["size"]
        return (size, size)
    
    @property
    def crop_percent(self) -> float:
        """Border crop percentage."""
        return self.config["image"]["crop_border_percent"]
    
    @property
    def clip_limit(self) -> float:
        """CLAHE clip limit."""
        return self.config["clahe"]["clip_limit"]
    
    @property
    def tile_grid_size(self) -> Tuple[int, int]:
        """CLAHE tile grid size."""
        return tuple(self.config["clahe"]["tile_grid_size"])
    
    @property
    def common_diseases(self) -> List[str]:
        """List of common disease labels."""
        try:
            return self.config["common_diseases"]
        except KeyError:
            raise KeyError("common_diseases not defined in config")
    
    @abstractmethod
    def load_and_clean_metadata(self) -> pd.DataFrame:
        """
        Load and parse dataset metadata/labels.
        
        Returns:
            DataFrame with columns: Path, Patient ID, [disease columns]
        """
        pass
    
    def filter_common_diseases(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter dataset to only include samples with common diseases.
        
        Drops samples where none of the common diseases are present.
        
        Args:
            df: DataFrame with disease label columns
        
        Returns:
            Filtered DataFrame
        """
        # Check which common diseases are present as columns
        available_diseases = [d for d in self.common_diseases if d in df.columns]
        
        if not len(available_diseases) == 10:
            raise ValueError(f"common disease columns incorrent. Available: {available_diseases}")
        
        # Keep rows where at least one common disease is present
        mask = (df[available_diseases] != 0).any(axis=1)
        
        original_count = len(df)
        filtered_df = df[mask].copy()
        dropped_count = original_count - len(filtered_df)
        
        logger.info(f"Filtered to common diseases: {original_count} -> {len(filtered_df)} "
                   f"(dropped {dropped_count})")
        
        return filtered_df
    
    def remove_duplicates(
        self,
        df: pd.DataFrame,
        subset_columns: List[str],
    ) -> pd.DataFrame:
        """
        Remove duplicate rows based on specified columns.
        
        Duplicates are defined as rows where patient ID, projection,
        and all disease labels are identical.
        
        Args:
            df: Input DataFrame
            subset_columns: Columns to consider for duplicates
        
        Returns:
            DataFrame with duplicates removed
        """
        original_count = len(df)
        df_dedup = df.drop_duplicates(subset=subset_columns, keep="first")
        dropped_count = original_count - len(df_dedup)
        
        logger.info(f"Removed duplicates: {original_count} -> {len(df_dedup)} "
                   f"(dropped {dropped_count})")
        
        return df_dedup
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names and order.
        
        Ensures consistent column naming across datasets:
        - Path, Patient ID, Projection, [common diseases in order]
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with standardized columns
        """
        # Rename common variations
        rename_map = {
            "Image Index": "Path",
            "ImageID": "Path",
            "Imageid": "Path",
            "PatientID": "Patient ID",
            "Patientid": "Patient ID",
            "View Position": "Projection",
            "AP/PA": "Projection",
        }
        
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        
        # Ensure disease columns exist and are in correct order
        for disease in self.common_diseases:
            if disease not in df.columns:
                logger.warning(f"Disease column '{disease}' not found, adding zeros")
                df[disease] = 0.0
        
        # Select and order columns
        base_cols = ["Path", "Patient ID"]
        if "Projection" in df.columns:
            base_cols.append("Projection")
        
        final_cols = base_cols + self.common_diseases
        available_cols = [c for c in final_cols if c in df.columns]
        
        return df[available_cols].copy()
    
    def validate_labels(self, df: pd.DataFrame) -> bool:
        """
        Validate that labels are consistent.
        
        Checks:
        - No Finding should be 1 if all diseases are 0
        - No NaN values in labels
        
        Args:
            df: DataFrame with disease label columns
        
        Returns:
            True if valid, raises AssertionError otherwise
        """
        diseases_only = [d for d in self.common_diseases if d != "No Finding"]
        
        # Check No Finding consistency
        if "No Finding" in df.columns:
            all_diseases_zero = (df[diseases_only] == 0).all(axis=1)
            no_finding_is_one = df["No Finding"] == 1
            
            # Where all diseases are 0, No Finding should be 1
            inconsistent = all_diseases_zero & ~no_finding_is_one
            if inconsistent.any():
                count = inconsistent.sum()
                logger.warning(f"Found {count} rows where all diseases=0 but No Finding!=1")
        
        # Check for NaN
        label_cols = [d for d in self.common_diseases if d in df.columns]
        nan_count = df[label_cols].isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in labels")
        
        return True
    
    def get_label_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Get the distribution of labels in the dataset.
        
        Args:
            df: DataFrame with disease label columns
        
        Returns:
            Dict mapping disease name to count
        """
        label_cols = [d for d in self.common_diseases if d in df.columns]
        return df[label_cols].sum().to_dict()
    
    def save_csv(self, df: pd.DataFrame, filename: str) -> Path:
        """
        Save DataFrame to CSV.
        
        Args:
            df: DataFrame to save
            filename: Output filename (without directory)
        
        Returns:
            Path to saved file
        """
        output_path = self.csv_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Saved: {output_path} ({len(df)} rows)")
        return output_path
