"""
Path utilities for FedChest.
This module provides functions to handle and manipulate file paths,
especially for image datasets and datalist entries.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union


def get_project_root() -> Path:
    """Get the project root directory."""
    # Navigate up from this file to find project root
    current = Path(__file__).resolve()
    
    # Look for pyproject.toml as marker
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    
    # Fallback to current working directory
    return Path.cwd()


def resolve_image_path(
    relative_path: str,
    data_root: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Resolve an image path from a datalist entry.
    
    Args:
        relative_path: Relative path stored in datalist JSON
        data_root: Root directory for data (if None, uses cwd)
    
    Returns:
        Absolute path to the image
    """
    path = Path(relative_path)
    
    if path.is_absolute():
        return path
    
    if data_root:
        return Path(data_root) / path
    
    return Path.cwd() / path


def create_datalist_entry(
    image_path: Union[str, Path],
    label: List[float],
    data_root: Optional[Union[str, Path]] = None,
) -> Dict:
    """
    Create a datalist entry with a relative path.
    
    Args:
        image_path: Full path to the image
        label: Label vector
        data_root: Root directory to make path relative to
    
    Returns:
        Dict with 'image' (relative path string) and 'label' keys
    """
    image_path = Path(image_path)
    
    if data_root:
        data_root = Path(data_root)
        try:
            # Make path relative to data_root
            relative_path = image_path.relative_to(data_root)
        except ValueError:
            # Path is not under data_root, use as-is
            relative_path = image_path
    else:
        relative_path = image_path
    
    return {
        "image": str(relative_path),
        "label": label,
    }


def validate_datalist_paths(
    datalist: List[Dict],
    data_root: Union[str, Path],
    check_exists: bool = True,
) -> Dict[str, int]:
    """
    Validate that all paths in a datalist are accessible.
    
    Args:
        datalist: List of datalist entries
        data_root: Root directory for resolving paths
        check_exists: If True, check files actually exist on disk
    
    Returns:
        Dict with counts of 'valid', 'missing', 'invalid' entries
    """
    data_root = Path(data_root)
    stats = {"valid": 0, "missing": 0, "invalid": 0}
    
    for entry in datalist:
        if "image" not in entry:
            stats["invalid"] += 1
            continue
        
        if check_exists:
            full_path = resolve_image_path(entry["image"], data_root)
            if full_path.exists():
                stats["valid"] += 1
            else:
                stats["missing"] += 1
        else:
            stats["valid"] += 1
    
    return stats
