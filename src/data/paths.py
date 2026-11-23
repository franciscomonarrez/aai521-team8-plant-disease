"""Shared filesystem locations for the data pipeline."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PLANTVILLAGE_DIR = RAW_DIR / "plantvillage"
PLANTVILLAGE_COLOR_DIR = PLANTVILLAGE_DIR / "color"
PLANTDOC_DIR = RAW_DIR / "plantdoc"
# Leaf IDs from the original PlantVillage release. Kept outside
# PLANTVILLAGE_DIR so ``download --force`` does not delete it.
LEAF_MAP_PATH = RAW_DIR / "plantvillage-leaf-map.json"
SPLITS_DIR = DATA_DIR / "splits"
LABEL_MAP_PATH = DATA_DIR / "plantdoc_label_map.csv"
DOWNLOAD_DIR = RAW_DIR / "_downloads"

# Suffixes accepted when walking class folders. Comparison is case-insensitive.
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
    ".jfif",
}
