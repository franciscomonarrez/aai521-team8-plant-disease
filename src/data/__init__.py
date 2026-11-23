"""Data download, splits, and PyTorch datasets for plant-disease images.

Importing this package does not import torch. Modeling code should import
from ``src.data.dataset`` (or use the lazy names below).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CLASS_TO_IDX",
    "PLANTVILLAGE_CLASSES",
    "PlantDiseaseDataset",
    "build_transforms",
    "load_plantdoc",
    "load_splits",
    "make_dataloader",
]


def __getattr__(name: str) -> Any:
    if name in {"CLASS_TO_IDX", "PLANTVILLAGE_CLASSES"}:
        from src.data import labels

        return getattr(labels, name)
    if name in {
        "PlantDiseaseDataset",
        "build_transforms",
        "load_plantdoc",
        "load_splits",
        "make_dataloader",
    }:
        from src.data import dataset

        return getattr(dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
