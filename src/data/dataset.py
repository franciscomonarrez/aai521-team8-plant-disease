"""PyTorch dataset and loaders for the plant-disease CSVs.

Images are 224x224 with ImageNet normalization. The train pipeline randomly
crops, flips, rotates, and jitters color. Val, test, and PlantDoc use a
resize to the shorter side of 256 and a center crop, which keeps aspect ratio.

Labels are integer indexes into ``PLANTVILLAGE_CLASSES`` (alphabetical). A
PlantDoc row uses the same index as the PlantVillage class it was mapped to,
even when that CSV does not contain every class.

Usage::

    from src.data.dataset import load_plantdoc, load_splits

    loaders = load_splits(batch_size=32, num_workers=4)
    images, labels = next(iter(loaders["train"]))
    plantdoc = load_plantdoc(batch_size=32)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.data.labels import CLASS_TO_IDX, IDX_TO_CLASS, NUM_CLASSES, PLANTVILLAGE_CLASSES
from src.data.paths import PROJECT_ROOT, SPLITS_DIR

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224

_TRAIN_NAMES = {"train", "training"}
_EVAL_NAMES = {"val", "valid", "validation", "test", "eval", "plantdoc"}


def build_transforms(split: str | bool = "train") -> transforms.Compose:
    """Build the train or eval transform.

    ``split`` may be ``"train"``, ``"val"``, ``"test"``, ``"plantdoc"``, or a
    boolean (True means train).
    """
    if isinstance(split, bool):
        train = split
    else:
        key = str(split).lower()
        if key in _TRAIN_NAMES:
            train = True
        elif key in _EVAL_NAMES:
            train = False
        else:
            allowed = ", ".join(sorted(_TRAIN_NAMES | _EVAL_NAMES))
            raise ValueError(f"Unknown split {split!r}. Expected one of: {allowed}")

    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            normalize,
        ]
    )


class PlantDiseaseDataset(Dataset):
    """CSV dataset of ``path,label`` rows.

    ``path`` may be relative to ``root`` (the repository root by default) or
    absolute. ``label`` is a PlantVillage class folder name.
    """

    def __init__(
        self,
        csv_path: str | Path,
        transform: transforms.Compose | None = None,
        class_to_idx: dict[str, int] | None = None,
        root: str | Path | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.root = Path(root) if root is not None else PROJECT_ROOT
        self.transform = transform
        self.class_to_idx = dict(class_to_idx) if class_to_idx is not None else dict(CLASS_TO_IDX)
        self.idx_to_class = {index: name for name, index in self.class_to_idx.items()}

        frame = pd.read_csv(self.csv_path, encoding="utf-8")
        missing_columns = {"path", "label"} - set(frame.columns)
        if missing_columns:
            raise ValueError(f"{self.csv_path} is missing columns: {sorted(missing_columns)}")
        unknown = sorted(set(frame["label"].astype(str)) - set(self.class_to_idx))
        if unknown:
            raise ValueError(
                f"{self.csv_path} has labels that are not PlantVillage classes: " + ", ".join(unknown)
            )
        self.paths = frame["path"].astype(str).tolist()
        self.targets = [self.class_to_idx[label] for label in frame["label"].astype(str).tolist()]

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def classes(self) -> list[str]:
        return [name for name, _index in sorted(self.class_to_idx.items(), key=lambda item: item[1])]

    def _resolve(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_file():
            return path
        return self.root / path

    def __getitem__(self, index: int) -> tuple[object, int]:
        path = self._resolve(self.paths[index])
        try:
            image = Image.open(path).convert("RGB")
        except OSError as exc:
            raise OSError(f"Could not read image {path}") from exc
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]


def make_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> DataLoader:
    """DataLoader with a notebook-safe default of ``num_workers=0``."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _dataset_for_csv(csv_path: Path, split: str) -> PlantDiseaseDataset:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}. Run python -m src.data.splits.")
    return PlantDiseaseDataset(csv_path, transform=build_transforms(split))


def load_splits(
    splits_dir: str | Path | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Train, val, and test loaders. Only the train loader is shuffled."""
    directory = Path(splits_dir) if splits_dir is not None else SPLITS_DIR
    loaders: dict[str, DataLoader] = {}
    for name in ("train", "val", "test"):
        dataset = _dataset_for_csv(directory / f"{name}.csv", name)
        loaders[name] = make_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
        )
    return loaders


def load_plantdoc(
    csv_path: str | Path | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
) -> DataLoader:
    """Field-photo test loader. Not shuffled."""
    path = Path(csv_path) if csv_path is not None else SPLITS_DIR / "plantdoc_test.csv"
    dataset = _dataset_for_csv(path, "plantdoc")
    return make_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


__all__ = [
    "CLASS_TO_IDX",
    "IMAGE_SIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "IDX_TO_CLASS",
    "NUM_CLASSES",
    "PLANTVILLAGE_CLASSES",
    "PlantDiseaseDataset",
    "build_transforms",
    "load_plantdoc",
    "load_splits",
    "make_dataloader",
]
