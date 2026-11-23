"""Reproducible PlantVillage splits and the PlantDoc field-test table.

PlantVillage color images are split 70/15/15, stratified by class, with
``random_state=42``. Rows are sorted by label and path before the split so
the result does not depend on directory order, and the written CSVs are
sorted the same way. Training code should shuffle in the DataLoader.

PlantDoc is not mixed into train or val. Every mapped PlantDoc image is a
field-photo test row whose ``label`` is the matching PlantVillage class.
Classes that do not match are listed in ``data/plantdoc_label_map.csv`` and
are left out of ``plantdoc_test.csv``.

Usage (from the repository root)::

    python -m src.data.splits
    python -m src.data.splits --map-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.labels import (
    PLANTDOC_TO_PLANTVILLAGE,
    PLANTDOC_TO_PLANTVILLAGE_LABEL,
    PLANTVILLAGE_CLASSES,
    UNMAPPED_PLANTVILLAGE,
)
from src.data.paths import (
    IMAGE_EXTENSIONS,
    LABEL_MAP_PATH,
    PLANTDOC_DIR,
    PLANTVILLAGE_COLOR_DIR,
    PROJECT_ROOT,
    SPLITS_DIR,
)

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_COLUMNS = ["path", "label"]
MAP_COLUMNS = ["plantdoc_class", "plantvillage_label", "status", "notes"]


def is_image(path: Path) -> bool:
    if not path.is_file() or path.name.startswith("."):
        return False
    if any(part.startswith(".") or part == "__MACOSX" for part in path.parts):
        return False
    return path.suffix.lower() in IMAGE_EXTENSIONS


def relative_posix(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def collect_class_folder_images(class_root: Path, project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Collect ``path,label`` rows from ``class_root/<label>/*``."""
    if not class_root.is_dir():
        raise FileNotFoundError(f"Class folder root does not exist: {class_root}")
    rows: list[dict[str, str]] = []
    class_dirs = sorted(path for path in class_root.iterdir() if path.is_dir() and not path.name.startswith("."))
    for class_dir in class_dirs:
        label = class_dir.name
        for image_path in sorted(path for path in class_dir.rglob("*") if is_image(path)):
            rows.append({"path": relative_posix(image_path, project_root), "label": label})
    if not rows:
        raise FileNotFoundError(f"No images found under {class_root}")
    frame = pd.DataFrame(rows, columns=SPLIT_COLUMNS)
    frame = frame.drop_duplicates(subset=["path"]).sort_values(SPLIT_COLUMNS).reset_index(drop=True)
    return frame


def check_plantvillage_classes(frame: pd.DataFrame) -> None:
    found = set(frame["label"].unique())
    expected = set(PLANTVILLAGE_CLASSES)
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    if missing or extra:
        parts = []
        if missing:
            parts.append("missing " + ", ".join(missing))
        if extra:
            parts.append("unexpected " + ", ".join(extra))
        raise ValueError("PlantVillage labels do not match the 38 color classes: " + "; ".join(parts))


def stratified_splits(
    frame: pd.DataFrame,
    seed: int = SEED,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> dict[str, pd.DataFrame]:
    """Stratified train/val/test split.

    The frame is sorted first. Both ``train_test_split`` calls use ``seed``.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1, got {total}")
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or not 0 < test_ratio < 1:
        raise ValueError("Each split ratio must be between 0 and 1.")

    ordered = frame.sort_values(SPLIT_COLUMNS).reset_index(drop=True)
    counts = ordered["label"].value_counts()
    # The second stratified cut needs at least two rows per class in the
    # 30% holdout, which is safe for this dataset (smallest class is >100)
    # and fails loudly on a toy folder a teammate pointed at by mistake.
    rare = counts[counts < 10]
    if not rare.empty:
        detail = ", ".join(f"{label} ({count})" for label, count in rare.items())
        raise ValueError(
            "Stratified 70/15/15 needs at least 10 images in every class so "
            f"val and test can both be stratified. Too small: {detail}"
        )

    train_df, holdout_df = train_test_split(
        ordered,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=ordered["label"],
    )
    test_fraction_of_holdout = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=test_fraction_of_holdout,
        random_state=seed,
        stratify=holdout_df["label"],
    )
    parts = {"train": train_df, "val": val_df, "test": test_df}
    for name, part in parts.items():
        parts[name] = part.loc[:, SPLIT_COLUMNS].sort_values(SPLIT_COLUMNS).reset_index(drop=True)
    _check_partition(ordered, parts)
    return parts


def _check_partition(original: pd.DataFrame, parts: dict[str, pd.DataFrame]) -> None:
    path_sets = {name: set(part["path"]) for name, part in parts.items()}
    names = list(path_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = path_sets[left] & path_sets[right]
            if overlap:
                raise RuntimeError(f"{left} and {right} share {len(overlap)} paths")
    combined = set().union(*path_sets.values())
    if combined != set(original["path"]):
        raise RuntimeError("Splits do not cover the source images exactly once.")
    labels = set(original["label"])
    for name, part in parts.items():
        missing = labels - set(part["label"])
        if missing:
            raise RuntimeError(f"{name} is missing classes: {sorted(missing)}")


def iter_plantdoc_class_dirs(plantdoc_root: Path) -> list[tuple[str, Path]]:
    """Return ``(upstream_split, class_dir)`` for train/ and test/."""
    found: list[tuple[str, Path]] = []
    for split_name in ("train", "test"):
        split_dir = plantdoc_root / split_name
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir() and not path.name.startswith(".")):
            found.append((split_name, class_dir))
    return found


def collect_plantdoc_test(plantdoc_root: Path, project_root: Path = PROJECT_ROOT) -> tuple[pd.DataFrame, list[str]]:
    """Map PlantDoc images onto PlantVillage labels.

    Both upstream train and test folders are included. PlantDoc is a field
    test set for models trained on PlantVillage, so its own train folder is
    not used for training. Returns the test frame and PlantDoc class names
    that had no crosswalk entry.
    """
    class_dirs = iter_plantdoc_class_dirs(plantdoc_root)
    if not class_dirs:
        raise FileNotFoundError(
            f"No PlantDoc train/ or test/ class folders under {plantdoc_root}. "
            "Run python -m src.data.download."
        )
    rows: list[dict[str, str]] = []
    unknown: list[str] = []
    for _split_name, class_dir in class_dirs:
        plantvillage_label = PLANTDOC_TO_PLANTVILLAGE_LABEL.get(class_dir.name)
        if plantvillage_label is None:
            unknown.append(class_dir.name)
            continue
        for image_path in sorted(path for path in class_dir.rglob("*") if is_image(path)):
            rows.append(
                {
                    "path": relative_posix(image_path, project_root),
                    "label": plantvillage_label,
                }
            )
    frame = pd.DataFrame(rows, columns=SPLIT_COLUMNS)
    if not frame.empty:
        frame = frame.drop_duplicates(subset=["path"]).sort_values(SPLIT_COLUMNS).reset_index(drop=True)
    return frame, sorted(set(unknown))


def label_map_frame(extra_unmapped_plantdoc: list[str] | None = None) -> pd.DataFrame:
    """Documented crosswalk, including PlantVillage classes with no field photos."""
    rows: list[dict[str, str]] = []
    known = set(PLANTDOC_TO_PLANTVILLAGE_LABEL)
    for plantdoc_class, plantvillage_label, notes in PLANTDOC_TO_PLANTVILLAGE:
        rows.append(
            {
                "plantdoc_class": plantdoc_class,
                "plantvillage_label": plantvillage_label,
                "status": "mapped",
                "notes": notes,
            }
        )
    for plantvillage_label, notes in UNMAPPED_PLANTVILLAGE:
        rows.append(
            {
                "plantdoc_class": "",
                "plantvillage_label": plantvillage_label,
                "status": "unmapped",
                "notes": notes,
            }
        )
    for class_name in extra_unmapped_plantdoc or []:
        if class_name in known:
            continue
        rows.append(
            {
                "plantdoc_class": class_name,
                "plantvillage_label": "",
                "status": "unmapped",
                "notes": (
                    "Found in the PlantDoc download but not in the curated crosswalk. "
                    "Excluded from plantdoc_test.csv."
                ),
            }
        )
    return pd.DataFrame(rows, columns=MAP_COLUMNS)


def write_frame(frame: pd.DataFrame, path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, columns=columns, encoding="utf-8")


def summarize(parts: dict[str, pd.DataFrame], plantdoc: pd.DataFrame | None) -> str:
    lines = []
    total = sum(len(part) for part in parts.values())
    lines.append(f"PlantVillage images: {total}")
    lines.append(f"Classes: {len(PLANTVILLAGE_CLASSES)}")
    for name in ("train", "val", "test"):
        part = parts[name]
        share = len(part) / total if total else 0
        lines.append(f"  {name}: {len(part)} ({share:.1%})")
    counts = (
        pd.concat([part.assign(split=name) for name, part in parts.items()], ignore_index=True)
        .groupby(["label", "split"])
        .size()
        .unstack(fill_value=0)
    )
    counts["n"] = counts.sum(axis=1)
    smallest = counts["n"].idxmin()
    largest = counts["n"].idxmax()
    lines.append(
        f"Smallest class {smallest}: n={int(counts.loc[smallest, 'n'])} "
        f"train/val/test="
        f"{int(counts.loc[smallest, 'train'])}/"
        f"{int(counts.loc[smallest, 'val'])}/"
        f"{int(counts.loc[smallest, 'test'])}"
    )
    lines.append(
        f"Largest class {largest}: n={int(counts.loc[largest, 'n'])} "
        f"train/val/test="
        f"{int(counts.loc[largest, 'train'])}/"
        f"{int(counts.loc[largest, 'val'])}/"
        f"{int(counts.loc[largest, 'test'])}"
    )
    ratio = counts["n"].max() / counts["n"].min()
    lines.append(f"Imbalance (largest/smallest): {ratio:.2f}")
    if plantdoc is None:
        lines.append("PlantDoc test CSV: not written (dataset missing)")
    else:
        lines.append(f"PlantDoc field-test images: {len(plantdoc)}")
        lines.append(f"PlantDoc classes present in the test CSV: {plantdoc['label'].nunique()}")
    lines.append(f"PlantVillage classes with no PlantDoc photos: {len(UNMAPPED_PLANTVILLAGE)}")
    return "\n".join(lines)


def build_splits(
    color_dir: Path = PLANTVILLAGE_COLOR_DIR,
    plantdoc_dir: Path = PLANTDOC_DIR,
    splits_dir: Path = SPLITS_DIR,
    label_map_path: Path = LABEL_MAP_PATH,
    seed: int = SEED,
    map_only: bool = False,
    project_root: Path = PROJECT_ROOT,
) -> None:
    if map_only:
        write_frame(label_map_frame(), label_map_path, MAP_COLUMNS)
        print(f"Wrote {label_map_path}")
        return

    if not color_dir.is_dir():
        write_frame(label_map_frame(), label_map_path, MAP_COLUMNS)
        raise FileNotFoundError(
            f"PlantVillage color images not found at {color_dir}.\n"
            "Run python -m src.data.download first. "
            f"The label map was still written to {label_map_path}."
        )

    plantvillage = collect_class_folder_images(color_dir, project_root)
    check_plantvillage_classes(plantvillage)
    parts = stratified_splits(plantvillage, seed=seed)
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        destination = splits_dir / f"{name}.csv"
        write_frame(part, destination, SPLIT_COLUMNS)
        print(f"Wrote {destination} ({len(part)} rows)")

    plantdoc_frame: pd.DataFrame | None = None
    extra_unknown: list[str] = []
    if plantdoc_dir.is_dir() and (plantdoc_dir / "train").is_dir():
        plantdoc_frame, extra_unknown = collect_plantdoc_test(plantdoc_dir, project_root)
        destination = splits_dir / "plantdoc_test.csv"
        write_frame(plantdoc_frame, destination, SPLIT_COLUMNS)
        print(f"Wrote {destination} ({len(plantdoc_frame)} rows)")
    else:
        print(f"PlantDoc not found at {plantdoc_dir}; skipped plantdoc_test.csv")

    write_frame(label_map_frame(extra_unknown), label_map_path, MAP_COLUMNS)
    print(f"Wrote {label_map_path}")
    if extra_unknown:
        print("Unmapped PlantDoc classes (excluded from the test CSV): " + ", ".join(extra_unknown))
    print(summarize(parts, plantdoc_frame))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--map-only", action="store_true", help="Write the label map and skip the image splits.")
    parser.add_argument("--plantvillage", type=Path, default=PLANTVILLAGE_COLOR_DIR)
    parser.add_argument("--plantdoc", type=Path, default=PLANTDOC_DIR)
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--label-map", type=Path, default=LABEL_MAP_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        build_splits(
            color_dir=args.plantvillage,
            plantdoc_dir=args.plantdoc,
            splits_dir=args.splits_dir,
            label_map_path=args.label_map,
            seed=args.seed,
            map_only=args.map_only,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
