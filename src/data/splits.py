"""Reproducible PlantVillage splits and the PlantDoc field-test table.

PlantVillage color images are split about 70/15/15 by leaf, not by image.
PlantVillage photographed most leaves several times, so an image-level split
puts the same leaf in train and in test. Each image gets a ``group``: the
leaf it shows when the original release's ``leaf-map.json`` knows it,
otherwise its own file name (so byte-identical copies under different Kaggle
prefixes still stay together). ``StratifiedGroupKFold`` with
``random_state=42`` deals the groups into 20 folds of similar class balance;
folds 0-2 are test, 3-5 are val, and 6-19 are train. The ``fold`` column is
written too, so train can be reused for grouped cross-validation.

Rows are sorted by path before the split so the result does not depend on
directory order, and the written CSVs are sorted the same way. Training code
should shuffle in the DataLoader.

PlantDoc is not mixed into train or val. Every mapped PlantDoc image is a
field-photo test row whose ``label`` is the matching PlantVillage class.
Classes that do not match are listed in ``data/plantdoc_label_map.csv`` and
are left out of ``plantdoc_test.csv``.

Usage (from the repository root)::

    python -m src.data.splits
    python -m src.data.splits --map-only
    python -m src.data.splits --image-level   # old leaky split, for comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from src.data.labels import (
    PLANTDOC_TO_PLANTVILLAGE,
    PLANTDOC_TO_PLANTVILLAGE_LABEL,
    PLANTVILLAGE_CLASSES,
    UNMAPPED_PLANTVILLAGE,
)
from src.data.paths import (
    IMAGE_EXTENSIONS,
    LABEL_MAP_PATH,
    LEAF_MAP_PATH,
    PLANTDOC_DIR,
    PLANTVILLAGE_COLOR_DIR,
    PROJECT_ROOT,
    SPLITS_DIR,
)

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
N_FOLDS = 20
SPLIT_COLUMNS = ["path", "label"]
PLANTVILLAGE_COLUMNS = ["path", "label", "group", "fold"]
MAP_COLUMNS = ["plantdoc_class", "plantvillage_label", "status", "notes"]

# leaf-map.json calls Apple black rot "Apple_Frogeye Spot".
LEAF_MAP_CLASS_ALIASES = {"Apple___Black_rot": "Apple_Frogeye Spot"}
# Released Target Spot files are named "Com.G_TgS_FL 1234"; leaf-map.json
# lists the same frames as "Com.G_FL 1234". All 1,404 match after this.
LEAF_MAP_STEM_PREFIXES = {
    "Tomato___Target_Spot": ("com.g_tgs_fl ", "com.g_fl "),
}


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


def original_stem(path: str) -> str:
    """Lower-cased file stem without the Kaggle ``<uuid>___`` prefix.

    This is the key ``leaf-map.json`` uses, e.g.
    ``.../0a1b___RS_HL 6268.JPG`` -> ``rs_hl 6268``.
    """
    stem = PurePosixPath(path).stem
    if "___" in stem:
        stem = stem.split("___", 1)[1]
    return stem.lower().strip()


def load_leaf_map(path: Path = LEAF_MAP_PATH) -> dict[str, dict[str, str]]:
    """Read ``leaf-map.json`` as ``{map_class: {file_stem: leaf_id}}``.

    Leaf ids are kept as written minus a trailing ``.0``, so batch prefixes
    such as ``OCT_769`` (citrus) and ``JR_12`` (pepper) stay distinct from
    leaf 769 or 12 of another batch. Entries with no leaf (four Cherry rows
    say ``n/a``) are dropped; those images fall back to a per-file group
    instead of being merged into one fake leaf.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"PlantVillage leaf map not found at {path}.\n"
            "Run python -m src.data.download (this file needs no Kaggle "
            "key; it comes from github.com/spMohanty/PlantVillage-Dataset)."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    by_class: dict[str, dict[str, str]] = {}
    for stem, entries in raw.items():
        for entry in entries:
            map_class, _sep, number = entry.partition(":::")
            number = number.strip()
            if not number or number.lower() == "n/a":
                continue
            leaf_id = number[:-2] if number.endswith(".0") else number
            known = by_class.setdefault(map_class, {})
            if known.setdefault(stem, leaf_id) != leaf_id:
                raise ValueError(
                    f"{path} gives {stem!r} two leaves in {map_class}"
                )
    return by_class


def leaf_groups(
    frame: pd.DataFrame, leaf_map: dict[str, dict[str, str]]
) -> pd.Series:
    """One group id per row: the photographed leaf when known.

    Rows found in the leaf map get ``leaf:<label>:<number>``. The rest get
    ``file:<label>:<stem>``; that still keeps copies of one original file
    (same name under a different Kaggle prefix) in the same group.
    """
    groups: list[str] = []
    for path, label in zip(frame["path"], frame["label"], strict=True):
        stem = original_stem(path)
        key = stem
        old, new = LEAF_MAP_STEM_PREFIXES.get(label, ("", ""))
        if old and key.startswith(old):
            key = new + key[len(old):]
        map_class = LEAF_MAP_CLASS_ALIASES.get(label, label)
        leaf = leaf_map.get(map_class, {}).get(key)
        if leaf is None:
            groups.append(f"file:{label}:{stem}")
        else:
            groups.append(f"leaf:{label}:{leaf}")
    return pd.Series(groups, index=frame.index, name="group")


def _check_ratios(
    train_ratio: float, val_ratio: float, test_ratio: float
) -> None:
    ratios = (train_ratio, val_ratio, test_ratio)
    total = sum(ratios)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1, got {total}")
    if not all(0 < ratio < 1 for ratio in ratios):
        raise ValueError("Each split ratio must be between 0 and 1.")


def grouped_splits(
    frame: pd.DataFrame,
    seed: int = SEED,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    n_folds: int = N_FOLDS,
) -> dict[str, pd.DataFrame]:
    """Leaf-grouped train/val/test split; no group is in two splits.

    ``frame`` needs ``path``, ``label`` and ``group``. It is sorted by path,
    then ``StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed)``
    assigns every group to one fold. The first ``test_ratio * n_folds``
    folds are test, the next ``val_ratio * n_folds`` are val, the rest are
    train. Split sizes are close to the ratios but not exact, because whole
    groups move together.
    """
    _check_ratios(train_ratio, val_ratio, test_ratio)
    test_folds = round(test_ratio * n_folds)
    val_folds = round(val_ratio * n_folds)
    for ratio, folds in ((test_ratio, test_folds), (val_ratio, val_folds)):
        if folds < 1 or abs(folds - ratio * n_folds) > 1e-6:
            raise ValueError(
                f"With {n_folds} folds, val and test ratios must be "
                f"multiples of 1/{n_folds}; got {ratio}"
            )
    if "group" not in frame.columns:
        raise ValueError("grouped_splits needs a 'group' column")

    ordered = frame.sort_values(SPLIT_COLUMNS).reset_index(drop=True)
    fold = np.full(len(ordered), -1, dtype=int)
    splitter = StratifiedGroupKFold(
        n_splits=n_folds, shuffle=True, random_state=seed
    )
    folds = splitter.split(ordered, ordered["label"], groups=ordered["group"])
    for index, (_rest, held_out) in enumerate(folds):
        fold[held_out] = index
    if (fold < 0).any():
        raise RuntimeError("StratifiedGroupKFold left rows without a fold")
    ordered["fold"] = fold

    split_name = np.where(
        fold < test_folds,
        "test",
        np.where(fold < test_folds + val_folds, "val", "train"),
    )
    parts = {
        name: ordered.loc[split_name == name, PLANTVILLAGE_COLUMNS]
        .sort_values(SPLIT_COLUMNS)
        .reset_index(drop=True)
        for name in ("train", "val", "test")
    }
    _check_partition(ordered, parts)
    return parts


def stratified_splits(
    frame: pd.DataFrame,
    seed: int = SEED,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> dict[str, pd.DataFrame]:
    """Image-level stratified split (the original one; it leaks leaves).

    Kept only to rebuild ``*_imagelevel.csv`` for comparison. The frame is
    sorted first. Both ``train_test_split`` calls use ``seed``.
    """
    _check_ratios(train_ratio, val_ratio, test_ratio)

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
            if "group" in parts[left] and "group" in parts[right]:
                shared = set(parts[left]["group"]) & set(parts[right]["group"])
                if shared:
                    raise RuntimeError(
                        f"{left} and {right} share {len(shared)} leaf groups"
                    )
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
    if all("group" in part for part in parts.values()):
        groups = pd.concat([part["group"] for part in parts.values()])
        with_leaf = groups.str.startswith("leaf:")
        lines.append(
            f"Leaf groups: {groups.nunique()}; rows with a leaf id from "
            f"leaf-map.json: {int(with_leaf.sum())} ({with_leaf.mean():.1%})"
        )
        # Fold assignment depends on the StratifiedGroupKFold version.
        lines.append(f"scikit-learn {sklearn.__version__}")
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
    leaf_map_path: Path = LEAF_MAP_PATH,
    image_level: bool = False,
) -> None:
    """Write the split CSVs, the PlantDoc test CSV, and the label map.

    With ``image_level=True`` only ``{train,val,test}_imagelevel.csv`` are
    written, using the original image-level split (for comparison).
    """
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

    leaf_map = None if image_level else load_leaf_map(leaf_map_path)
    plantvillage = collect_class_folder_images(color_dir, project_root)
    check_plantvillage_classes(plantvillage)
    splits_dir.mkdir(parents=True, exist_ok=True)
    if image_level:
        parts = stratified_splits(plantvillage, seed=seed)
        for name, part in parts.items():
            destination = splits_dir / f"{name}_imagelevel.csv"
            write_frame(part, destination, SPLIT_COLUMNS)
            print(f"Wrote {destination} ({len(part)} rows)")
        return

    plantvillage["group"] = leaf_groups(plantvillage, leaf_map)
    parts = grouped_splits(plantvillage, seed=seed)
    for name, part in parts.items():
        destination = splits_dir / f"{name}.csv"
        write_frame(part, destination, PLANTVILLAGE_COLUMNS)
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
    parser.add_argument(
        "--leaf-map-json",
        type=Path,
        default=LEAF_MAP_PATH,
        help="PlantVillage leaf-map.json (fetched by src.data.download).",
    )
    parser.add_argument(
        "--image-level",
        action="store_true",
        help=(
            "Write the old image-level split to *_imagelevel.csv instead. "
            "It puts the same leaf in train and test; comparison only."
        ),
    )
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
            leaf_map_path=args.leaf_map_json,
            image_level=args.image_level,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
