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
field-photo test row whose ``label`` is the matching PlantVillage class and
whose ``source_split`` is PlantDoc's own train/ or test/ folder. Classes that
do not match are listed in ``data/plantdoc_label_map.csv`` and are left out.
Images listed in ``data/plantdoc_excluded.csv`` (hand-reviewed: duplicate
photos, the same photo filed under two labels, and file names that name a
different disease) are left out too. The build fails if that list does not
match the PlantDoc copy on disk, if two paths differ only by letter case, or
if two kept images are byte-identical.

Usage (from the repository root)::

    python -m src.data.splits
    python -m src.data.splits --plantdoc-only  # no PlantVillage needed
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

from src.data.download import (
    PLANTDOC_COMMIT,
    case_fold_key,
    read_plantdoc_source,
)
from src.data.duplicates import md5_file
from src.data.labels import (
    PLANTDOC_TO_PLANTVILLAGE,
    PLANTDOC_TO_PLANTVILLAGE_LABEL,
    PLANTDOC_WITHOUT_USABLE_IMAGES,
    PLANTVILLAGE_CLASSES,
    UNMAPPED_PLANTVILLAGE,
)
from src.data.paths import (
    IMAGE_EXTENSIONS,
    LABEL_MAP_PATH,
    LEAF_MAP_PATH,
    PLANTDOC_DIR,
    PLANTDOC_EXCLUDED_PATH,
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
PLANTDOC_COLUMNS = ["path", "label", "source_split"]
MAP_COLUMNS = [
    "plantdoc_class", "plantvillage_label", "status", "confidence", "notes",
]
EXCLUDED_COLUMNS = [
    "path", "source_split", "label", "reason", "kept_path", "detail",
]
# file_name: the name says another disease, crop or a symptom in a healthy
# folder. duplicate: another copy of the same photo (same label) is kept.
# conflicting_duplicate: the same photo is filed under another label; at
# most one copy is kept, and only when a file name settles the label.
EXCLUSION_REASONS = ("file_name", "duplicate", "conflicting_duplicate")

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


def scan_plantdoc(
    plantdoc_root: Path, project_root: Path = PROJECT_ROOT
) -> tuple[pd.DataFrame, list[str], int]:
    """Every PlantDoc image in a mapped folder, before any exclusion.

    Returns ``path,label,source_split`` rows, the PlantDoc folders with no
    crosswalk entry, and the number of images seen in all folders.
    """
    class_dirs = iter_plantdoc_class_dirs(plantdoc_root)
    if not class_dirs:
        raise FileNotFoundError(
            f"No PlantDoc train/ or test/ class folders under {plantdoc_root}. "
            "Run python -m src.data.download."
        )
    rows: list[dict[str, str]] = []
    unknown: list[str] = []
    seen = 0
    for split_name, class_dir in class_dirs:
        images = sorted(p for p in class_dir.rglob("*") if is_image(p))
        seen += len(images)
        plantvillage_label = PLANTDOC_TO_PLANTVILLAGE_LABEL.get(class_dir.name)
        if plantvillage_label is None:
            unknown.append(class_dir.name)
            continue
        for image_path in images:
            rows.append(
                {
                    "path": relative_posix(image_path, project_root),
                    "label": plantvillage_label,
                    "source_split": split_name,
                }
            )
    frame = pd.DataFrame(rows, columns=PLANTDOC_COLUMNS)
    frame = frame.sort_values(SPLIT_COLUMNS).reset_index(drop=True)
    return frame, sorted(set(unknown)), seen


def check_plantdoc_copy(
    plantdoc_root: Path, frame: pd.DataFrame, seen: int
) -> None:
    """Fail unless this is the pinned PlantDoc copy with both case twins.

    A copy cloned on macOS or Windows before the case-twin fix is missing
    six images, and a Linux copy has twins that the committed CSV renamed.
    """
    repair = (
        "Run python -m src.data.download --plantdoc-only to repair it "
        "(downloads about 1 MB)."
    )
    source = read_plantdoc_source(plantdoc_root)
    if source is None or source.get("commit") != PLANTDOC_COMMIT:
        raise ValueError(
            f"{plantdoc_root} is not the PlantDoc copy this project pins "
            f"(commit {PLANTDOC_COMMIT[:12]}, recorded in SOURCE.json). "
            + repair
        )
    if seen != source.get("images"):
        raise ValueError(
            f"Found {seen} PlantDoc images, but SOURCE.json lists "
            f"{source.get('images')}. " + repair
        )
    folded = frame["path"].map(case_fold_key)
    clash = frame.loc[folded.duplicated(keep=False), "path"].tolist()
    if clash:
        raise ValueError(
            "PlantDoc paths that differ only by letter case would load the "
            f"same file on macOS/Windows: {clash[:4]}. " + repair
        )


def load_plantdoc_exclusions(
    path: Path = PLANTDOC_EXCLUDED_PATH,
) -> pd.DataFrame:
    """Read the reviewed PlantDoc exclusion list."""
    if not path.is_file():
        raise FileNotFoundError(f"PlantDoc exclusion list not found: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(EXCLUDED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    unknown = set(frame["reason"]) - set(EXCLUSION_REASONS)
    if unknown:
        raise ValueError(f"{path} has unknown reasons: {sorted(unknown)}")
    if frame["path"].duplicated().any():
        raise ValueError(f"{path} lists a path twice")
    return frame.loc[:, EXCLUDED_COLUMNS]


def apply_plantdoc_exclusions(
    frame: pd.DataFrame, exclusions: pd.DataFrame
) -> pd.DataFrame:
    """Drop the listed rows after checking the list matches ``frame``."""
    labels = dict(zip(frame["path"], frame["label"], strict=True))
    absent = sorted(set(exclusions["path"]) - set(labels))
    if absent:
        raise ValueError(
            f"{len(absent)} excluded PlantDoc paths are not in this copy "
            f"(e.g. {absent[0]}). The exclusion list was made for commit "
            f"{PLANTDOC_COMMIT[:12]} at data/raw/plantdoc."
        )
    relabeled = [
        path
        for path, label in zip(
            exclusions["path"], exclusions["label"], strict=True
        )
        if labels[path] != label
    ]
    if relabeled:
        raise ValueError(
            "The crosswalk label changed for excluded PlantDoc images "
            f"(e.g. {relabeled[0]}); review data/plantdoc_excluded.csv."
        )
    kept = frame.loc[~frame["path"].isin(exclusions["path"])]
    targets = set(exclusions["kept_path"]) - {""}
    dangling = sorted(targets - set(kept["path"]))
    if dangling:
        raise ValueError(f"kept_path is not a kept image: {dangling[0]}")
    return kept.reset_index(drop=True)


def check_no_byte_duplicates(
    frame: pd.DataFrame, project_root: Path = PROJECT_ROOT
) -> None:
    """Fail if two rows point at byte-identical files."""
    digests = frame["path"].map(lambda path: md5_file(project_root / path))
    copies = frame.loc[digests.duplicated(keep=False), "path"].tolist()
    if copies:
        raise RuntimeError(
            f"{len(copies)} PlantDoc test images are byte-identical copies, "
            f"e.g. {copies[:2]}. Add them to data/plantdoc_excluded.csv."
        )


def collect_plantdoc_test(
    plantdoc_root: Path,
    project_root: Path = PROJECT_ROOT,
    excluded_path: Path = PLANTDOC_EXCLUDED_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Map PlantDoc images onto PlantVillage labels, minus the exclusions.

    Both upstream train and test folders are included. PlantDoc is a field
    test set for models trained on PlantVillage, so its own train folder is
    not used for training. Returns the kept rows, the exclusion list, and
    PlantDoc class names that had no crosswalk entry.
    """
    frame, unknown, seen = scan_plantdoc(plantdoc_root, project_root)
    check_plantdoc_copy(plantdoc_root, frame, seen)
    exclusions = load_plantdoc_exclusions(excluded_path)
    kept = apply_plantdoc_exclusions(frame, exclusions)
    check_no_byte_duplicates(kept, project_root)
    no_images = {
        PLANTDOC_TO_PLANTVILLAGE_LABEL[name]
        for name in PLANTDOC_WITHOUT_USABLE_IMAGES
    }
    expected = set(PLANTDOC_TO_PLANTVILLAGE_LABEL.values()) - no_images
    if set(kept["label"]) != expected:
        raise ValueError(
            "PlantDoc classes with images do not match the crosswalk: "
            f"missing {sorted(expected - set(kept['label']))}, "
            f"unexpected {sorted(set(kept['label']) - expected)}"
        )
    return kept, exclusions, unknown


def label_map_frame(extra_unmapped_plantdoc: list[str] | None = None) -> pd.DataFrame:
    """Documented crosswalk, including PlantVillage classes with no field photos.

    ``status`` is ``mapped`` (field photos in plantdoc_test.csv),
    ``no_usable_images`` (the folder maps here but every image was
    excluded) or ``unmapped`` (no PlantDoc folder).
    """
    rows: list[dict[str, str]] = []
    known = set(PLANTDOC_TO_PLANTVILLAGE_LABEL)
    for plantdoc_class, plantvillage_label, confidence, notes in (
        PLANTDOC_TO_PLANTVILLAGE
    ):
        status = (
            "no_usable_images"
            if plantdoc_class in PLANTDOC_WITHOUT_USABLE_IMAGES
            else "mapped"
        )
        rows.append(
            {
                "plantdoc_class": plantdoc_class,
                "plantvillage_label": plantvillage_label,
                "status": status,
                "confidence": confidence,
                "notes": notes,
            }
        )
    for plantvillage_label, notes in UNMAPPED_PLANTVILLAGE:
        rows.append(
            {
                "plantdoc_class": "",
                "plantvillage_label": plantvillage_label,
                "status": "unmapped",
                "confidence": "",
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
                "confidence": "",
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


def summarize(
    parts: dict[str, pd.DataFrame],
    plantdoc: pd.DataFrame | None,
    excluded: pd.DataFrame | None = None,
) -> str:
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
        lines.append(summarize_plantdoc(plantdoc, excluded))
    return "\n".join(lines)


def summarize_plantdoc(
    plantdoc: pd.DataFrame, excluded: pd.DataFrame | None = None
) -> str:
    """Counts for plantdoc_test.csv and the exclusions behind it."""
    lines = [
        f"PlantDoc field-test images: {len(plantdoc)} "
        f"(upstream train/ {int((plantdoc['source_split'] == 'train').sum())},"
        f" test/ {int((plantdoc['source_split'] == 'test').sum())})"
    ]
    if excluded is not None:
        reasons = excluded["reason"].value_counts()
        detail = ", ".join(
            f"{reason} {int(reasons.get(reason, 0))}"
            for reason in EXCLUSION_REASONS
        )
        lines.append(
            f"PlantDoc images excluded: {len(excluded)} ({detail}); "
            "see data/plantdoc_excluded.csv"
        )
    lines.append(
        "PlantDoc classes present in the test CSV: "
        f"{plantdoc['label'].nunique()}"
    )
    without = len(UNMAPPED_PLANTVILLAGE) + len(PLANTDOC_WITHOUT_USABLE_IMAGES)
    lines.append(f"PlantVillage classes with no PlantDoc photos: {without}")
    return "\n".join(lines)


def _write_plantdoc(
    plantdoc_dir: Path,
    splits_dir: Path,
    project_root: Path,
    excluded_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    plantdoc, excluded, unknown = collect_plantdoc_test(
        plantdoc_dir, project_root, excluded_path
    )
    destination = splits_dir / "plantdoc_test.csv"
    write_frame(plantdoc, destination, PLANTDOC_COLUMNS)
    print(f"Wrote {destination} ({len(plantdoc)} rows)")
    return plantdoc, excluded, unknown


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
    plantdoc_only: bool = False,
    plantdoc_excluded_path: Path = PLANTDOC_EXCLUDED_PATH,
) -> None:
    """Write the split CSVs, the PlantDoc test CSV, and the label map.

    With ``image_level=True`` only ``{train,val,test}_imagelevel.csv`` are
    written, using the original image-level split (for comparison). With
    ``plantdoc_only=True`` only ``plantdoc_test.csv`` and the label map are
    written, so PlantVillage is not needed.
    """
    if map_only:
        write_frame(label_map_frame(), label_map_path, MAP_COLUMNS)
        print(f"Wrote {label_map_path}")
        return

    if plantdoc_only:
        splits_dir.mkdir(parents=True, exist_ok=True)
        plantdoc, excluded, unknown = _write_plantdoc(
            plantdoc_dir, splits_dir, project_root, plantdoc_excluded_path
        )
        write_frame(label_map_frame(unknown), label_map_path, MAP_COLUMNS)
        print(f"Wrote {label_map_path}")
        print(summarize_plantdoc(plantdoc, excluded))
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
    excluded: pd.DataFrame | None = None
    extra_unknown: list[str] = []
    if plantdoc_dir.is_dir() and (plantdoc_dir / "train").is_dir():
        plantdoc_frame, excluded, extra_unknown = _write_plantdoc(
            plantdoc_dir, splits_dir, project_root, plantdoc_excluded_path
        )
    else:
        print(f"PlantDoc not found at {plantdoc_dir}; skipped plantdoc_test.csv")

    write_frame(label_map_frame(extra_unknown), label_map_path, MAP_COLUMNS)
    print(f"Wrote {label_map_path}")
    if extra_unknown:
        print("Unmapped PlantDoc classes (excluded from the test CSV): " + ", ".join(extra_unknown))
    print(summarize(parts, plantdoc_frame, excluded))


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
    parser.add_argument(
        "--plantdoc-only",
        action="store_true",
        help="Write only plantdoc_test.csv and the label map (no PlantVillage needed).",
    )
    parser.add_argument(
        "--plantdoc-excluded",
        type=Path,
        default=PLANTDOC_EXCLUDED_PATH,
        help="Reviewed list of PlantDoc images to leave out.",
    )
    args = parser.parse_args(argv)
    modes = [args.map_only, args.image_level, args.plantdoc_only]
    if sum(modes) > 1:
        parser.error("Choose at most one of --map-only, --image-level and --plantdoc-only.")
    return args


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
            plantdoc_only=args.plantdoc_only,
            plantdoc_excluded_path=args.plantdoc_excluded,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
