"""Download PlantVillage (Kaggle, color subset) and PlantDoc (GitHub).

Credentials are read from the environment (``KAGGLE_USERNAME`` and
``KAGGLE_KEY``) or from ``~/.kaggle/kaggle.json``. This module never writes
credentials and nothing under ``data/raw/`` is committed.

Usage (from the repository root)::

    python -m src.data.download
    python -m src.data.download --dry-run
    python -m src.data.download --plantdoc-only
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from src.data.labels import PLANTVILLAGE_CLASSES
from src.data.paths import (
    DOWNLOAD_DIR,
    IMAGE_EXTENSIONS,
    PLANTDOC_DIR,
    PLANTVILLAGE_COLOR_DIR,
    PLANTVILLAGE_DIR,
    RAW_DIR,
)

KAGGLE_DATASET = "abdallahalidev/plantvillage-dataset"
PLANTDOC_REPO = "https://github.com/pratikkayal/PlantDoc-Dataset.git"


class DownloadError(RuntimeError):
    """A dataset could not be fetched or laid out as expected."""


def _lock_down(path: Path) -> None:
    """Stop a world-readable token file from being rejected by the Kaggle client."""
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        path.chmod(0o600)


def kaggle_credentials() -> str | None:
    """Return where credentials were found, or None.

    The classic username/key pair is what this project documents. Current
    Kaggle clients also accept a single API token. The returned string is a
    location, never the secret.
    """
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()
    if username and key:
        return "environment variables KAGGLE_USERNAME and KAGGLE_KEY"
    if os.environ.get("KAGGLE_API_TOKEN", "").strip():
        return "environment variable KAGGLE_API_TOKEN"
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    legacy_token = config_dir / "kaggle.json"
    if legacy_token.is_file():
        _lock_down(legacy_token)
        return str(legacy_token)
    access_token = config_dir / "access_token"
    if access_token.is_file():
        _lock_down(access_token)
        return str(access_token)
    return None


def _credential_help() -> str:
    return (
        "Kaggle credentials were not found, so PlantVillage was not downloaded.\n"
        "The dataset is abdallahalidev/plantvillage-dataset (color subset).\n\n"
        "Use one of these, and do not commit the key:\n"
        "  1. export KAGGLE_USERNAME=... and export KAGGLE_KEY=...\n"
        "  2. Save the API token JSON to ~/.kaggle/kaggle.json\n"
        "     (https://www.kaggle.com/settings -> API -> Create New Token)\n"
        "     chmod 600 ~/.kaggle/kaggle.json\n"
        "The current Kaggle client also accepts KAGGLE_API_TOKEN or ~/.kaggle/access_token."
    )


def _is_ignored_part(part: str) -> bool:
    return part == "__MACOSX" or part.startswith(".")


def _is_image_name(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTENSIONS


def select_color_members(names: list[str]) -> tuple[str, list[str]]:
    """Pick the color-subset members inside a Kaggle zip.

    The archive also contains grayscale and segmented copies, and sometimes a
    second nested ``color`` tree. Prefer the color directory that holds the
    most class images, breaking ties toward the shallower path so the
    top-level ``color/`` folder wins over a duplicate.
    """
    grouped: dict[str, list[str]] = {}
    for name in names:
        if name.endswith("/"):
            continue
        parts = PurePosixPath(name).parts
        if not parts or any(_is_ignored_part(part) for part in parts):
            continue
        if "color" not in parts or not _is_image_name(name):
            continue
        color_index = parts.index("color")
        if color_index + 2 >= len(parts):
            continue
        if "___" not in parts[color_index + 1]:
            continue
        prefix = "/".join(parts[: color_index + 1])
        grouped.setdefault(prefix, []).append(name)

    if not grouped:
        return _select_class_folder_members(names)

    def rank(item: tuple[str, list[str]]) -> tuple[int, int, int]:
        prefix, members = item
        return (len(members), -prefix.count("/"), -len(prefix))

    prefix, members = max(grouped.items(), key=rank)
    return prefix, members


def _select_class_folder_members(names: list[str]) -> tuple[str, list[str]]:
    """Fallback for an archive that has class folders but no ``color/`` directory."""
    grouped: dict[str, list[str]] = {}
    for name in names:
        if name.endswith("/"):
            continue
        parts = PurePosixPath(name).parts
        if not parts or any(_is_ignored_part(part) for part in parts):
            continue
        if not _is_image_name(name):
            continue
        class_indexes = [index for index, part in enumerate(parts[:-1]) if "___" in part]
        if len(class_indexes) != 1:
            continue
        prefix = "/".join(parts[: class_indexes[0]])
        grouped.setdefault(prefix, []).append(name)
    if not grouped:
        return "", []
    prefix, members = max(grouped.items(), key=lambda item: len(item[1]))
    return prefix, members


def _member_relative(member: str, prefix: str) -> str:
    if prefix:
        return member[len(prefix) :].lstrip("/")
    return member.lstrip("/")


def extract_color_zip(zip_path: Path, dest_root: Path) -> Path:
    """Extract only the color subset to ``dest_root/color``."""
    color_dir = dest_root / "color"
    if color_dir.exists():
        shutil.rmtree(color_dir)
    color_dir.mkdir(parents=True, exist_ok=True)

    with ZipFile(zip_path) as archive:
        prefix, members = select_color_members(archive.namelist())
        if not members:
            raise DownloadError(
                f"{zip_path} has no PlantVillage color class folders. "
                "Expected a color/ directory of folders such as Apple___healthy."
            )
        print(f"Extracting {len(members)} color images from {prefix or '(archive root)'}")
        root = color_dir.resolve()
        for member in members:
            relative = _member_relative(member, prefix)
            target = (color_dir / relative).resolve()
            if target != root and root not in target.parents:
                raise DownloadError(f"Unsafe path in archive: {member}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
    return color_dir


def _color_ready(color_dir: Path) -> bool:
    if not color_dir.is_dir():
        return False
    class_dirs = [path for path in color_dir.iterdir() if path.is_dir() and "___" in path.name]
    return len(class_dirs) >= len(PLANTVILLAGE_CLASSES)


def _check_color_classes(color_dir: Path) -> int:
    found = {path.name for path in color_dir.iterdir() if path.is_dir()}
    expected = set(PLANTVILLAGE_CLASSES)
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    image_count = sum(1 for path in color_dir.rglob("*") if path.is_file() and _is_image_name(path.name))
    print(f"PlantVillage color: {len(found & expected)} expected class folders, {image_count} images")
    print(f"  {color_dir}")
    if missing or extra:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("unexpected: " + ", ".join(extra))
        raise DownloadError("Color class folders do not match the 38 PlantVillage labels. " + "; ".join(details))
    return image_count


def _find_zip(directory: Path) -> Path | None:
    zips = sorted(directory.glob("*.zip"))
    if not zips:
        return None
    preferred = [path for path in zips if "plantvillage" in path.name.lower()]
    return preferred[0] if preferred else zips[0]


def download_plantvillage(force: bool = False, dry_run: bool = False) -> Path:
    """Download the Kaggle dataset and keep the color subset."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if _color_ready(PLANTVILLAGE_COLOR_DIR) and not force:
        print(f"PlantVillage color subset already present: {PLANTVILLAGE_COLOR_DIR}")
        _check_color_classes(PLANTVILLAGE_COLOR_DIR)
        return PLANTVILLAGE_COLOR_DIR

    where = kaggle_credentials()
    if where is None:
        raise DownloadError(_credential_help())
    print(f"Kaggle credentials: {where}")
    print(f"Dataset: {KAGGLE_DATASET}")
    print(f"Destination: {PLANTVILLAGE_COLOR_DIR}")
    if dry_run:
        print("Dry run: not downloading PlantVillage.")
        return PLANTVILLAGE_COLOR_DIR

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import kaggle
    except ImportError as exc:
        raise DownloadError("The kaggle package is not installed. pip install -r requirements.txt") from exc

    api = kaggle.api
    try:
        api.authenticate()
    except SystemExit as exc:
        raise DownloadError(_credential_help()) from exc
    print("Downloading the Kaggle archive (about 2 GB). Grayscale and segmented copies are not extracted.")
    api.dataset_download_files(
        KAGGLE_DATASET,
        path=str(DOWNLOAD_DIR),
        unzip=False,
        quiet=False,
        force=force,
    )
    zip_path = _find_zip(DOWNLOAD_DIR)
    if zip_path is None:
        raise DownloadError(f"Kaggle finished but no zip file was found in {DOWNLOAD_DIR}")

    if force and PLANTVILLAGE_DIR.exists():
        shutil.rmtree(PLANTVILLAGE_DIR)
    PLANTVILLAGE_DIR.mkdir(parents=True, exist_ok=True)
    extract_color_zip(zip_path, PLANTVILLAGE_DIR)
    zip_path.unlink(missing_ok=True)
    _check_color_classes(PLANTVILLAGE_COLOR_DIR)
    return PLANTVILLAGE_COLOR_DIR


def _plantdoc_ready(dest: Path) -> bool:
    train_dir = dest / "train"
    test_dir = dest / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        return False
    train_classes = [path for path in train_dir.iterdir() if path.is_dir()]
    return len(train_classes) > 0


def download_plantdoc(force: bool = False, dry_run: bool = False) -> Path:
    """Clone the PlantDoc classification images into ``data/raw/plantdoc``."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if _plantdoc_ready(PLANTDOC_DIR) and not force:
        print(f"PlantDoc already present: {PLANTDOC_DIR}")
        return PLANTDOC_DIR

    print(f"PlantDoc repository: {PLANTDOC_REPO}")
    print(f"Destination: {PLANTDOC_DIR}")
    if dry_run:
        print("Dry run: not cloning PlantDoc.")
        return PLANTDOC_DIR

    if PLANTDOC_DIR.exists():
        shutil.rmtree(PLANTDOC_DIR)
    PLANTDOC_DIR.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", PLANTDOC_REPO, str(PLANTDOC_DIR)],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise DownloadError(
            "git clone of PlantDoc failed. Clone it manually with:\n"
            f"  git clone --depth 1 {PLANTDOC_REPO} {PLANTDOC_DIR}"
        ) from exc

    git_dir = PLANTDOC_DIR / ".git"
    if git_dir.exists():
        shutil.rmtree(git_dir)
    if not _plantdoc_ready(PLANTDOC_DIR):
        raise DownloadError(
            f"PlantDoc clone at {PLANTDOC_DIR} has no train/ and test/ class folders. "
            "The upstream layout may have changed."
        )
    image_count = sum(1 for path in PLANTDOC_DIR.rglob("*") if path.is_file() and _is_image_name(path.name))
    print(f"PlantDoc images: {image_count}")
    return PLANTDOC_DIR


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plantvillage-only", action="store_true", help="Download only the Kaggle color subset.")
    parser.add_argument("--plantdoc-only", action="store_true", help="Clone only PlantDoc.")
    parser.add_argument("--force", action="store_true", help="Replace data that is already on disk.")
    parser.add_argument("--dry-run", action="store_true", help="Check credentials and print paths. Do not download.")
    args = parser.parse_args(argv)
    if args.plantvillage_only and args.plantdoc_only:
        parser.error("Choose at most one of --plantvillage-only and --plantdoc-only.")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    do_plantdoc = not args.plantvillage_only
    do_plantvillage = not args.plantdoc_only
    failures: list[str] = []

    if do_plantdoc:
        try:
            download_plantdoc(force=args.force, dry_run=args.dry_run)
        except DownloadError as exc:
            failures.append(str(exc))

    if do_plantvillage:
        try:
            download_plantvillage(force=args.force, dry_run=args.dry_run)
        except DownloadError as exc:
            failures.append(str(exc))

    if failures:
        print("\n".join(failures), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
