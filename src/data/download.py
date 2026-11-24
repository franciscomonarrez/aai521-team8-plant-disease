"""Download PlantVillage (Kaggle, color subset), its leaf map, and PlantDoc.

Kaggle credentials are read from KAGGLE_USERNAME/KAGGLE_KEY or
~/.kaggle/kaggle.json and are never written anywhere. PlantDoc is pinned
to one commit. A few of its files differ only by letter case, so the second
file of each pair is saved as <name>__case2 to keep it on macOS/Windows.

Usage (from the repository root)::

    python -m src.data.download
    python -m src.data.download --dry-run
    python -m src.data.download --plantdoc-only
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import urllib.request
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from src.data.labels import PLANTVILLAGE_CLASSES
from src.data.paths import (
    DOWNLOAD_DIR,
    IMAGE_EXTENSIONS,
    LEAF_MAP_PATH,
    PLANTDOC_DIR,
    PLANTVILLAGE_COLOR_DIR,
    PLANTVILLAGE_DIR,
    RAW_DIR,
)

KAGGLE_DATASET = "abdallahalidev/plantvillage-dataset"
PLANTDOC_REPO = "https://github.com/pratikkayal/PlantDoc-Dataset.git"
# PlantDoc commit the CSVs were built from (2 May 2021).
PLANTDOC_COMMIT = "5467f6012d78d1c446145d5f582da6096f852ae8"
PLANTDOC_SOURCE_FILE = "SOURCE.json"
CASE_TWIN_MARK = "__case"

# Leaf ids from the original PlantVillage repo (Mohanty et al. 2016),
# pinned and checksummed so the grouped split can't change.
LEAF_MAP_COMMIT = "7f7ecc7e1eaca78107e3affe7cb5abd9427e139a"
LEAF_MAP_URL = (
    "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/"
    f"{LEAF_MAP_COMMIT}/leaf-map.json"
)
LEAF_MAP_SHA256 = (
    "3b4b253683c6911744959a3870a92509b1faa1ee34e9f8585e21d0fe6337a25b"
)


class DownloadError(RuntimeError):
    """A dataset could not be fetched or laid out as expected."""


def _lock_down(path: Path) -> None:
    """Stop a world-readable token file from being rejected by the Kaggle client."""
    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        path.chmod(0o600)


def kaggle_credentials() -> str | None:
    """Return where the credentials were found (not the secret), or None."""
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
    """Pick the color-subset members of the Kaggle zip.

    The zip also has grayscale and segmented copies, and sometimes a second
    nested color folder; use the one with the most images, preferring the
    shallower path.
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_leaf_map(
    force: bool = False,
    dry_run: bool = False,
    dest: Path = LEAF_MAP_PATH,
) -> Path:
    """Fetch the PlantVillage leaf-ID map (about 2.4 MB) to ``dest``."""
    if dest.is_file() and not force and _sha256(dest) == LEAF_MAP_SHA256:
        print(f"PlantVillage leaf map already present: {dest}")
        return dest

    print(f"PlantVillage leaf map: {LEAF_MAP_URL}")
    print(f"Destination: {dest}")
    if dry_run:
        print("Dry run: not downloading the leaf map.")
        return dest

    manual = f"Fetch it manually with:\n  curl -L -o '{dest}' {LEAF_MAP_URL}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(LEAF_MAP_URL, timeout=60) as response:
            with partial.open("wb") as handle:
                shutil.copyfileobj(response, handle)
    except (OSError, http.client.HTTPException) as exc:
        partial.unlink(missing_ok=True)
        raise DownloadError(
            f"Could not download the leaf map: {exc}\n{manual}"
        ) from exc

    digest = _sha256(partial)
    if digest != LEAF_MAP_SHA256:
        partial.unlink(missing_ok=True)
        raise DownloadError(
            f"Leaf map checksum mismatch: got {digest}, "
            f"expected {LEAF_MAP_SHA256}.\n{manual}"
        )
    partial.replace(dest)
    print(f"PlantVillage leaf map saved: {dest}")
    return dest


def _plantdoc_ready(dest: Path) -> bool:
    train_dir = dest / "train"
    test_dir = dest / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        return False
    train_classes = [path for path in train_dir.iterdir() if path.is_dir()]
    return len(train_classes) > 0


def case_fold_key(path: str) -> str:
    """Key under which macOS/Windows treat two names as the same file."""
    return unicodedata.normalize("NFC", path).casefold()


def case_twin_renames(paths: Iterable[str]) -> dict[str, str]:
    """Rename files whose paths differ only by letter case.

    In each group (sorted, so CAR1.jpg comes first) the first file keeps its
    name and the others become ``<stem>__case<n><suffix>``.
    """
    paths = list(paths)
    groups: dict[str, list[str]] = {}
    for path in paths:
        groups.setdefault(case_fold_key(path), []).append(path)
    taken = set(groups)
    renames: dict[str, str] = {}
    for members in groups.values():
        for number, path in enumerate(sorted(members)[1:], start=2):
            pure = PurePosixPath(path)
            new_name = f"{pure.stem}{CASE_TWIN_MARK}{number}{pure.suffix}"
            new_path = str(pure.with_name(new_name))
            if case_fold_key(new_path) in taken:
                raise DownloadError(f"Cannot rename {path}: {new_path} exists")
            taken.add(case_fold_key(new_path))
            renames[path] = new_path
    return renames


def _git_out(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True
    )
    return result.stdout


def _pin_plantdoc_commit(repo: Path) -> None:
    """Make sure the pinned commit is in a shallow clone of PlantDoc."""
    head = _git_out(repo, "rev-parse", "HEAD").decode().strip()
    if head != PLANTDOC_COMMIT:
        print(
            f"PlantDoc upstream HEAD is {head[:12]}; fetching the pinned "
            f"commit {PLANTDOC_COMMIT[:12]}."
        )
        _git_out(repo, "fetch", "--quiet", "--depth", "1", "origin",
                 PLANTDOC_COMMIT)


def _plantdoc_tree(repo: Path) -> dict[str, str]:
    """``{path: blob id}`` for the files under train/ and test/."""
    listing = _git_out(repo, "ls-tree", "-r", "-z", PLANTDOC_COMMIT)
    entries: dict[str, str] = {}
    for record in listing.split(b"\0"):
        if not record:
            continue
        meta, _tab, raw_path = record.partition(b"\t")
        _mode, kind, blob = meta.decode().split()
        path = raw_path.decode("utf-8")
        if kind == "blob" and path.split("/", 1)[0] in ("train", "test"):
            entries[path] = blob
    return entries


def _git_blob_id(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _write_case_twins(
    dest: Path, repo: Path, entries: dict[str, str], renames: dict[str, str]
) -> None:
    """Write each case twin from its git blob under its final name.

    A case-insensitive checkout leaves only one file per group, so all member
    names are removed first.
    """
    renamed_folds = {case_fold_key(path) for path in renames}
    members = sorted(p for p in entries if case_fold_key(p) in renamed_folds)
    for path in members:
        (dest / path).unlink(missing_ok=True)
    for path in members:
        target = dest / renames.get(path, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_git_out(repo, "cat-file", "blob", entries[path]))


def _verify_plantdoc(
    dest: Path, entries: dict[str, str], renames: dict[str, str]
) -> None:
    """Fail unless ``dest`` holds exactly the pinned files, byte for byte."""
    expected = {renames.get(path, path): blob for path, blob in entries.items()}
    found: set[str] = set()
    for split in ("train", "test"):
        for folder, _dirs, files in os.walk(dest / split):
            for name in files:
                found.add((Path(folder) / name).relative_to(dest).as_posix())
    found = {p for p in found if not PurePosixPath(p).name.startswith(".")}
    missing = sorted(set(expected) - found)
    extra = sorted(found - set(expected))
    changed = sorted(
        path for path in set(expected) & found
        if _git_blob_id(dest / path) != expected[path]
    )
    if missing or extra or changed:
        sample = (missing + extra + changed)[:5]
        raise DownloadError(
            f"PlantDoc at {dest} does not match commit "
            f"{PLANTDOC_COMMIT[:12]}: {len(missing)} missing, "
            f"{len(extra)} unexpected, {len(changed)} changed "
            f"(e.g. {sample}).\n"
            "Re-download it with: python -m src.data.download "
            "--plantdoc-only --force"
        )


def read_plantdoc_source(dest: Path = PLANTDOC_DIR) -> dict | None:
    """The ``SOURCE.json`` record written after download, if any."""
    path = dest / PLANTDOC_SOURCE_FILE
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_plantdoc_source(
    dest: Path, entries: dict[str, str], renames: dict[str, str]
) -> None:
    record = {
        "repository": PLANTDOC_REPO,
        "commit": PLANTDOC_COMMIT,
        "images": len(entries),
        "case_renames": dict(sorted(renames.items())),
    }
    (dest / PLANTDOC_SOURCE_FILE).write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )


def _finish_plantdoc(dest: Path, repo: Path) -> None:
    """Restore case twins, check every file, and record the source."""
    entries = _plantdoc_tree(repo)
    renames = case_twin_renames(entries)
    _write_case_twins(dest, repo, entries, renames)
    _verify_plantdoc(dest, entries, renames)
    _write_plantdoc_source(dest, entries, renames)
    for old, new in sorted(renames.items()):
        print(f"  case twin kept as {new} (upstream {PurePosixPath(old).name})")
    print(
        f"PlantDoc images: {len(entries)} at commit {PLANTDOC_COMMIT[:12]}, "
        "every file checked."
    )


def _repair_plantdoc(dest: Path) -> None:
    """Fix an older PlantDoc copy in place."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=DOWNLOAD_DIR) as scratch:
        repo = Path(scratch) / "plantdoc-tree"
        subprocess.run(
            ["git", "clone", "--quiet", "--depth", "1", "--no-checkout",
             "--filter=blob:none", PLANTDOC_REPO, str(repo)],
            check=True,
        )
        _pin_plantdoc_commit(repo)
        _finish_plantdoc(dest, repo)


def download_plantdoc(force: bool = False, dry_run: bool = False) -> Path:
    """Clone the PlantDoc classification images into ``data/raw/plantdoc``."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manual = (
        "Check git and the network, then retry. A manual clone is not "
        "enough: the case twins and SOURCE.json come from this script."
    )
    if _plantdoc_ready(PLANTDOC_DIR) and not force:
        source = read_plantdoc_source(PLANTDOC_DIR)
        if source and source.get("commit") == PLANTDOC_COMMIT:
            print(
                f"PlantDoc already present: {PLANTDOC_DIR} "
                f"(commit {PLANTDOC_COMMIT[:12]})"
            )
            return PLANTDOC_DIR
        print(
            f"PlantDoc at {PLANTDOC_DIR} predates the case-twin fix. "
            "Repairing it from a file-list-only clone (about 1 MB)."
        )
        if dry_run:
            print("Dry run: not repairing PlantDoc.")
            return PLANTDOC_DIR
        try:
            _repair_plantdoc(PLANTDOC_DIR)
        except subprocess.CalledProcessError as exc:
            raise DownloadError(f"PlantDoc repair failed. {manual}") from exc
        return PLANTDOC_DIR

    print(f"PlantDoc repository: {PLANTDOC_REPO} @ {PLANTDOC_COMMIT[:12]}")
    print(f"Destination: {PLANTDOC_DIR}")
    if dry_run:
        print("Dry run: not cloning PlantDoc.")
        return PLANTDOC_DIR

    if PLANTDOC_DIR.exists():
        shutil.rmtree(PLANTDOC_DIR)
    PLANTDOC_DIR.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--no-checkout",
             PLANTDOC_REPO, str(PLANTDOC_DIR)],
            check=True,
        )
        _pin_plantdoc_commit(PLANTDOC_DIR)
        # On macOS/Windows git warns that some paths collided; the twins
        # are rewritten under distinct names right after.
        subprocess.run(
            ["git", "-C", str(PLANTDOC_DIR), "checkout", "--quiet",
             PLANTDOC_COMMIT],
            check=True,
        )
        if not _plantdoc_ready(PLANTDOC_DIR):
            raise DownloadError(
                f"PlantDoc clone at {PLANTDOC_DIR} has no train/ and test/ "
                "class folders. The upstream layout may have changed."
            )
        _finish_plantdoc(PLANTDOC_DIR, PLANTDOC_DIR)
    except subprocess.CalledProcessError as exc:
        raise DownloadError(f"git clone of PlantDoc failed. {manual}") from exc
    finally:
        git_dir = PLANTDOC_DIR / ".git"
        if git_dir.exists():
            shutil.rmtree(git_dir)
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
        # The leaf map needs no Kaggle key, so fetch it first.
        try:
            download_leaf_map(force=args.force, dry_run=args.dry_run)
        except DownloadError as exc:
            failures.append(str(exc))
        try:
            download_plantvillage(force=args.force, dry_run=args.dry_run)
        except DownloadError as exc:
            failures.append(str(exc))

    if failures:
        print("\n".join(failures), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
