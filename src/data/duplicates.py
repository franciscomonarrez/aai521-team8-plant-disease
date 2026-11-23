"""Exact and near-duplicate checks for image files.

``md5_file`` finds byte-identical files. ``phash`` is the 64-bit DCT
perceptual hash computed the same way as ``imagehash.phash`` (32x32
grayscale with LANCZOS, 2-D DCT-II, top-left 8x8 block, bits above the
median), so resized or re-encoded copies of one photo end up a few bits
apart. ``src.data.splits`` uses the MD5 check to keep byte-identical copies
out of the PlantDoc test set; the EDA notebook uses both.

A small pHash distance is a hint, not proof: two different stock photos on
a white background can also be close. Pairs were checked by eye before any
PlantDoc image was excluded (see ``data/plantdoc_excluded.csv``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dct

HASH_COLUMNS = ["path", "md5", "phash", "width", "height", "exif_orientation"]
EXIF_ORIENTATION_TAG = 0x0112


def md5_file(path: str | Path) -> str:
    """Hex MD5 of a file's bytes."""
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phash(image: Image.Image) -> int:
    """64-bit perceptual hash, equal to ``int(str(imagehash.phash(im)), 16)``."""
    small = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.float64)
    low = dct(dct(pixels, axis=0), axis=1)[:8, :8]
    bits = (low > np.median(low)).ravel()
    return int("".join("1" if bit else "0" for bit in bits), 2)


def _hash_one(path: str) -> tuple[str, int, int, int, int]:
    """MD5, pHash, raw size and EXIF orientation (0 if absent) of one file."""
    with Image.open(path) as image:
        width, height = image.size
        orientation = int(image.getexif().get(EXIF_ORIENTATION_TAG) or 0)
        value = phash(image)
    return md5_file(path), value, width, height, orientation


def hash_images(
    paths: Sequence[str],
    root: str | Path,
    workers: int | None = None,
) -> pd.DataFrame:
    """Hash every ``root / path``; one row per path, in the given order.

    Width and height are the stored pixel size, before any EXIF rotation.
    ``workers=0`` hashes in this process; otherwise a process pool is used.
    """
    root = Path(root)
    files = [str(root / path) for path in paths]
    if workers == 0:
        results = [_hash_one(path) for path in files]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_hash_one, files, chunksize=64))
    frame = pd.DataFrame(results, columns=HASH_COLUMNS[1:])
    frame.insert(0, "path", list(paths))
    frame["phash"] = frame["phash"].astype(np.uint64)
    return frame


# Set bits in each byte value; summing 8 lookups counts a 64-bit hash.
_BYTE_BITS = np.array([bin(value).count("1") for value in range(256)])


def _popcount(values: np.ndarray) -> np.ndarray:
    as_bytes = values.astype(np.uint64).view(np.uint8).reshape(-1, 8)
    return _BYTE_BITS[as_bytes].sum(axis=1)


def hamming(hashes: np.ndarray, value: int) -> np.ndarray:
    """Bit distance from ``value`` to every hash in ``hashes``."""
    return _popcount(np.bitwise_xor(hashes.astype(np.uint64), np.uint64(value)))


def near_pairs(hashes: Sequence[int], max_distance: int) -> pd.DataFrame:
    """Index pairs ``i < j`` whose hashes are at most ``max_distance`` apart."""
    values = np.asarray(hashes, dtype=np.uint64)
    rows: list[tuple[int, int, int]] = []
    for i in range(len(values) - 1):
        distance = hamming(values[i + 1 :], int(values[i]))
        for offset in np.flatnonzero(distance <= max_distance):
            rows.append((i, i + 1 + int(offset), int(distance[offset])))
    return pd.DataFrame(rows, columns=["i", "j", "distance"])


def connected_groups(n: int, pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    """Group id per item (0..n-1), joining every pair; singletons get -1."""
    parent = list(range(n))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for left, right in pairs:
        parent[find(left)] = find(right)
    roots = np.array([find(item) for item in range(n)])
    sizes = pd.Series(roots).value_counts()
    group = np.full(n, -1)
    for number, root in enumerate(sorted(sizes[sizes > 1].index)):
        group[roots == root] = number
    return group


def nearest(
    queries: Sequence[int], references: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """For each query hash: index of the closest reference and the distance."""
    reference = np.asarray(references, dtype=np.uint64)
    index = np.empty(len(queries), dtype=int)
    distance = np.empty(len(queries), dtype=int)
    for position, value in enumerate(np.asarray(queries, dtype=np.uint64)):
        bits = hamming(reference, int(value))
        index[position] = int(bits.argmin())
        distance[position] = int(bits[index[position]])
    return index, distance
