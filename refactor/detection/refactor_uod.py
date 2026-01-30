from __future__ import annotations

import os
import uuid
from pathlib import Path

IMAGES_DIR = Path("../../dataset/detection/UOD/images")
LABELS_DIR = Path("../../dataset/detection/UOD/labels")
PREFIX = "4_"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABEL_EXTS = {".txt"}


def collect_files(root: Path, exts: set[str]) -> list[Path]:
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def add_prefix(name: str) -> str:
    if name.startswith(PREFIX):
        return name
    stem, suffix = os.path.splitext(name)
    return f"{PREFIX}{stem}{suffix}"


def two_phase_rename(paths: list[Path]) -> int:
    # Phase 1: rename to temporary unique names to avoid collisions (e.g., 000001 -> 4_000001 when it already exists)
    temp_map: dict[Path, Path] = {}
    for p in paths:
        new_name = add_prefix(p.name)
        if new_name == p.name:
            continue
        tmp = p.with_name(f".tmp__{uuid.uuid4().hex}__{p.name}")
        temp_map[p] = tmp

    for src, tmp in temp_map.items():
        src.rename(tmp)

    # Phase 2: rename to final names
    count = 0
    for src, tmp in temp_map.items():
        final = tmp.with_name(add_prefix(src.name))
        tmp.rename(final)
        count += 1

    return count


def main() -> None:
    image_files = collect_files(IMAGES_DIR, IMAGE_EXTS)  # includes train/val/... subfolders
    label_files = collect_files(LABELS_DIR, LABEL_EXTS)  # includes train/val/... subfolders

    renamed_images = two_phase_rename(image_files)
    renamed_labels = two_phase_rename(label_files)

    print(f"Renamed images: {renamed_images}")
    print(f"Renamed labels: {renamed_labels}")
    print(f"Example: dataset/detection/UOD/images/train/000000.jpg -> dataset/detection/UOD/images/train/{PREFIX}000000.jpg")


if __name__ == "__main__":
    main()


