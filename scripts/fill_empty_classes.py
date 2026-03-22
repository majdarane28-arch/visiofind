"""
Fill only class folders under data/reference that currently have 0 images.
Uses the same DuckDuckGo queries as build_dataset.py (extended or basic preset).
Does not re-download Kaggle archives.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from build_dataset import QUERIES_BASIC, QUERIES_EXTENDED, fetch_for_class  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def count_images(class_dir: Path) -> int:
    if not class_dir.is_dir():
        return 0
    return sum(
        1 for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill VisioFind reference classes that have no images yet."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to visiofind root",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=60,
        help="Target images per class (same as build_dataset_kaggle --per-class)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="extended",
        choices=["basic", "extended"],
        help="Query preset (must match build_dataset)",
    )
    parser.add_argument(
        "--bing-only",
        action="store_true",
        help="Ne pas appeler DuckDuckGo (souvent bloqué) — uniquement Bing/icrawler, plus rapide",
    )
    args = parser.parse_args()

    root = args.project_root / "data" / "reference"
    query_map = QUERIES_EXTENDED if args.preset == "extended" else QUERIES_BASIC

    to_fill: list[tuple[str, str, list[str], Path]] = []
    for section, classes in query_map.items():
        for class_name, queries in classes.items():
            class_dir = root / section / class_name
            if count_images(class_dir) == 0:
                to_fill.append((section, class_name, queries, class_dir))

    if not to_fill:
        print(f"No empty classes under {root} (all known classes have images).", flush=True)
        return

    print(f"Filling {len(to_fill)} empty class folder(s) under {root}", flush=True)
    for section, class_name, queries, class_dir in to_fill:
        print(f"\n[{section}] {class_name} -> {class_dir}", flush=True)
        n = fetch_for_class(
            class_dir=class_dir,
            queries=queries,
            per_class=args.per_class,
            skip_ddgs=args.bing_only,
        )
        print(f"[{section}] {class_name}: {n} images", flush=True)

    print("\nfill_empty_classes: complete.", flush=True)


if __name__ == "__main__":
    main()
