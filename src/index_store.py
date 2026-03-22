from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

from .embeddings import embed_images

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class SearchResult:
    path: Path
    label: str
    score: float


def list_images(root: Path) -> List[Path]:
    files: List[Path] = []
    for path in root.rglob("*"):
        if path.suffix.lower() in IMAGE_EXTS and path.is_file():
            files.append(path)
    return sorted(files)


def load_pil(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def build_index(dataset_root: Path) -> Tuple[np.ndarray, List[Path], List[str]]:
    paths = list_images(dataset_root)
    if not paths:
        return np.empty((0, 512), dtype=np.float32), [], []

    images = [load_pil(p) for p in paths]
    vectors = embed_images(images)
    labels = [p.parent.name for p in paths]
    return vectors, paths, labels


def build_index_cached(
    dataset_root: Path,
    cache_file: Path,
    force_rebuild: bool = False,
) -> Tuple[np.ndarray, List[Path], List[str]]:
    """
    Build CLIP index with on-disk cache.
    Cache invalidation uses simple dataset fingerprint:
    - number of files
    - max mtime
    """
    paths = list_images(dataset_root)
    if not paths:
        return np.empty((0, 512), dtype=np.float32), [], []

    max_mtime = max(int(p.stat().st_mtime) for p in paths)
    file_count = len(paths)

    if cache_file.exists() and not force_rebuild:
        try:
            data = np.load(cache_file, allow_pickle=True)
            cached_count = int(data["file_count"].item())
            cached_mtime = int(data["max_mtime"].item())
            if cached_count == file_count and cached_mtime == max_mtime:
                vectors = data["vectors"].astype(np.float32)
                cached_paths = [Path(p) for p in data["paths"].tolist()]
                labels = data["labels"].tolist()
                return vectors, cached_paths, labels
        except Exception:
            pass

    vectors, idx_paths, labels = build_index(dataset_root)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_file,
        vectors=vectors,
        paths=np.array([str(p) for p in idx_paths], dtype=object),
        labels=np.array(labels, dtype=object),
        file_count=np.array([file_count], dtype=np.int32),
        max_mtime=np.array([max_mtime], dtype=np.int64),
    )
    return vectors, idx_paths, labels


def search_similar(
    query_vector: np.ndarray,
    index_vectors: np.ndarray,
    index_paths: List[Path],
    index_labels: List[str],
    top_k: int = 8,
) -> List[SearchResult]:
    if index_vectors.shape[0] == 0:
        return []

    query = query_vector.reshape(1, -1)
    scores = np.dot(index_vectors, query.T).squeeze(-1)
    top_idx = np.argsort(-scores)[:top_k]
    return [
        SearchResult(
            path=index_paths[i],
            label=index_labels[i],
            score=float(scores[i]),
        )
        for i in top_idx
    ]
