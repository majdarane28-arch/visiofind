from __future__ import annotations

from pathlib import Path
from typing import List

import cv2


def extract_video_frames(
    video_path: Path,
    output_dir: Path,
    every_n_frames: int = 15,
    max_frames: int = 24,
) -> List[Path]:
    """
    Extract key frames from a video.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    saved_paths: List[Path] = []
    frame_idx = 0
    saved_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % every_n_frames == 0:
                output_file = output_dir / f"frame_{saved_idx:04d}.jpg"
                cv2.imwrite(str(output_file), frame)
                saved_paths.append(output_file)
                saved_idx += 1
                if saved_idx >= max_frames:
                    break
            frame_idx += 1
    finally:
        cap.release()

    return saved_paths
