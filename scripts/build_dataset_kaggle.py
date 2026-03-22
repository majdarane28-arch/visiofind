from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, List

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

ANIMALS_DATASET_REF = "alessiocorrado99/animals10"
PLACES_DATASET_REF = "puneet6060/intel-image-classification"


ANIMALS_MAP: Dict[str, str] = {
    "cane": "chien",
    "gatto": "chat",
    "cavallo": "cheval",
    "elefante": "elephant",
    "farfalla": "papillon",
    "gallina": "poule",
    "mucca": "vache",
    "pecora": "mouton",
    "ragno": "araignee",
    "scoiattolo": "ecureuil",
}

PLACES_MAP: Dict[str, str] = {
    "street": "rue_urbaine",
    "buildings": "rue_urbaine",
    "forest": "foret",
    "mountain": "montagne",
    "glacier": "montagne",
    "sea": "plage",
}


def run(cmd: List[str], cwd: Path | None = None):
    r = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        err = (r.stderr or "").strip() or (r.stdout or "").strip() or f"exit {r.returncode}"
        raise RuntimeError(f"Commande échouée:\n  {' '.join(cmd)}\n\n{err}")


def kaggle_json_path() -> Path:
    base = os.environ.get("KAGGLE_CONFIG_DIR")
    if base:
        return Path(base) / "kaggle.json"
    return Path.home() / ".kaggle" / "kaggle.json"


def has_kaggle_credentials() -> bool:
    """True si variables d'env ou kaggle.json avec username + key."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    p = kaggle_json_path()
    if not p.is_file():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(data.get("username") and data.get("key"))


def print_kaggle_setup_error() -> None:
    cfg = kaggle_json_path()
    print(
        "\n=== Kaggle API non configurée ===\n"
        f"Fichier attendu : {cfg}\n\n"
        "1) Va sur https://www.kaggle.com/ → compte → Settings → API\n"
        "   Clique « Create New API Token » : cela télécharge kaggle.json\n"
        "2) Copie ce fichier ICI (remplace l’ancien si besoin) :\n"
        f"   {cfg.parent}\n"
        "   Le fichier doit contenir exactement : {{ \"username\": \"...\", \"key\": \"...\" }}\n\n"
        "OU définis les variables d’environnement (même effet) :\n"
        "   KAGGLE_USERNAME=ton_username\n"
        "   KAGGLE_KEY=ta_cle_api\n\n"
        "3) Vérifie avec :\n"
        "   python scripts/check_kaggle_auth.py\n\n"
        "Alternative (CLI récente) : kaggle auth login\n",
        flush=True,
    )


def run_kaggle(args: List[str]):
    """
    Run Kaggle CLI with robust fallbacks for Windows/venv setups.
    """
    attempts = [
        [sys.executable, "-m", "kaggle.cli", *args],
        [sys.executable, "-m", "kaggle", *args],
        ["kaggle", *args],
    ]
    last_error: Exception | None = None
    for cmd in attempts:
        try:
            run(cmd)
            return
        except Exception as ex:
            last_error = ex
    if last_error:
        raise last_error
    raise RuntimeError("Failed to run Kaggle CLI")


def download_and_extract(dataset_ref: str, download_dir: Path) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    run_kaggle(
        [
            "datasets",
            "download",
            "-d",
            dataset_ref,
            "-p",
            str(download_dir),
            "--force",
        ]
    )
    zip_files = sorted(download_dir.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No zip found after download for {dataset_ref}")
    zip_path = zip_files[-1]

    extract_dir = download_dir / "extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir


def collect_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def copy_limited(src_images: List[Path], out_dir: Path, per_class: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    random.shuffle(src_images)
    selected = src_images[:per_class]
    for i, src in enumerate(selected):
        dst = out_dir / f"{i:04d}{src.suffix.lower()}"
        shutil.copy2(src, dst)


def build_animals(extracted_dir: Path, out_root: Path, per_class: int):
    all_dirs = [p for p in extracted_dir.rglob("*") if p.is_dir()]
    for src_name, target_name in ANIMALS_MAP.items():
        src_candidates = [d for d in all_dirs if d.name.lower() == src_name]
        if not src_candidates:
            print(f"[WARN] Animals class not found: {src_name}")
            continue
        src_dir = src_candidates[0]
        images = collect_images(src_dir)
        copy_limited(images, out_root / "animaux" / target_name, per_class)
        print(f"[animaux] {target_name}: {min(len(images), per_class)} images")


def build_places(extracted_dir: Path, out_root: Path, per_class: int):
    all_dirs = [p for p in extracted_dir.rglob("*") if p.is_dir()]
    grouped: Dict[str, List[Path]] = {}
    for src_name, target_name in PLACES_MAP.items():
        src_candidates = [d for d in all_dirs if d.name.lower() == src_name]
        if not src_candidates:
            continue
        for src_dir in src_candidates:
            grouped.setdefault(target_name, []).extend(collect_images(src_dir))

    for target_name, images in grouped.items():
        copy_limited(images, out_root / "objets_lieux" / target_name, per_class)
        print(f"[objets_lieux] {target_name}: {min(len(images), per_class)} images")


def main():
    parser = argparse.ArgumentParser(description="Build VisioFind dataset from Kaggle")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--per-class", type=int, default=120)
    args = parser.parse_args()

    if not has_kaggle_credentials():
        print_kaggle_setup_error()
        sys.exit(1)

    project_root = args.project_root
    data_root = project_root / "data" / "reference"
    tmp_root = project_root / "data" / "_kaggle_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    print("Downloading animals dataset from Kaggle...")
    animals_extract = download_and_extract(ANIMALS_DATASET_REF, tmp_root / "animals10")
    build_animals(animals_extract, data_root, args.per_class)

    print("Downloading places dataset from Kaggle...")
    places_extract = download_and_extract(PLACES_DATASET_REF, tmp_root / "intel")
    build_places(places_extract, data_root, args.per_class)

    print("Done. Dataset prepared in data/reference")


if __name__ == "__main__":
    main()
