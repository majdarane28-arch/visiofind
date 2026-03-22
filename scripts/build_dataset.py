from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Dict, List

import requests
from ddgs import DDGS
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

# Requêtes orientées scènes / lifestyle (moins "product photo") pour limiter les 403 DDG.
QUERIES_BASIC: Dict[str, Dict[str, List[str]]] = {
    "animaux": {
        "chat": ["cat sitting outdoors", "domestic cat garden", "kitten grass"],
        "chien": ["dog running park", "labrador walking", "puppy street"],
        "oiseau": ["bird on branch", "sparrow in nature", "small bird flying"],
        "cheval": ["horse in field", "horse running meadow", "pony farm"],
        "elephant": ["elephant savanna", "african elephant wildlife", "elephants herd"],
    },
    "objets_lieux": {
        "chaise": ["wooden chair patio", "chair in cafe", "outdoor chair garden"],
        "chaussures": ["shoes outdoor", "people wearing shoes", "casual shoes street"],
        "table": ["picnic table park", "dining table home", "kitchen table scene"],
        "plage": ["sandy beach waves", "tropical coastline", "seaside sunset"],
        "rue_urbaine": ["busy city street", "urban avenue people", "downtown crossing"],
        "montagne": ["alpine peaks snow", "mountain valley hiking", "rocky summit view"],
        "parc": ["green park trees", "public park path", "city park bench"],
        "restaurant": ["restaurant dining room", "cafe interior people", "bistro tables evening"],
    },
}

QUERIES_EXTENDED: Dict[str, Dict[str, List[str]]] = {
    "animaux": {
        "chat": ["cat sitting outdoors", "tabby cat garden", "cat portrait window light"],
        "chien": ["dog running park", "golden retriever field", "street dog portrait"],
        "oiseau": ["bird on branch", "colorful bird flying", "small bird nature"],
        "cheval": ["horse galloping field", "horse stable outdoor", "pony meadow"],
        "elephant": ["elephant savanna", "african elephant herd", "elephant drinking water"],
        "vache": ["cow in pasture", "dairy cow farm", "cattle grass field"],
        "mouton": ["sheep flock meadow", "lamb spring field", "sheep grazing hill"],
        "papillon": ["butterfly on flower", "monarch butterfly garden", "butterfly macro leaf"],
        "araignee": ["spider web morning dew", "spider on leaf", "garden spider close"],
        "poule": ["hen farmyard", "chicken coop outdoor", "rooster farm morning"],
    },
    "objets_lieux": {
        "chaise": ["wooden chair patio", "chair in garden", "cafe dining chair outdoor"],
        "chaussures": [
            "shoes outdoor",
            "people wearing shoes",
            "casual shoes street",
            "sneakers on feet",
        ],
        "table": ["picnic table park", "dining table kitchen", "wooden table outdoor meal"],
        "sac": ["backpack hiking trail", "person carrying backpack urban", "handbag street style"],
        "telephone": ["phone in hand street", "smartphone outdoors", "people using phone cafe"],
        "ordinateur": ["laptop on wooden desk", "working on laptop cafe", "open laptop window light"],
        "voiture": ["car on city street", "urban traffic cars", "parked car sidewalk"],
        "velo": ["bicycle city street", "cycling bike path", "parked bike rack"],
        "plage": ["sandy beach waves", "tropical shore palm", "rocky coast daylight"],
        "rue_urbaine": ["busy pedestrian street", "urban avenue crossing", "city night lights street"],
        "montagne": ["snowy mountain ridge", "alpine valley hiking", "summit clouds view"],
        "parc": ["tree lined park path", "fountain city park", "jogging park morning"],
        "restaurant": ["busy restaurant interior", "cafe bar seating", "dinner tables evening"],
        "foret": ["dense forest trail", "sunlight through trees", "mossy woodland path"],
        "aeroport": ["airport terminal people", "runway plane takeoff", "boarding gate hall"],
    },
}

# Many CDNs block requests without a browser-like User-Agent.
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}


def download_image(url: str, output_file: Path, timeout: int = 15) -> bool:
    try:
        r = requests.get(url, timeout=timeout, headers=HTTP_HEADERS)
        if r.status_code != 200:
            return False
        content_type = r.headers.get("Content-Type", "")
        if "image" not in content_type:
            return False
        output_file.write_bytes(r.content)
        return True
    except Exception:
        return False


def _count_images(class_dir: Path) -> int:
    if not class_dir.is_dir():
        return 0
    return sum(1 for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def fetch_bing_icrawler(class_dir: Path, queries: List[str], per_class: int, start: int) -> int:
    """
    Repli si DuckDuckGo est injoignable (firewall / TLS / pays).
    Utilise Bing Images via icrawler (téléchargement séquentiel léger).
    """
    try:
        from icrawler.builtin import BingImageCrawler
    except ImportError:
        print(
            "[WARN] icrawler non installé — installe: pip install icrawler",
            flush=True,
        )
        return start

    class_dir.mkdir(parents=True, exist_ok=True)
    saved = start
    import tempfile

    for query in queries:
        if saved >= per_class:
            break
        remaining = per_class - saved
        time.sleep(3)
        with tempfile.TemporaryDirectory() as tmp:
            try:
                crawler = BingImageCrawler(
                    feeder_threads=1,
                    parser_threads=1,
                    downloader_threads=2,
                    storage={"root_dir": tmp},
                )
                crawler.crawl(keyword=query, max_num=min(remaining + 15, 100))
            except Exception as ex:
                print(f"[WARN] Bing/icrawler ({query}): {ex}", flush=True)
                continue

            imgs = sorted(
                [p for p in Path(tmp).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
            )
            for p in imgs:
                if saved >= per_class:
                    break
                ext = p.suffix.lower() if p.suffix else ".jpg"
                dest = class_dir / f"{saved:04d}{ext}"
                try:
                    shutil.copy2(p, dest)
                    saved += 1
                except OSError as ex:
                    print(f"[WARN] copy {p} -> {dest}: {ex}", flush=True)

    return saved


def fetch_for_class(
    class_dir: Path, queries: List[str], per_class: int, skip_ddgs: bool = False
) -> int:
    """
    1) DuckDuckGo (ddgs) — 2) si trop peu d'images : Bing via icrawler.
    Séquentiel uniquement (pas de multi-thread côté ddgs).
    Si skip_ddgs=True, passe directement au repli Bing (utile si duckduckgo.com est bloqué).
    """
    class_dir.mkdir(parents=True, exist_ok=True)
    saved = _count_images(class_dir)

    if not skip_ddgs:
        try:
            with DDGS(timeout=30) as ddgs:
                for query in queries:
                    if saved >= per_class:
                        break
                    time.sleep(5)
                    for attempt in range(5):
                        try:
                            for item in ddgs.images(query, max_results=per_class * 3):
                                if saved >= per_class:
                                    break
                                image_url = item.get("image")
                                if not image_url:
                                    continue
                                output_file = class_dir / f"{saved:04d}.jpg"
                                ok = download_image(image_url, output_file)
                                if ok:
                                    saved += 1
                            break
                        except Exception as ex:
                            wait = min(60, 3 * (2**attempt))
                            print(
                                f"[WARN] Query failed ({query}) attempt {attempt + 1}/5: {ex} "
                                f"(sleep {wait}s)",
                                flush=True,
                            )
                            time.sleep(wait)
        except Exception as ex:
            print(f"[WARN] DuckDuckGo indisponible: {ex}", flush=True)
    else:
        print("[INFO] DuckDuckGo ignoré (bing-only) — utilisation de Bing/icrawler.", flush=True)

    if saved < per_class:
        print(
            f"[INFO] DuckDuckGo: {saved}/{per_class} images — essai Bing (icrawler)...",
            flush=True,
        )
        saved = fetch_bing_icrawler(class_dir, queries, per_class, saved)

    return saved


def main():
    parser = argparse.ArgumentParser(description="Build VisioFind reference dataset.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to visiofind root",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=80,
        help="Number of images to collect per class",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="extended",
        choices=["basic", "extended"],
        help="Dataset preset size",
    )
    args = parser.parse_args()

    root = args.project_root / "data" / "reference"
    query_map = QUERIES_EXTENDED if args.preset == "extended" else QUERIES_BASIC
    print(f"Building dataset in: {root}")
    print(f"Preset: {args.preset}")
    for section, classes in query_map.items():
        for class_name, queries in tqdm(classes.items(), desc=section):
            class_dir = root / section / class_name
            count = fetch_for_class(class_dir=class_dir, queries=queries, per_class=args.per_class)
            print(f"[{section}] {class_name}: {count} images")

    print("Dataset build complete.")


if __name__ == "__main__":
    main()
