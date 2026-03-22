"""
Vérifie que l'API Kaggle est bien reconnue (kaggle.json ou variables d'environnement).
Usage: python scripts/check_kaggle_auth.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def kaggle_json_path() -> Path:
    base = os.environ.get("KAGGLE_CONFIG_DIR")
    if base:
        return Path(base) / "kaggle.json"
    return Path.home() / ".kaggle" / "kaggle.json"


def main() -> int:
    cfg = kaggle_json_path()
    print(f"Chemin kaggle.json attendu : {cfg}\n")

    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        print("OK — KAGGLE_USERNAME et KAGGLE_KEY sont définis.\n")
    elif cfg.is_file():
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"ERREUR — JSON invalide dans {cfg} : {e}")
            return 1
        if not data.get("username") or not data.get("key"):
            print(
                "ERREUR — Le fichier doit contenir les clés \"username\" et \"key\" "
                "(télécharge un nouveau token depuis Kaggle → Settings → API)."
            )
            return 1
        print('OK — Fichier présent avec "username" et "key".\n')
    else:
        print(
            "ERREUR — Aucun identifiant trouvé.\n"
            f"  - Crée le dossier : {cfg.parent}\n"
            f"  - Place kaggle.json dedans (token API depuis kaggle.com → Settings → API)\n"
            "  - Ou définis KAGGLE_USERNAME + KAGGLE_KEY\n"
        )
        return 1

    print("Test API : liste des fichiers du dataset animals10 ...")
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "files",
            "-d",
            "alessiocorrado99/animals10",
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        out = (r.stderr or r.stdout or "").strip()
        print("ECHEC — La commande kaggle a échoué :\n")
        print(out or f"exit code {r.returncode}")
        print(
            "\nCauses fréquentes :\n"
            "  - Clé API révoquée ou mauvais compte → régénère le token sur Kaggle.\n"
            "  - Conditions du dataset non acceptées → ouvre la page du dataset sur Kaggle et accepte les règles.\n"
            "  - Réseau / proxy / VPN.\n"
        )
        return 1

    print("OK — Kaggle répond correctement. Tu peux lancer :\n")
    print("  python scripts/build_dataset_kaggle.py --per-class 60\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
