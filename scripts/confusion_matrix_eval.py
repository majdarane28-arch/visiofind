"""
Évalue la recherche CLIP comme classification leave-one-out :
pour chaque image du dataset, la prédiction est la classe du plus proche voisin
(cosine, comme dans l'app), en excluant l'image elle-même.

Produit une matrice de confusion (comptages ou normalisée par ligne = rappel par classe).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.index_store import build_index_cached  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Matrice de confusion leave-one-out (CLIP + plus proche voisin)."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=ROOT,
        help="Racine du projet VisioFind",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["animaux", "objets_lieux", "terrain_sport"],
        help="Sous-dossier de data/reference/ à évaluer",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Recalculer l'index CLIP (ignorer le cache .npz)",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        default="none",
        choices=["none", "true"],
        help="none: effectifs ; true: normaliser par ligne (rappel par classe vrai)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Enregistrer la figure PNG (matplotlib) ; sinon affichage console uniquement",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Résolution de la figure PNG",
    )
    args = parser.parse_args()

    data_root = args.project_root / "data" / "reference" / args.mode
    cache_file = args.project_root / "data" / "index_cache" / f"{args.mode}.npz"

    vectors, _paths, labels = build_index_cached(
        data_root, cache_file, force_rebuild=args.force_rebuild
    )
    n = vectors.shape[0]
    if n == 0:
        print("Dataset vide (aucune image).", file=sys.stderr)
        sys.exit(1)

    # Similarité cosinus : embeddings déjà normalisés.
    sims = vectors @ vectors.T
    np.fill_diagonal(sims, -np.inf)
    pred_idx = np.argmax(sims, axis=1)
    y_true = np.array(labels, dtype=object)
    y_pred = np.array([labels[i] for i in pred_idx], dtype=object)

    acc = float(np.mean(y_true == y_pred))
    print(f"Images: {n} | Mode: {args.mode}")
    print(f"Accuracy leave-one-out (top-1 voisin): {acc * 100:.2f}%")

    classes = sorted(set(labels))
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    if args.normalize == "true":
        row_sums = cm.sum(axis=1, keepdims=True).astype(np.float64)
        cm_disp = np.zeros_like(cm, dtype=np.float64)
        np.divide(cm, row_sums, out=cm_disp, where=row_sums > 0)
        title_suffix = " (normalisé par ligne, rappel)"
    else:
        cm_disp = cm.astype(float)
        title_suffix = ""

    if args.output:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(max(8, len(classes) * 0.35), max(6, len(classes) * 0.35)))
        im = ax.imshow(cm_disp, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set(
            xticks=np.arange(len(classes)),
            yticks=np.arange(len(classes)),
            xticklabels=classes,
            yticklabels=classes,
            ylabel="Vrai (dossier)",
            xlabel="Prédit (voisin le plus proche)",
            title=f"Matrice de confusion{title_suffix}\n{args.mode} — acc={acc * 100:.1f}%",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        fig.tight_layout()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=args.dpi)
        plt.close(fig)
        print(f"Figure enregistrée: {args.output}")
    else:
        print("\nMatrice de confusion (comptages ou normalisée selon --normalize):")
        print("Classes (ordre lignes/colonnes):", ", ".join(classes))
        np.set_printoptions(linewidth=200, suppress=True, precision=3)
        print(cm_disp)


if __name__ == "__main__":
    main()
