from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

import streamlit as st
from PIL import Image

from src.embeddings import embed_images, human_or_face_similarity
from src.index_store import SearchResult, build_index_cached, load_pil, search_similar
from src.media import extract_video_frames
from src.places import google_maps_link, place_metadata

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "reference"
CACHE_ROOT = PROJECT_ROOT / "data" / "index_cache"
MODE_LABELS = {
    "animaux": "Animaux",
    "objets_lieux": "Objets et lieux",
    "terrain_sport": "Terrains de sport",
}

# Seuil fixe de détection personne/visage (CLIP).
# Si l'image uploadée dépasse ce seuil, on bloque la recherche.
HUMAN_OR_FACE_THRESHOLD = 0.35


@st.cache_resource
def get_index(mode: str, force_rebuild: bool = False):
    dataset_root = DATA_ROOT / mode
    cache_file = CACHE_ROOT / f"{mode}.npz"
    vectors, paths, labels = build_index_cached(dataset_root, cache_file, force_rebuild=force_rebuild)
    return vectors, paths, labels


@st.cache_data(show_spinner=False)
def get_video_frames(video_bytes: bytes) -> List[Image.Image]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        video_path = tmp_path / "temp_video.mp4"
        video_path.write_bytes(video_bytes)

        frames = extract_video_frames(
            video_path=video_path,
            output_dir=tmp_path / "frames",
            every_n_frames=15,
            max_frames=24,
        )
        images = []
        for f in frames:
            with Image.open(f) as img:
                images.append(img.convert("RGB"))
        return images


def show_results(results: List[SearchResult], mode: str):
    if not results:
        st.warning("Aucun resultat. Lance d'abord le script de creation dataset.")
        return
    cols = st.columns(4)
    for i, result in enumerate(results):
        col = cols[i % 4]
        with col:
            st.image(str(result.path), use_container_width=True)
            st.caption(f"{result.label} - similarite: {result.score * 100:.1f}%")

            if mode == "objets_lieux":
                meta = place_metadata(result.label)
                if meta:
                    lat = float(meta["lat"])
                    lon = float(meta["lon"])
                    name = str(meta["name"])
                    st.markdown(f"Lieu detecte: **{name}**")
                    st.markdown(f"[Voir sur Google Maps]({google_maps_link(lat, lon)})")


def main():
    st.set_page_config(page_title="VisioFind", layout="wide")
    st.title("VisioFind - Recherche visuelle image/video")
    st.write(
        "Upload une image ou une video, choisis une frame, puis retrouve des elements similaires "
        "(animaux ou objets/lieux)."
    )

    mode = st.sidebar.radio(
        "Type de recherche",
        options=list(MODE_LABELS.keys()),
        index=0,
        format_func=lambda x: MODE_LABELS.get(x, x),
    )
    top_k = st.sidebar.slider("Nombre de resultats", min_value=4, max_value=20, value=8, step=2)
    force_reindex = st.sidebar.checkbox("Forcer reindexation", value=False)

    uploaded = st.file_uploader(
        "Ajoute une image ou une video",
        type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "avi", "mkv"],
    )

    if uploaded is None:
        st.info("Ajoute un fichier pour commencer.")
        return

    suffix = Path(uploaded.name).suffix.lower()
    selected_image: Image.Image | None = None

    if suffix in {".mp4", ".mov", ".avi", ".mkv"}:
        video_bytes = uploaded.getvalue()
        with st.spinner("Extraction des frames en cours (mise en cache)..."):
            frames = get_video_frames(video_bytes)

        if not frames:
            st.error("Impossible de lire cette video.")
            return

        st.subheader("Frames extraites")
        
        file_key = f"{uploaded.name}_{uploaded.size}"
        if "selected_frame_idx" not in st.session_state or st.session_state.get("last_uploaded") != file_key:
            st.session_state.selected_frame_idx = 0
            st.session_state.last_uploaded = file_key

        cols = st.columns(4)
        for i, frame in enumerate(frames):
            col = cols[i % 4]
            with col:
                st.image(frame, use_container_width=True)
                if st.button(f"Choisir #{i+1}", key=f"frame_{i}"):
                    st.session_state.selected_frame_idx = i

        # Sécurise l'index sélectionné : si l'utilisateur recharge une vidéo,
        # il peut arriver que l'index précédent soit hors limites.
        idx = st.session_state.selected_frame_idx
        if not isinstance(idx, int):
            idx = 0
        if idx < 0 or idx >= len(frames):
            idx = 0
            st.session_state.selected_frame_idx = 0

        selected_image = frames[idx]
        st.markdown(f"**Frame selectionnee : #{idx + 1}**")
        st.image(selected_image, width=450)
    else:
        selected_image = Image.open(uploaded).convert("RGB")
        st.image(selected_image, caption="Image upload", width=450)

    if st.button("Lancer la recherche visuelle", type="primary"):
        with st.spinner("Indexation/recherche en cours..."):
            # Calcule l'embedding uniquement au clic.
            # Cela évite que Streamlit reste bloqué (chargement CLIP, réseau) avant d'afficher le bouton.
            query_vector = embed_images([selected_image])[0]  # (512,) normalisé
            guard_score = human_or_face_similarity(query_vector)
            human_blocked = guard_score >= HUMAN_OR_FACE_THRESHOLD
            if human_blocked:
                score_pct = guard_score * 100.0
                st.error("Je ne peux pas rechercher des humains/visages sur cette photo.")
                st.info(
                    "Merci d'uploader une autre image (animaux, objets/lieux, ou terrains de sport)."
                )
                st.caption(
                    f"Détection personne/visage: {score_pct:.1f}% (seuil: {HUMAN_OR_FACE_THRESHOLD*100:.0f}%)"
                )
                return

            if force_reindex:
                get_index.clear()
            vectors, paths, labels = get_index(mode, force_rebuild=force_reindex)
            if vectors.shape[0] == 0:
                st.error(
                    "Dataset vide. Lance d'abord: python scripts/build_dataset.py --per-class 80"
                )
                return
            results = search_similar(query_vector, vectors, paths, labels, top_k=top_k)

        st.subheader("Resultats similaires")
        show_results(results, mode)


if __name__ == "__main__":
    main()
