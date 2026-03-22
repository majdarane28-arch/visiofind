from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

import streamlit as st
from PIL import Image

from src.embeddings import embed_images
from src.index_store import SearchResult, build_index_cached, load_pil, search_similar
from src.media import extract_video_frames
from src.places import google_maps_link, place_metadata

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "reference"
CACHE_ROOT = PROJECT_ROOT / "data" / "index_cache"


@st.cache_resource
def get_index(mode: str, force_rebuild: bool = False):
    dataset_root = DATA_ROOT / mode
    cache_file = CACHE_ROOT / f"{mode}.npz"
    vectors, paths, labels = build_index_cached(dataset_root, cache_file, force_rebuild=force_rebuild)
    return vectors, paths, labels


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
        options=["animaux", "objets_lieux"],
        index=0,
        format_func=lambda x: "Animaux" if x == "animaux" else "Objets et lieux",
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
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / uploaded.name
            video_path.write_bytes(uploaded.read())

            frames = extract_video_frames(
                video_path=video_path,
                output_dir=tmp_path / "frames",
                every_n_frames=15,
                max_frames=24,
            )
            if not frames:
                st.error("Impossible de lire cette video.")
                return

            st.subheader("Frames extraites")
            options = [f.name for f in frames]
            selected_name = st.selectbox("Choisis une frame a analyser", options=options)
            selected_path = next(f for f in frames if f.name == selected_name)
            st.image(str(selected_path), caption=f"Frame selectionnee: {selected_name}", width=450)
            selected_image = load_pil(selected_path)
    else:
        selected_image = Image.open(uploaded).convert("RGB")
        st.image(selected_image, caption="Image upload", width=450)

    if st.button("Lancer la recherche visuelle", type="primary"):
        with st.spinner("Indexation/recherche en cours..."):
            if force_reindex:
                get_index.clear()
            vectors, paths, labels = get_index(mode, force_rebuild=force_reindex)
            if vectors.shape[0] == 0:
                st.error(
                    "Dataset vide. Lance d'abord: python scripts/build_dataset.py --per-class 80"
                )
                return
            query_vector = embed_images([selected_image])[0]
            results = search_similar(query_vector, vectors, paths, labels, top_k=top_k)

        st.subheader("Resultats similaires")
        show_results(results, mode)


if __name__ == "__main__":
    main()
