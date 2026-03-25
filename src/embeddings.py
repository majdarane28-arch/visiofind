from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


@lru_cache(maxsize=1)
def get_clip():
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    model.eval()
    return model, processor


# Prompts de référence pour détecter la présence d'une personne.
# On utilise CLIP (déjà présent dans le projet) pour éviter d'ajouter un 2e modèle.
HUMAN_TEXT_PROMPTS = [
    "a photo of a person",
    "a photo of a human",
    "a portrait of a person",
    "a standing person",
]


# Prompts pour détecter une présence de visage (portrait / face close-up).
FACE_TEXT_PROMPTS = [
    "a close-up face",
    "a photo of a face",
    "a portrait close-up",
    "a human face",
]


def _extract_features(raw):
    """Extrait de façon sécurisée le tenseur des features (gère les différences de version transformers)."""
    if isinstance(raw, torch.Tensor):
        return raw
    for attr in ("text_embeds", "image_embeds", "pooler_output"):
        val = getattr(raw, attr, None)
        if val is not None:
            return val
    if hasattr(raw, "values"):
        vals = list(raw.values())
        return vals[1] if len(vals) > 1 else vals[0]
    return raw[1] if isinstance(raw, (list, tuple)) and len(raw) > 1 else raw[0]


@lru_cache(maxsize=1)
def get_human_text_features() -> "np.ndarray":
    import numpy as np  # local import pour éviter cyclic/overhead
    model, processor = get_clip()
    with torch.no_grad():
        inputs = processor(
            text=HUMAN_TEXT_PROMPTS,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        raw = model.get_text_features(**inputs)
        feats = _extract_features(raw)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float32)


@lru_cache(maxsize=1)
def get_human_or_face_text_features() -> "np.ndarray":
    """
    Features CLIP pour prompts "personne" + "visage".
    On garde un cache séparé pour que la détection reste rapide.
    """
    import numpy as np  # local import pour compatibilité typing

    model, processor = get_clip()
    guard_prompts = HUMAN_TEXT_PROMPTS + FACE_TEXT_PROMPTS
    with torch.no_grad():
        inputs = processor(
            text=guard_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        feats = model.get_text_features(**inputs)
        feats = _extract_features(feats)
        # Sécurise : selon les versions transformers, on peut recevoir un type inattendu.
        if not isinstance(feats, torch.Tensor):
            feats = torch.tensor(feats)
        denom = feats.norm(dim=-1, keepdim=True)
        denom = torch.clamp(denom, min=1e-12)
        feats = feats / denom
    return feats.cpu().numpy().astype(np.float32)


def make_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    max_dim = max(w, h)
    if w == h:
        return img
    new_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
    new_img.paste(img, ((max_dim - w) // 2, (max_dim - h) // 2))
    return new_img


def embed_images(images: Iterable[Image.Image]) -> np.ndarray:
    model, processor = get_clip()
    imgs = [make_square(img) for img in images]
    if not imgs:
        return np.empty((0, 512), dtype=np.float32)

    with torch.no_grad():
        inputs = processor(images=imgs, return_tensors="pt", padding=True)
        raw = model.get_image_features(**inputs)
        feats = _extract_features(raw)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float32)


def human_similarity(image_vector: np.ndarray) -> float:
    """
    Calcule la similarité CLIP (cosine) entre l'image et les prompts "humain".
    image_vector doit être normalisé (embed_images le fait déjà).
    """
    import numpy as np  # local import pour compatibilité typing
    text_feats = get_human_text_features()  # (n_prompts, 512)
    if image_vector.ndim != 1:
        image_vector = image_vector.reshape(-1)
    # dot sur vecteurs unitaires = similarité cosinus
    sims = np.dot(text_feats, image_vector)  # (n_prompts,)
    return float(np.max(sims))


def human_or_face_similarity(image_vector: np.ndarray) -> float:
    """
    Similarité max CLIP entre l'image et les prompts (personne OU visage).
    image_vector doit être normalisé (embed_images le fait déjà).
    """
    import numpy as np  # local import pour compatibilité typing

    text_feats = get_human_or_face_text_features()  # (n_prompts, 512)
    if image_vector.ndim != 1:
        image_vector = image_vector.reshape(-1)

    sims = np.dot(text_feats, image_vector)  # (n_prompts,)
    return float(np.max(sims))
