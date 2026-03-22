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
        # transformers récents renvoient BaseModelOutputWithPooling, pas un Tensor
        feats = raw.pooler_output if hasattr(raw, "pooler_output") else raw
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float32)
