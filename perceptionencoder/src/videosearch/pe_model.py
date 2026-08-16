"""Thin wrapper around Meta's PE-Core CLIP for frame/text embeddings.

Loads the vendored ``core.vision_encoder`` package from ``vendor/perception_models``
so we avoid the repo's heavy (CUDA-pinned) ``pip install -e .`` path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_MODEL = "PE-Core-L14-336"

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_VENDOR = _PROJECT_ROOT / "vendor" / "perception_models"

# Redirect the Hugging Face cache to a project-local, writable dir. The default
# (~/.cache/huggingface) may contain a root-owned, unreadable token file on this
# host. Must run before huggingface_hub is imported so HF_TOKEN_PATH is derived
# from this HF_HOME.
os.environ.setdefault("HF_HOME", str(_PROJECT_ROOT / ".hf_home"))
Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
from PIL import Image


def _ensure_vendor_on_path() -> None:
    if not _VENDOR.exists():
        raise FileNotFoundError(
            f"Vendored perception_models not found at {_VENDOR}. "
            "Clone it with: git clone --depth 1 "
            "https://github.com/facebookresearch/perception_models.git "
            f"{_VENDOR}"
        )
    p = str(_VENDOR)
    if p not in sys.path:
        sys.path.insert(0, p)


class PEModel:
    """PE-Core CLIP encoder for images (video frames) and text."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        _ensure_vendor_on_path()
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms

        self.model_name = model_name
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype if self.device.type == "cuda" else torch.float32

        model = pe.CLIP.from_config(model_name, pretrained=True)
        self.model = model.to(self.device).eval()
        self.image_transform = transforms.get_image_transform(model.image_size)
        self.tokenizer = transforms.get_text_tokenizer(model.context_length)
        self.embed_dim = int(self.model.visual.output_dim)

    @property
    def _autocast(self):
        if self.device.type == "cuda":
            return torch.autocast(self.device.type, dtype=self.dtype)
        return torch.autocast("cpu", enabled=False)

    @torch.inference_mode()
    def embed_images(self, images: list[Image.Image], batch_size: int = 64) -> np.ndarray:
        """L2-normalized per-image embeddings, shape [len(images), D]."""
        if not images:
            return np.zeros((0, self.embed_dim), dtype=np.float32)
        out: list[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            tensor = torch.stack([self.image_transform(im) for im in batch]).to(self.device)
            with self._autocast:
                feats = self.model.encode_image(tensor, normalize=True)
            out.append(feats.float().cpu().numpy())
        return np.concatenate(out, axis=0)

    def embed_chunk(self, frames: list[Image.Image]) -> np.ndarray:
        """Mean-pool per-frame embeddings into one L2-normalized chunk vector [D]."""
        feats = self.embed_images(frames)
        if feats.shape[0] == 0:
            return np.zeros((self.embed_dim,), dtype=np.float32)
        pooled = feats.mean(axis=0)
        norm = np.linalg.norm(pooled)
        if norm > 0:
            pooled = pooled / norm
        return pooled.astype(np.float32)

    @torch.inference_mode()
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """L2-normalized text embeddings, shape [len(texts), D]."""
        tokens = self.tokenizer(texts).to(self.device)
        with self._autocast:
            feats = self.model.encode_text(tokens, normalize=True)
        return feats.float().cpu().numpy().astype(np.float32)
