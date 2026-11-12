"""Feature caching (M6): embed every product image once, so training never has to touch an image again.

The embedder is frozen, so the features never change. Training a classifier on them takes seconds instead of
re-running the image model for every experiment.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from wardrobe.models.embedder import Embedder

FEATURES_DIR = Path("data/features")


def embed_in_batches(embedder: Embedder, images: list[Image.Image], batch_size: int = 64) -> np.ndarray:
    """One row per image, in the same order as images.
    
    One image per call leaves the GPU waiting for Python; all images at once do not fit into GPU memory.
    """
    
    batches = []
    for i in range(0,len(images),batch_size):
        batches +=[embedder.embed_images(images[i: i+batch_size])] 
    return np.concatenate(batches)


def save_features(path: Path, ids: np.ndarray, vectors: np.ndarray) -> None:
    """Stores the product ids next to the vectors: row i of vectors belongs to ids[i]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=ids, vectors=vectors)


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(ids, vectors) as written by save_features."""
    data = np.load(path)
    return data["ids"], data["vectors"]
