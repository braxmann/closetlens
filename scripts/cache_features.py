"""Embed every product image once and store the vectors in data/features/<model>.npz.

Run from the project root: .venv/bin/python -m scripts.cache_features --model hf-hub:Marqo/marqo-fashionSigLIP
"""
import argparse
import re
import time

import numpy as np

from wardrobe.evaluation.dataset import load_images, load_products
from wardrobe.models.embedder import OpenClipEmbedder
from wardrobe.training.features import FEATURES_DIR, embed_in_batches, save_features


def features_path(model: str):
    # "hf-hub:Marqo/marqo-fashionSigLIP" -> data/features/marqo-fashionSigLIP.npz
    return FEATURES_DIR / f"{re.sub(r'[:/]', '-', model.removeprefix('hf-hub:').split('/')[-1])}.npz"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="hf-hub:Marqo/marqo-fashionSigLIP")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    ids = np.sort(load_products()["id"].to_numpy())
    print(f"Loading {len(ids)} images ...", flush=True)
    # Image.open is lazy: this holds the compressed files (~400 MB), each image is decoded only when embedded.
    images = load_images(list(ids))

    embedder = OpenClipEmbedder(args.model)
    start = time.perf_counter()
    vectors = embed_in_batches(embedder, [images[int(id)] for id in ids], args.batch_size)
    print(f"Embedded {len(ids)} images in {time.perf_counter() - start:.0f} s -> shape {vectors.shape}")

    path = features_path(args.model)
    save_features(path, ids, vectors)
    print(f"Written to {path}")


if __name__ == "__main__":
    main()
