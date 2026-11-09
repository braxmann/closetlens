"""Download the Fashion Product Images dataset and build data/datasets/fashion_products/products.parquet.

Two versions of the same dataset (same product ids) are combined:
- ashraq/fashion-product-images-small: all labels (articleType, baseColour, usage, ...), but only 60x80 px images
- ceyda/fashion-products-small: 384x512 px images, but without articleType and baseColour

Run from the project root: .venv/bin/python -m scripts.prepare_dataset
"""
import subprocess
from pathlib import Path

import pandas as pd

DATASET_DIR = Path("data/datasets/fashion_products")
SOURCES = {
    "labels": "ashraq/fashion-product-images-small",
    "large": "ceyda/fashion-products-small",
}
LABEL_COLUMNS = [
    "id", "gender", "masterCategory", "subCategory", "articleType", "baseColour", "usage", "productDisplayName",
]
# Only what a wardrobe can contain; watches, bags, perfume etc. are dropped.
KEEP_MASTER_CATEGORIES = ["Apparel", "Footwear"]


def download(prefix: str, repo: str) -> list[Path]:
    paths = []
    for part in (0, 1):
        path = DATASET_DIR / f"{prefix}_{part}.parquet"
        if not path.exists():
            url = f"https://huggingface.co/api/datasets/{repo}/parquet/default/train/{part}.parquet"
            print(f"Downloading {url}")
            # curl instead of Python's urllib: it brings its own certificates, also on Windows.
            subprocess.run(["curl", "-sfL", url, "-o", str(path)], check=True)
        paths.append(path)
    return paths


def main() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    labels = pd.concat(
        [pd.read_parquet(path, columns=LABEL_COLUMNS) for path in download("raw", SOURCES["labels"])],
        ignore_index=True,
    )
    large = pd.concat(
        [pd.read_parquet(path, columns=["id", "image"]) for path in download("large", SOURCES["large"])],
        ignore_index=True,
    )
    # The id is a number in one version and text in the other.
    large["id"] = large["id"].astype(int)
    # Keep only the JPEG bytes; the "path" part of the image field is not needed.
    large["image"] = large["image"].map(lambda field: field["bytes"])

    # Inner join: only products that exist in both versions.
    products = labels.merge(large, on="id", how="inner")
    products = products[products["masterCategory"].isin(KEEP_MASTER_CATEGORIES)].reset_index(drop=True)

    output = DATASET_DIR / "products.parquet"
    products.to_parquet(output, index=False)
    print(f"{len(products)} products written to {output} ({output.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
