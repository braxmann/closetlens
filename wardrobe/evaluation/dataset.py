"""Access to the Fashion Product Images dataset built by scripts/prepare_dataset.py."""
from io import BytesIO
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

PRODUCTS_PATH = Path("data/datasets/fashion_products/products.parquet")


def load_products() -> pd.DataFrame:
    """One row per product with its labels, without images: those would make the table ~400 MB."""
    label_columns = [name for name in pq.read_schema(PRODUCTS_PATH).names if name != "image"]
    return pd.read_parquet(PRODUCTS_PATH, columns=label_columns)


def load_images(ids: list[int]) -> dict[int, Image.Image]:
    """Images for the given product ids, read in one pass."""
    table = pd.read_parquet(PRODUCTS_PATH, columns=["id", "image"], filters=[("id", "in", list(ids))])
    return {int(row.id): Image.open(BytesIO(row.image)) for row in table.itertuples()}
