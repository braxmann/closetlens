"""Draw a random sample of test products as numbered contact sheets for a manual label audit.

Writes data/evaluation/audit/sheet_<n>.png (25 tiles each) and audit.csv with one row per tile.
Each tile shows our labels with the original dataset value in parentheses, e.g. "pants (Jeans)".
Look at the sheets and write the wrong attributes into the "wrong" column of audit.csv,
e.g. "main_color" or "category main_color". Leave it empty if all labels are right.

Run from the project root: .venv/bin/python -m scripts.make_audit_sheets
"""
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from wardrobe.evaluation.dataset import load_images, load_products
from wardrobe.evaluation.split import ATTRIBUTE_MAPS
from scripts.make_test_set import TEST_SET_PATH

AUDIT_DIR = Path("data/evaluation/audit")
AUDIT_PATH = AUDIT_DIR / "audit.csv"
COLUMNS, ROWS = 5, 5
TILE_SIZE = (240, 320)
TEXT_HEIGHT = 90


def label_table(sample_size: int, seed: int) -> pd.DataFrame:
    """One row per sampled product with our labels and the dataset values they come from."""
    test_set = pd.read_parquet(TEST_SET_PATH)
    labels = test_set.pivot(index="id", columns="attribute", values="label").reset_index()
    labels = labels.sample(sample_size, random_state=seed).reset_index(drop=True)
    source_columns = [column for column, _ in ATTRIBUTE_MAPS.values()]
    labels = labels.merge(load_products()[["id", *source_columns]], on="id", how="left")
    labels.insert(0, "tile", range(1, len(labels) + 1))
    labels["wrong"] = ""
    return labels


def draw_tile(number: int, image: Image.Image, row, font) -> Image.Image:
    tile = Image.new("RGB", (TILE_SIZE[0], TILE_SIZE[1] + TEXT_HEIGHT), "white")
    image = image.convert("RGB")
    image.thumbnail(TILE_SIZE)
    tile.paste(image, ((TILE_SIZE[0] - image.width) // 2, 0))
    lines = [f"#{number}  id {row.id}"]
    for attribute, (column, _) in ATTRIBUTE_MAPS.items():
        label = getattr(row, attribute)
        # A product is only tested on the attributes it was sampled for.
        lines.append(f"{attribute}: {label} ({getattr(row, column)})" if pd.notna(label) else f"{attribute}: -")
    ImageDraw.Draw(tile).multiline_text((6, TILE_SIZE[1] + 4), "\n".join(lines), fill="black", font=font)
    return tile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="replace an existing audit.csv")
    args = parser.parse_args()

    if AUDIT_PATH.exists() and not args.force:
        raise SystemExit(f"{AUDIT_PATH} already exists and may contain your audit; use --force to replace it.")
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    labels = label_table(args.size, args.seed)
    images = load_images(labels["id"].tolist())
    font = ImageFont.load_default(size=15)
    per_sheet = COLUMNS * ROWS
    for start in range(0, len(labels), per_sheet):
        sheet = Image.new("RGB", (COLUMNS * TILE_SIZE[0], ROWS * (TILE_SIZE[1] + TEXT_HEIGHT)), "lightgray")
        for offset, row in enumerate(labels.iloc[start:start + per_sheet].itertuples()):
            x = (offset % COLUMNS) * TILE_SIZE[0]
            y = (offset // COLUMNS) * (TILE_SIZE[1] + TEXT_HEIGHT)
            sheet.paste(draw_tile(row.tile, images[row.id], row, font), (x, y))
        sheet.save(AUDIT_DIR / f"sheet_{start // per_sheet + 1}.png")

    labels.to_csv(AUDIT_PATH, index=False)
    print(f"{len(labels)} products on {-(-len(labels) // per_sheet)} sheets in {AUDIT_DIR}, fill in {AUDIT_PATH}")


if __name__ == "__main__":
    main()
