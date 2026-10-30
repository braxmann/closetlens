"""Cut out all photos in data/photos and save them as PNGs in output/.

Run from the project root: .venv/bin/python -m scripts.try_background_remover
"""
from pathlib import Path
from time import perf_counter

from PIL import Image

from wardrobe.models.background_remover import RembgBackgroundRemover

PHOTO_DIR = Path("data/photos")
OUTPUT_DIR = Path("output")
MODEL = "birefnet-general"

OUTPUT_DIR.mkdir(exist_ok=True)

start = perf_counter()
remover = RembgBackgroundRemover(MODEL)
print(f"Model loaded in {perf_counter() - start:.1f} s")

for path in sorted(PHOTO_DIR.glob("*.jpeg")):
    start = perf_counter()
    result = remover.rem_background(Image.open(path))
    result.save(OUTPUT_DIR / f"{path.stem}.png")
    print(f"{path.name}: {perf_counter() - start:.1f} s, {result.mode} {result.size}")
