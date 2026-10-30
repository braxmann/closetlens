"""Cut out photos, describe them and write output/<name>.png and output/<name>.json.

Run from the project root: .venv/bin/python -m scripts.analyze data/photos/IMG_0265.jpeg
"""
import argparse
from pathlib import Path

from PIL import Image

from wardrobe.models.attribute_recognizer import AttributeRecognizer, OllamaAttributeRecognizer
from wardrobe.models.background_remover import BackgroundRemover, RembgBackgroundRemover

OUTPUT_DIR = Path("output")


def process_photo(
    photo: Path,
    remover: BackgroundRemover,
    recognizer: AttributeRecognizer,
) -> None:
    input = Image.open(photo)
    output = remover.rem_background(input)
    
    attributes = recognizer.recognize_attributes(output)
    output.save(OUTPUT_DIR / f"{photo.stem}.png")
    (OUTPUT_DIR / f"{photo.stem}.json").write_text(
        attributes.model_dump_json(indent=2), encoding="utf-8"
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Cut out and describe garment photos.")
    parser.add_argument("photos", nargs="+", type=Path, help="one or more photo files")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)

    # The only place that knows the concrete models. Created once, before the loop.
    remover = RembgBackgroundRemover("birefnet-general")
    recognizer = OllamaAttributeRecognizer("qwen2.5vl:3b")

    for photo in args.photos:
        process_photo(photo, remover, recognizer)
        print(f"Done: {photo.name}")


if __name__ == "__main__":
    main()
