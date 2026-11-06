from io import BytesIO
from typing import Protocol

from ollama import chat
from PIL import Image

from wardrobe.models.attributes import GarmentAttributes

# The allowed values come from the schema passed as `format`, so the prompt only states the rules.
PROMPT = """Describe the single garment in this image. Judge only what is visible.

Rules:
- main_color: the color covering the largest area. Use "multicolor" only if no color dominates.
  For stripes, use "multicolor" if both colors take up about the same area.
- pattern: "graphic" means a print or logo. A small logo on an otherwise plain garment is still "solid".
- brand: only a brand name you can actually read as text or a logo on the garment. Otherwise use null.
  Never guess the brand from the style of the garment.
- material, formality: use null if you cannot judge it from the image.
- Use "other" only if you can recognize the value, but it is not one of the allowed options."""

# The model does not need the full 24 MP, and smaller images are sent and processed faster.
MAX_IMAGE_SIZE = (1024, 1024)


class AttributeRecognizer(Protocol):
    def recognize_attributes(self, image: Image.Image) -> GarmentAttributes:
        ...


class OllamaAttributeRecognizer:
    def __init__(self, model: str, background: str = "white"):
        self.model = model
        self.background = background

    def _to_jpeg_bytes(self, image: Image.Image) -> bytes:
        # JPEG has no alpha channel, so transparent areas must be filled with a known color first.
        rgba = image.convert("RGBA")
        flattened = Image.new("RGBA", rgba.size, self.background)
        flattened.alpha_composite(rgba)
        flattened.thumbnail(MAX_IMAGE_SIZE)

        buffer = BytesIO()
        flattened.convert("RGB").save(buffer, format="JPEG", quality=90)
        return buffer.getvalue()

    def recognize_attributes(self, image: Image.Image) -> GarmentAttributes:
        response = chat(
            model=self.model,
            messages=[{"role": "user", "content": PROMPT, "images": [self._to_jpeg_bytes(image)]}],
            format=GarmentAttributes.model_json_schema(),  # forces JSON that matches the schema
            options={"temperature": 0},  # same input, same answer: needed for comparing runs
        )
        # Validates the answer, so a broken response fails here and not later in the database.
        return GarmentAttributes.model_validate_json(response.message.content)
