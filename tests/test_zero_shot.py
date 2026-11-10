import numpy as np
from PIL import Image

from wardrobe.models.zero_shot import ZeroShotAttributeRecognizer


class FakeEmbedder:
    def __init__(self, matching_texts: set[str]):
        self.matching_texts = matching_texts

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        return np.array([[1, 0] for _ in images])

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        # One row per text: [1, 0] if the text is in matching_texts, else [0, 1]
        return np.array([[1, 0] if text in self.matching_texts else [0, 1] for text in texts])


def test_picks_the_matching_value_for_every_attribute():
    embedder = FakeEmbedder({
        "shirt",
        "a photo of a blue colored garment",
        "a photo of a garment with a checked pattern",
        "a photo of a garment made of cotton",
        "a photo of casual clothing",
    })
    recognizer = ZeroShotAttributeRecognizer(embedder)

    attributes = recognizer.recognize_attributes(Image.new("RGB", (10, 10)))

    assert attributes.category == "shirt"
    assert attributes.main_color == "blue"
    assert attributes.pattern == "checked"
    assert attributes.material == "cotton"
    assert attributes.formality == "casual"
    assert attributes.brand is None


def test_never_answers_other():
    # Even if "other" were the most similar text, it is not a class the recognizer chooses from.
    embedder = FakeEmbedder({"other"})
    recognizer = ZeroShotAttributeRecognizer(embedder)

    attributes = recognizer.recognize_attributes(Image.new("RGB", (10, 10)))

    assert attributes.category != "other"
