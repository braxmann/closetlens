"""Attribute recognizer with trained linear probes (M6), zero-shot for every attribute without a probe.

The learning curve showed that no method wins everywhere: the probe is better for category and formality,
zero-shot for main_color. So this recognizer asks the probe where one was saved and zero-shot for the rest.
"""
from PIL import Image
from sklearn.linear_model import LogisticRegression

from wardrobe.models.attributes import GarmentAttributes
from wardrobe.models.embedder import Embedder
from wardrobe.models.zero_shot import ZeroShotAttributeRecognizer


class ProbeAttributeRecognizer:
    def __init__(self, embedder: Embedder, probes: dict[str, LogisticRegression]):
        self.embedder = embedder
        self.probes = probes
        # Both use the same embedder, so the image is embedded once and both read the same vector.
        self.zero_shot = ZeroShotAttributeRecognizer(embedder)

    def recognize_attributes(self, image: Image.Image) -> GarmentAttributes:
        image_vector = self.embedder.embed_images([image])[0]
        answers = self.zero_shot.answers_for(image_vector)
        for attribute, probe in self.probes.items():
            # scikit-learn expects a table with one row per sample, so the single vector becomes a 1-row matrix.
            answers[attribute] = probe.predict(image_vector.reshape(1, -1))[0]
        return GarmentAttributes.model_validate(answers)
