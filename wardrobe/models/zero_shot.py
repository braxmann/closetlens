from typing import get_args

import numpy as np
from PIL import Image

from wardrobe.models.attributes import Category, Color, Formality, GarmentAttributes, Material, Pattern
from wardrobe.models.embedder import Embedder

# Attribute -> (allowed values, text template). "{}" is replaced by the value.
# Tuned on the validation set (M5): the bare word separates t-shirt from shirt much better than
# "a photo of a {}", and "colored" makes the model pick multicolor less often. The others are first guesses.
TEMPLATES = {
    "category": (get_args(Category), "{}"),
    "main_color": (get_args(Color), "a photo of a {} colored garment"),
    "pattern": (get_args(Pattern), "a photo of a garment with a {} pattern"),
    "material": (get_args(Material), "a photo of a garment made of {}"),
    "formality": (get_args(Formality), "a photo of {} clothing"),
}


class ZeroShotAttributeRecognizer:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        # Attribute -> (values, text embeddings with one row per value)
        self.classes: dict[str, tuple[list[str], np.ndarray]] = {}
        for attribute, (values, template) in TEMPLATES.items():
            values_n = [x for x in values if x !="other"]
            texts = [template.format(value) for value in values_n]
            self.classes[attribute] = (values_n, self.embedder.embed_texts(texts))


    def recognize_attributes(self, image: Image.Image) -> GarmentAttributes:
        image_vector = self.embedder.embed_images([image])[0]
        # model_validate takes a dict and checks every value against the schema, like GarmentAttributes(**...).
        return GarmentAttributes.model_validate(self.answers_for(image_vector))

    def answers_for(self, image_vector: np.ndarray) -> dict[str, str | None]:
        """Attribute -> value for an image that is already embedded.

        Separate from recognize_attributes, so the probe recognizer can reuse it without embedding the image twice.
        """
        answers = {}
        for attribute, (values, text_embeddings) in self.classes.items():
            # Similarity to every value of this attribute, pick the most similar one.
            similarities = text_embeddings @ image_vector
            best_fit = np.argmax(similarities)
            answers[attribute] = values[best_fit]
        answers["brand"] = None
        return answers
