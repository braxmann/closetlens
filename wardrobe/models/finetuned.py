"""Attribute recognizer with the fine-tuned Marqo image tower (M6, scripts/finetune.py).

The fine-tuned model answers the attributes it has a head for (category, main_color, formality).
pattern and material come from zero-shot with the *original* embedder: fine-tuning changed the image tower,
so its vectors no longer live in the same space as the text vectors, and comparing them with text would be meaningless.
"""
from pathlib import Path
from typing import cast

import open_clip
import torch
from PIL import Image
from torch import nn

from wardrobe.models.attributes import GarmentAttributes
from wardrobe.models.embedder import Embedder
from wardrobe.models.zero_shot import ZeroShotAttributeRecognizer
from wardrobe.training.finetune import MultiHeadClassifier, add_lora, eval_transform


def load_finetuned(path: Path, device: str) -> tuple[MultiHeadClassifier, dict[str, list[str]]]:
    """Rebuild the model exactly as it was trained, load the trained weights into it.

    Returns the model and the class names per attribute: output i of a head means the i-th name.
    """
    checkpoint = torch.load(path, map_location=device)
    clip_model, _, _ = open_clip.create_model_and_transforms(checkpoint["model"])
    visual = cast(nn.Module, clip_model.visual)
    del clip_model
    if checkpoint.get("method") == "lora":
        # The LoRA layers must exist before loading, otherwise their weights have nowhere to go.
        add_lora(visual, checkpoint["lora_rank"], checkpoint["lora_alpha"])
    model = MultiHeadClassifier(visual, feature_dim=768, classes=checkpoint["classes"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint["classes"]


def flatten_on_white(image: Image.Image) -> Image.Image:
    """Cut-outs from the app are transparent around the garment; the training images had white backgrounds."""
    if image.mode in ("RGBA", "LA", "P"):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return image.convert("RGB")


class FinetunedAttributeRecognizer:
    def __init__(self, checkpoint: Path, embedder: Embedder):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.classes = load_finetuned(checkpoint, self.device)
        self.transform = eval_transform()
        self.zero_shot = ZeroShotAttributeRecognizer(embedder)
        self.embedder = embedder

    def recognize_attributes(self, image: Image.Image) -> GarmentAttributes:
        image = flatten_on_white(image)
        answers = self.zero_shot.answers_for(self.embedder.embed_images([image])[0])
        # unsqueeze(0): the model expects a batch, so one image becomes a batch of one.
        batch = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(batch)
        for attribute, logits in outputs.items():
            answers[attribute] = self.classes[attribute][int(logits.argmax(dim=1))]
        return GarmentAttributes.model_validate(answers)
