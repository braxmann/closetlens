from typing import Protocol

import numpy as np
import open_clip
import torch
from PIL import Image


class Embedder(Protocol):
    name:str
    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        ...

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        ...


class OpenClipEmbedder:
    def __init__(self, model: str):
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        # The training transforms (random augmentation) are not needed for embedding.
        self.model, _, self.preprocess_val = open_clip.create_model_and_transforms(model)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model)
        self.name = model

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        batch = torch.stack([self.preprocess_val(image) for image in images]).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(batch, normalize=True)
        # numpy can only read tensors in CPU memory.
        return features.cpu().numpy()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens, normalize=True)
        return features.cpu().numpy()
