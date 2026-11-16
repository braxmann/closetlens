"""Partial fine-tuning (M6): the Marqo image tower learns too, not only a classifier on top of it.

Linear probe: frozen embedder -> features computed once -> logistic regression on the CPU.
Fine-tuning:  every image goes through the network in every epoch, forward and backward, on the GPU,
              and the last transformer blocks change their weights for our attributes.

One model with one output layer ("head") per attribute: the image is encoded once and all heads read the same vector.
"""
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms

# Marks "no label for this attribute" (e.g. a product whose usage is not mapped). CrossEntropyLoss skips it.
MISSING = -100
# The Marqo/SigLIP preprocessing: 224 x 224 pixels, values scaled from [0, 1] to [-1, 1].
IMAGE_SIZE = 224
MEAN, STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]


def train_transform() -> transforms.Compose:
    """Augmentation: every epoch sees a slightly different version of each image, which makes memorising harder.

    Only changes that keep the labels true: a mirrored shirt is still a shirt. No colour changes,
    because main_color is one of the labels and a "brightened" navy shirt might really look blue.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform() -> transforms.Compose:
    """No randomness when measuring: the same image must always give the same answer."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def encode_labels(values: np.ndarray, classes: list[str]) -> np.ndarray:
    """Label strings -> class numbers; unknown or missing values become MISSING."""
    index = {name: number for number, name in enumerate(classes)}
    return np.array([index.get(value, MISSING) for value in values], dtype=np.int64)


class GarmentDataset(Dataset):
    """Images as compressed bytes plus one label column per attribute.

    Bytes instead of opened images: the DataLoader copies the dataset into every worker process,
    and bytes can be copied (pickled) while PIL images that are still open cannot.
    """

    def __init__(self, images: list[bytes], labels: dict[str, np.ndarray], transform: transforms.Compose):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, dict[str, int]]:
        image = Image.open(BytesIO(self.images[i])).convert("RGB")
        return self.transform(image), {attribute: int(column[i]) for attribute, column in self.labels.items()}


class MultiHeadClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feature_dim: int, classes: dict[str, list[str]]):
        super().__init__()
        self.backbone = backbone
        # ModuleDict instead of a normal dict: only then does PyTorch see the heads as part of the model
        # (their weights move to the GPU with .to(), are returned by .parameters() and saved in state_dict()).
        self.heads = nn.ModuleDict({attribute: nn.Linear(feature_dim, len(names)) for attribute, names in classes.items()})

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(images)
        # "Logits": one raw score per class. softmax would turn them into probabilities, argmax picks the answer.
        return {attribute: head(features) for attribute, head in self.heads.items()}


def freeze_except_last_blocks(visual: nn.Module, n_blocks: int) -> None:
    """Freeze the Marqo image tower except its last n transformer blocks and the layers after them.

    The early blocks detect general things (edges, textures) that stay useful; the late blocks combine them into
    concepts and are the ones worth adapting. Frozen layers need no gradients, which also saves VRAM.
    """
    for parameter in visual.parameters():
        parameter.requires_grad = False
    trunk = visual.trunk
    trainable = list(trunk.blocks[len(trunk.blocks) - n_blocks:]) + [trunk.norm, trunk.attn_pool, visual.head]
    for module in trainable:
        for parameter in module.parameters():
            parameter.requires_grad = True


def balanced_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """Loss weight per class, n_samples / (n_classes * count): rare classes count more.

    The same formula as class_weight="balanced" in the linear probe, so both are judged under the same rules.
    """
    known = labels[labels != MISSING]
    counts = np.bincount(known, minlength=n_classes).astype(np.float32)
    return torch.tensor(len(known) / (n_classes * np.maximum(counts, 1)))


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(trainable, total) number of weights."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
