import numpy as np
import torch
from torch import nn

from wardrobe.training.finetune import (
    MISSING, MultiHeadClassifier, SmoothedCrossEntropy, balanced_weights, encode_labels,
)


def test_unknown_and_missing_labels_become_missing():
    labels = encode_labels(np.array(["shirt", None, "dress", "hat"], dtype=object), ["dress", "shirt"])
    assert labels.tolist() == [1, MISSING, 0, MISSING]


def test_rare_classes_get_more_weight():
    weights = balanced_weights(np.array([0, 0, 0, 1, MISSING]), n_classes=2)
    assert weights[1] > weights[0]
    # Same formula as scikit-learn's class_weight="balanced": 4 known labels / (2 classes * count)
    assert torch.allclose(weights, torch.tensor([4 / 6, 4 / 2]))


def test_one_output_per_attribute_with_one_score_per_class():
    # A tiny stand-in for the image tower: 3 input numbers -> 8 features.
    model = MultiHeadClassifier(nn.Linear(3, 8), feature_dim=8, classes={"category": ["a", "b", "c"], "formality": ["x", "y"]})

    outputs = model(torch.randn(5, 3))

    assert outputs["category"].shape == (5, 3)
    assert outputs["formality"].shape == (5, 2)


def test_missing_labels_do_not_count_in_the_loss():
    criterion = nn.CrossEntropyLoss(ignore_index=MISSING)
    logits = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
    # The second row would be completely wrong, but it is MISSING and so ignored.
    with_missing = criterion(logits, torch.tensor([0, MISSING]))
    only_first = criterion(logits[:1], torch.tensor([0]))
    assert torch.isclose(with_missing, only_first)


def test_without_smoothing_it_equals_pytorch_cross_entropy():
    weight = torch.tensor([1.0, 5.0, 20.0])
    logits = torch.randn(6, 3)
    target = torch.tensor([0, 1, 2, MISSING, 1, 0])
    expected = nn.CrossEntropyLoss(weight=weight, ignore_index=MISSING)(logits, target)
    assert torch.isclose(SmoothedCrossEntropy(weight)(logits, target), expected)


def test_smoothing_keeps_a_confident_correct_answer_cheap_despite_large_weights():
    # The case that broke PyTorch's built-in version: a correct answer for a common class,
    # while a rare class has a weight of 44.
    weight = torch.tensor([0.3, 44.0])
    logits = torch.tensor([[5.0, -5.0]])
    target = torch.tensor([0])
    loss = SmoothedCrossEntropy(weight, smoothing=0.1)(logits, target)
    # (1 - 0.1) * ~0 + 0.1 * (~0 + 10) / 2 = ~0.5, independent of the weight 44
    assert loss < 1.0
    assert nn.CrossEntropyLoss(weight=weight, label_smoothing=0.1)(logits, target) > 10
