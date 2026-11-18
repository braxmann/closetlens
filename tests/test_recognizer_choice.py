from pathlib import Path

from tests.test_zero_shot import FakeEmbedder
from wardrobe.api import main
from wardrobe.models.zero_shot import ZeroShotAttributeRecognizer


def test_falls_back_to_zero_shot_without_a_checkpoint(monkeypatch):
    # A fresh machine has no data/models/: the server must still start and recognize garments.
    monkeypatch.setattr(main, "FINETUNED_CHECKPOINT", Path("does/not/exist.pt"))

    recognizer = main.make_recognizer(FakeEmbedder(set()))

    assert isinstance(recognizer, ZeroShotAttributeRecognizer)
