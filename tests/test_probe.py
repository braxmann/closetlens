import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression

from tests.test_zero_shot import FakeEmbedder
from wardrobe.models.probe import ProbeAttributeRecognizer
from wardrobe.training.linear_probe import load_probes, save_probes


def category_probe() -> LogisticRegression:
    # The fake embedder returns [1, 0] for every image; this probe learned [1, 0] -> "sweater".
    X = np.array([[1, 0], [0, 1]])
    return LogisticRegression().fit(X, ["sweater", "jacket"])


def test_probe_answers_its_attribute_and_zero_shot_the_rest():
    # Zero-shot alone would say "shirt" (matching text) and "blue".
    embedder = FakeEmbedder({"shirt", "a photo of a blue colored garment"})
    recognizer = ProbeAttributeRecognizer(embedder, {"category": category_probe()})

    attributes = recognizer.recognize_attributes(Image.new("RGB", (10, 10)))

    assert attributes.category == "sweater"  # from the probe, not "shirt" from zero-shot
    assert attributes.main_color == "blue"  # no probe for main_color, so zero-shot answers
    assert attributes.brand is None


def test_saved_probes_load_with_their_model_name(tmp_path):
    path = tmp_path / "probes.joblib"
    save_probes(path, "fake-model", {"category": category_probe()})

    model, probes = load_probes(path)

    assert model == "fake-model"
    assert probes["category"].predict(np.array([[1, 0]]))[0] == "sweater"
