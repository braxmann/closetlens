import numpy as np
from PIL import Image

from wardrobe.training.features import embed_in_batches, load_features, save_features


class CountingEmbedder:
    """Embeds image number i (stored in its width) as [i, 0] and counts the calls."""

    def __init__(self):
        self.calls = 0

    def embed_images(self, images):
        self.calls += 1
        return np.array([[image.width, 0] for image in images], dtype=np.float32)


def test_embed_in_batches_keeps_order_and_includes_the_last_small_batch():
    images = [Image.new("RGB", (width, 1)) for width in range(1, 151)]
    embedder = CountingEmbedder()
    vectors = embed_in_batches(embedder, images, batch_size=64)
    assert vectors.shape == (150, 2)
    assert vectors[:, 0].tolist() == list(range(1, 151))
    assert embedder.calls == 3  # 64 + 64 + 22


def test_features_round_trip(tmp_path):
    ids, vectors = np.array([3, 1]), np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    save_features(tmp_path / "f.npz", ids, vectors)
    loaded_ids, loaded_vectors = load_features(tmp_path / "f.npz")
    assert loaded_ids.tolist() == [3, 1]
    assert np.array_equal(loaded_vectors, vectors)
