from io import BytesIO

import numpy as np
import pytest
from PIL import Image
from sqlmodel import Session
from pathlib import Path

from wardrobe.models.attributes import GarmentAttributes
from wardrobe.services.garments import (
    correct_garment, create_garment, process_garment, search_garments, similar_garments,
)
from wardrobe.storage.database import Garment, GarmentStatus, create_db
from wardrobe.storage.files import FileStorage


class FakeBackgroundRemover:
    def rem_background(self, image):
        return image.convert("RGBA")


class FailingBackgroundRemover:
    def rem_background(self, image):
        raise RuntimeError("model failed")


class FakeRecognizer:
    def recognize_attributes(self, image):
        return GarmentAttributes(category="pants", main_color="blue", pattern="solid")

class FakeEmbedder:
    name = "fake-embedder"

    def embed_images(self, images):
        return np.array([[0.6, 0.8] for _ in images])

    def embed_texts(self, texts):
        return np.array([[1.0, 0.0] for _ in texts])


class CorrectingRecognizer:
    def __init__(self, session, garment):
        self.session = session
        self.garment = garment

    def recognize_attributes(self, image):
        correction = GarmentAttributes(category="dress", main_color="red", pattern="dotted")
        correct_garment(self.session, self.garment, correction)
        return GarmentAttributes(category="pants", main_color="blue", pattern="solid")



def photo_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def session(tmp_path):
    engine = create_db(tmp_path / "test.db")
    with Session(engine) as session:
        yield session

@pytest.fixture
def storage(tmp_path):
    return FileStorage(tmp_path)


def test_create_garment_is_pending(session, storage):
    garment = create_garment(session,storage,photo_bytes())
    assert garment.status == GarmentStatus.PENDING
    assert Path(garment.photo_path).exists()
    

def test_process_garment_saves_attributes(session, storage):
    garment = create_garment(session,storage,photo_bytes())
    process_garment(session, storage, garment, FakeBackgroundRemover(), FakeRecognizer(), FakeEmbedder())
    assert garment.status == GarmentStatus.DONE
    assert garment.category == "pants"
    assert Path(garment.cutout_path).exists()
    assert (storage.base_dir / "thumbnails" / Path(garment.cutout_path).name).exists()


def test_process_garment_saves_embedding(session, storage):
    garment = create_garment(session, storage, photo_bytes())
    process_garment(session, storage, garment, FakeBackgroundRemover(), FakeRecognizer(), FakeEmbedder())
    session.refresh(garment)  # read back what is really in the database
    assert garment.embedding_model == "fake-embedder"
    assert np.frombuffer(garment.embedding, dtype=np.float32).tolist() == pytest.approx([0.6, 0.8])


def test_process_garment_marks_failure(session, storage):
    garment = create_garment(session, storage, photo_bytes())
    process_garment(session, storage, garment, FailingBackgroundRemover(), FakeRecognizer(), FakeEmbedder())
    assert garment.status == GarmentStatus.FAILED

def test_process_garment_keeps_correction(session, storage):
    garment = create_garment(session, storage, photo_bytes())
    recognizer = CorrectingRecognizer(session, garment)
    process_garment(session, storage, garment, FakeBackgroundRemover(), recognizer, FakeEmbedder())
    assert garment.category == "dress"
    assert Path(garment.cutout_path).exists()
    # The correction only concerns the attributes; the image embedding is stored anyway.
    assert garment.embedding is not None


def add_embedded(session, vector, model="fake-embedder") -> Garment:
    garment = Garment(
        photo_path="x.jpeg", status=GarmentStatus.DONE, embedding_model=model,
        embedding=np.array(vector, dtype=np.float32).tobytes(),
    )
    session.add(garment)
    session.commit()
    session.refresh(garment)
    return garment


def test_similar_garments_most_similar_first_without_itself(session):
    shirt = add_embedded(session, [1.0, 0.0])
    close = add_embedded(session, [0.8, 0.6])
    far = add_embedded(session, [0.0, 1.0])
    add_embedded(session, [1.0, 0.0], model="other-model")  # other vector space, never compared
    assert [g.id for g in similar_garments(session, shirt, k=5)] == [close.id, far.id]


def test_similar_garments_without_embedding_is_empty(session, storage):
    garment = create_garment(session, storage, photo_bytes())
    assert similar_garments(session, garment) == []


def test_search_garments_finds_the_matching_image(session):
    # FakeEmbedder embeds every text as [1, 0].
    match = add_embedded(session, [1.0, 0.0])
    add_embedded(session, [0.0, 1.0])
    assert search_garments(session, FakeEmbedder(), "blue linen shirt", k=1) == [match]


def test_search_without_any_embeddings_is_empty(session):
    assert search_garments(session, FakeEmbedder(), "anything") == []
