from io import BytesIO

import pytest
from PIL import Image
from sqlmodel import Session
from pathlib import Path

from wardrobe.models.attributes import GarmentAttributes
from wardrobe.services.garments import create_garment, process_garment, correct_garment
from wardrobe.storage.database import GarmentStatus, create_db
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
    process_garment(session, storage, garment, FakeBackgroundRemover(), FakeRecognizer())
    assert garment.status == GarmentStatus.DONE
    assert garment.category == "pants"
    assert Path(garment.cutout_path).exists()
    assert (storage.base_dir / "thumbnails" / Path(garment.cutout_path).name).exists()


def test_process_garment_marks_failure(session, storage):
    garment = create_garment(session, storage, photo_bytes())
    process_garment(session, storage, garment, FailingBackgroundRemover(), FakeRecognizer())
    assert garment.status == GarmentStatus.FAILED

def test_process_garment_keeps_correction(session, storage):
    garment = create_garment(session, storage, photo_bytes())
    recognizer = CorrectingRecognizer(session, garment)
    process_garment(session, storage, garment, FakeBackgroundRemover(), recognizer)
    assert garment.category == "dress"
    assert Path(garment.cutout_path).exists()
