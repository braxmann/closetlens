from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlmodel import Session

from wardrobe.api.main import app, get_session, get_storage
from wardrobe.storage.database import create_db
from wardrobe.storage.files import FileStorage

CORRECTION = {"category": "pants", "main_color": "blue", "pattern": "solid"}


def photo_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def client(tmp_path):
    engine = create_db(tmp_path / "test.db")

    def test_session():
        with Session(engine) as session:
            yield session

    # Swap the real database and folder for temporary ones.
    app.dependency_overrides[get_session] = test_session
    app.dependency_overrides[get_storage] = lambda: FileStorage(tmp_path)
    # Without "with TestClient(...)" the lifespan does not run, so no worker and no real models.
    yield TestClient(app)
    app.dependency_overrides.clear()


def upload(client) -> dict:
    response = client.post("/garments", files={"photo": ("photo.jpeg", photo_bytes(), "image/jpeg")})
    assert response.status_code == 200
    return response.json()


def test_upload_is_pending_and_listed(client):
    garment = upload(client)
    assert garment["status"] == "pending"
    assert [g["id"] for g in client.get("/garments").json()] == [garment["id"]]


def test_upload_without_photo_is_rejected(client):
    assert client.post("/garments").status_code == 422


def test_get_unknown_garment_is_404(client):
    assert client.get("/garments/999").status_code == 404


def test_correction_sets_attributes_and_done(client):
    garment = upload(client)
    response = client.patch(f"/garments/{garment['id']}", json=CORRECTION)
    assert response.status_code == 200
    assert response.json()["category"] == "pants"
    assert response.json()["status"] == "done"


def test_correction_with_invalid_value_is_rejected(client):
    garment = upload(client)
    response = client.patch(f"/garments/{garment['id']}", json={**CORRECTION, "category": "Hose"})
    assert response.status_code == 422
