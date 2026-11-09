from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlmodel import Session

from wardrobe.api.main import app, get_session, get_storage
from wardrobe.storage.database import create_db
from wardrobe.storage.files import FileStorage


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

    app.dependency_overrides[get_session] = test_session
    app.dependency_overrides[get_storage] = lambda: FileStorage(tmp_path)
    # Without "with TestClient(...)" the lifespan does not run, so no worker and no real models.
    yield TestClient(app)
    app.dependency_overrides.clear()


def upload(client) -> str:
    response = client.post("/upload", files={"photo": ("photo.jpeg", photo_bytes(), "image/jpeg")})
    assert response.status_code == 200
    return response.text


def test_empty_gallery(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "0 Objekte" in response.text


def test_upload_returns_polling_icon_and_count(client):
    html = upload(client)
    assert 'id="garment-1"' in html
    assert 'hx-trigger="every 2s"' in html  # still pending, so it keeps asking
    assert "1 Objekte" in html


def test_edit_saves_correction_and_stops_polling(client):
    upload(client)
    form = {"category": "pants", "main_color": "blue", "pattern": "solid", "brand": "null", "material": ""}
    response = client.post("/items/1", data=form, follow_redirects=False)
    assert response.status_code == 303

    icon = client.get("/items/1/icon").text
    assert "Hose" in icon
    assert "hx-trigger" not in icon  # done, so polling stops


def test_edit_rejects_invalid_value(client):
    upload(client)
    form = {"category": "Hose", "main_color": "blue", "pattern": "solid"}
    assert client.post("/items/1", data=form).status_code == 422


def test_filter_by_category(client):
    upload(client)
    upload(client)
    client.post("/items/1", data={"category": "pants", "main_color": "blue", "pattern": "solid"})
    html = client.get("/", params={"category": "pants"}).text
    assert 'id="garment-1"' in html
    assert 'id="garment-2"' not in html


def test_unknown_item_is_404(client):
    assert client.get("/items/999").status_code == 404
