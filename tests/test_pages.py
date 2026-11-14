from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlmodel import Session

from wardrobe.api.dependencies import get_embedder
from wardrobe.api.main import app, get_session, get_storage
from wardrobe.storage.database import Garment, create_db
from wardrobe.storage.files import FileStorage


def photo_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, format="JPEG")
    return buffer.getvalue()


class FakeEmbedder:
    name = "fake-embedder"

    def embed_texts(self, texts):
        return np.array([[1.0, 0.0] for _ in texts])


def set_embedding(client, garment_id: int, vector: list[float]) -> None:
    """Stores an embedding directly, as the worker would."""
    with Session(client.engine) as session:
        garment = session.get(Garment, garment_id)
        garment.embedding = np.array(vector, dtype=np.float32).tobytes()
        garment.embedding_model = "fake-embedder"
        session.add(garment)
        session.commit()


@pytest.fixture
def client(tmp_path):
    engine = create_db(tmp_path / "test.db")

    def test_session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[get_session] = test_session
    app.dependency_overrides[get_storage] = lambda: FileStorage(tmp_path)
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    # Without "with TestClient(...)" the lifespan does not run, so no worker and no real models.
    client = TestClient(app)
    client.engine = engine
    yield client
    app.dependency_overrides.clear()


def upload(client) -> str:
    response = client.post("/upload", files={"photo": ("photo.jpeg", photo_bytes(), "image/jpeg")})
    assert response.status_code == 200
    return response.text


def test_empty_gallery(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "0 Teile" in response.text


def test_upload_returns_polling_icon_and_count(client):
    html = upload(client)
    assert 'id="garment-1"' in html
    assert 'hx-trigger="every 2s"' in html  # still pending, so it keeps asking
    assert "1 Teil<" in html


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


def test_search_ranks_best_match_first(client):
    for _ in range(3):
        upload(client)
    set_embedding(client, 1, [0.0, 1.0])
    set_embedding(client, 2, [1.0, 0.0])  # FakeEmbedder embeds every text as [1, 0]
    # garment 3 has no embedding yet and cannot be found
    html = client.get("/", params={"q": "red shirt"}).text
    assert html.index('id="garment-2"') < html.index('id="garment-1"')
    assert 'id="garment-3"' not in html


def test_detail_shows_similar_garments(client):
    upload(client)
    upload(client)
    set_embedding(client, 1, [1.0, 0.0])
    set_embedding(client, 2, [0.8, 0.6])
    html = client.get("/items/1").text
    assert "Ähnliche Teile" in html
    assert 'id="garment-2"' in html
