# Shared by the JSON API (main.py) and the HTML pages (web/pages.py).
# In its own file because main.py imports the pages, so the pages cannot import main.py back.
from fastapi import Request
from sqlmodel import Session

from wardrobe.models.embedder import Embedder
from wardrobe.storage.database import create_db
from wardrobe.storage.files import FileStorage

engine = create_db()
storage = FileStorage()


def get_session():
    with Session(engine) as session:
        yield session


# A dependency like get_session, so tests can swap it for a temporary folder.
def get_storage() -> FileStorage:
    return storage


def get_embedder(request: Request) -> Embedder | None:
    """The embedding model the worker uses too, loaded once in main.py's lifespan.

    None while the lifespan has not run, for example in tests that do not override this dependency.
    """
    return getattr(request.app.state, "embedder", None)
