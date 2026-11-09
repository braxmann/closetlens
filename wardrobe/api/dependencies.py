# Shared by the JSON API (main.py) and the HTML pages (web/pages.py).
# In its own file because main.py imports the pages, so the pages cannot import main.py back.
from sqlmodel import Session

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
