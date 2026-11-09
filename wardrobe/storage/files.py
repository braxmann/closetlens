from pathlib import Path
from uuid import uuid4

from PIL import Image

BASE_DIR = Path("data")
THUMBNAIL_SIZE = (256, 256)


class FileStorage:
    def __init__(self, base_dir: Path = BASE_DIR):
        self.base_dir = base_dir

    def _new_path(self, folder: str, suffix: str) -> Path:
        # Random names, because two uploads can share a file name like IMG_0001.jpeg.
        directory = self.base_dir / folder
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{uuid4().hex}{suffix}"

    def save_photo(self, data: bytes) -> Path:
        path = self._new_path("originals", ".jpeg")
        path.write_bytes(data)
        return path

    def save_cutout(self, image: Image.Image) -> Path:
        path = self._new_path("cutouts", ".png")
        image.save(path)

        # Small copy for the gallery, same file name in its own folder, so no database column is needed.
        thumbnail = image.copy()  # copy first: thumbnail() changes the image in place and returns None
        thumbnail.thumbnail(THUMBNAIL_SIZE)  # keeps the aspect ratio
        thumbnail_dir = self.base_dir / "thumbnails"
        thumbnail_dir.mkdir(parents=True, exist_ok=True)
        thumbnail.save(thumbnail_dir / path.name)
        return path
