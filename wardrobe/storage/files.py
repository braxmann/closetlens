from pathlib import Path
from uuid import uuid4

from PIL import Image

BASE_DIR = Path("data")


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
        return path
