from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from sqlalchemy import Engine
from sqlmodel import Field, SQLModel, create_engine

DATABASE_PATH = Path("data/wardrobe.db")


class GarmentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class Garment(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    photo_path: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: GarmentStatus = Field(default=GarmentStatus.PENDING, index=True)

    # Empty until the worker has processed the photo.
    cutout_path: str | None = None

    # One column per attribute, so the gallery can filter on them. Values are validated by GarmentAttributes.
    category: str | None = Field(default=None, index=True)
    main_color: str | None = Field(default=None, index=True)
    pattern: str | None = None
    brand: str | None = None
    material: str | None = None
    formality: str | None = None

    # Image embedding of the cutout as float32 bytes (np.frombuffer(..., dtype=np.float32)), set by the worker.
    # exclude=True: raw bytes cannot be written as JSON, and API clients have no use for them.
    embedding: bytes | None = Field(default=None, exclude=True)
    embedding_model:str| None = None

def create_db(path: Path = DATABASE_PATH) -> Engine:
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{path}")
    SQLModel.metadata.create_all(engine)
    return engine
