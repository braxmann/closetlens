import logging

import numpy as np
from PIL import Image
from sqlmodel import select

from wardrobe.storage.database import Garment, GarmentStatus

logger = logging.getLogger(__name__)


def create_garment(session, storage, data) -> Garment:
    path = storage.save_photo(data)
    garment = Garment(photo_path=str(path), status="pending")
    session.add(garment)
    session.commit()
    session.refresh(garment)
    return garment

def process_garment(session, storage, garment, remover, recognizer, embedder) -> None:
    garment.status = GarmentStatus.PROCESSING
    session.add(garment)
    session.commit()
    garment_id = garment.id
    try:
        image = Image.open(garment.photo_path)
        cutout = remover.rem_background(image)
        attributes = recognizer.recognize_attributes(cutout)
        embedding = embedder.embed_images([cutout])[0]
        path = storage.save_cutout(cutout)
        # Reload before changing anything: a correction may have arrived while the models ran.
        session.refresh(garment)
        garment.embedding = embedding.astype(np.float32).tobytes()
        garment.embedding_model = embedder.name
        garment.cutout_path = str(path)
        if garment.status != GarmentStatus.PROCESSING:
            # Corrected in the meantime: keep the correction, only add the cutout.
            session.add(garment)
            session.commit()
            return

        for field, value in attributes.model_dump().items():
            setattr(garment, field, value)
        
        garment.status = GarmentStatus.DONE
        session.add(garment)
        session.commit()
    except Exception:
        # A failed photo must not stop the worker, so the error ends here and is only logged.
        logger.exception("Processing failed for garment %s", garment_id)
        session.rollback()  # discards half-written changes from the try block
        garment.status = GarmentStatus.FAILED
        session.add(garment)
        session.commit()


def correct_garment(session, garment, correction) -> Garment:
    for field, value in correction.model_dump().items():
        setattr(garment, field, value)
    # A manual correction is final, even if the model had failed before.
    garment.status = GarmentStatus.DONE
    session.add(garment)
    session.commit()
    session.refresh(garment)
    return garment


def load_embeddings(session, model: str) -> tuple[list[Garment], np.ndarray]:
    """All garments with an embedding from this model, and their vectors as a matrix with one row per garment.

    Row i of the matrix belongs to garments[i].
    """
    # Vectors of different models live in different spaces and must never be compared.
    query = select(Garment).where(Garment.embedding_model == model)
    garments = list(session.exec(query).all())
    if not garments:  # nothing embedded yet
        vectors = np.empty((0, 768), dtype=np.float32)
    else:
        vectors = np.stack([np.frombuffer(garment.embedding, dtype=np.float32) for garment in garments])
    return garments, vectors


def best_matches(garments: list[Garment], vectors: np.ndarray, query: np.ndarray, k: int | None) -> list[Garment]:
    """The k garments whose vectors have the highest cosine similarity to query, most similar first.

    All vectors are normalized, so the dot product is the cosine similarity.
    """
    if not garments:  # the empty matrix has no known width, so there is nothing to multiply
        return []
    similarities = vectors @ query
    order = np.argsort(similarities)[::-1]  # argsort sorts ascending; reversed = most similar first
    return [garments[i] for i in order[:k]]


def similar_garments(session, garment: Garment, k: int = 6) -> list[Garment]:
    """The k garments whose image is most similar to this one, most similar first."""
    if garment.embedding is None:  # not processed yet
        return []
    garments, vectors = load_embeddings(session, garment.embedding_model)
    query = np.frombuffer(garment.embedding, dtype=np.float32)
    # The garment itself is always the most similar one, so ask for one more and drop it.
    matches = best_matches(garments, vectors, query, k + 1)
    return [match for match in matches if match.id != garment.id][:k]


def search_garments(session, embedder, text: str, k: int | None = 6) -> list[Garment]:
    """The k garments whose image fits a text like "blue linen shirt" best, most similar first; k=None: all."""
    garments, vectors = load_embeddings(session, embedder.name)
    query = embedder.embed_texts([text])[0]
    return best_matches(garments, vectors, query, k)
