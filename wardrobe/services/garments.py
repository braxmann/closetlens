import logging

from wardrobe.storage.database import Garment, GarmentStatus
from PIL import Image

logger = logging.getLogger(__name__)


def create_garment(session, storage, data) -> Garment:
    path = storage.save_photo(data)
    garment = Garment(photo_path=str(path), status="pending")
    session.add(garment)
    session.commit()
    session.refresh(garment)
    return garment

def process_garment(session, storage, garment, remover, recognizer) -> None:
    garment.status = GarmentStatus.PROCESSING
    session.add(garment)
    session.commit()
    garment_id = garment.id
    try:
        image = Image.open(garment.photo_path)
        cutout = remover.rem_background(image)
        attributes = recognizer.recognize_attributes(cutout)
        path = storage.save_cutout(cutout)
        # Reload before changing anything: a correction may have arrived while the models ran.
        session.refresh(garment)
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
