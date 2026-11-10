from threading import Event

from sqlalchemy import Engine
from sqlmodel import Session, select

from wardrobe.models.attribute_recognizer import AttributeRecognizer
from wardrobe.models.background_remover import BackgroundRemover
from wardrobe.models.embedder import Embedder
from wardrobe.services.garments import process_garment
from wardrobe.storage.database import Garment, GarmentStatus
from wardrobe.storage.files import FileStorage

POLL_SECONDS = 1


def next_pending(session: Session) -> Garment | None:
    query = select(Garment).where(Garment.status == GarmentStatus.PENDING).order_by(Garment.created_at)
    return session.exec(query).first()

def reset_stuck_garments(session:Session) -> None:
    query = select(Garment).where(Garment.status == GarmentStatus.PROCESSING)
    garments = session.exec(query).all()
    for garment in garments:
        garment.status = GarmentStatus.PENDING
        session.add(garment)
    session.commit()
 
def run_worker(
    engine: Engine,
    storage: FileStorage,
    remover: BackgroundRemover,
    recognizer: AttributeRecognizer,
    embedder: Embedder,
    stop: Event,
) -> None:
    while not stop.is_set():
        # A fresh session per job: a session belongs to one thread and should not live forever.
        with Session(engine) as session:
            garment = next_pending(session)
            if garment is not None:
                process_garment(session, storage, garment, remover, recognizer, embedder)
                continue  # look for the next job right away
        stop.wait(POLL_SECONDS)  # nothing to do; wakes up early when stop is set
