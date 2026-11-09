from contextlib import asynccontextmanager
from threading import Event, Thread

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session, select

from wardrobe.api.dependencies import engine, get_session, get_storage, storage
from wardrobe.jobs.worker import reset_stuck_garments, run_worker
from wardrobe.models.attribute_recognizer import OllamaAttributeRecognizer
from wardrobe.models.attributes import GarmentAttributes
from wardrobe.models.background_remover import RembgBackgroundRemover
from wardrobe.services.garments import correct_garment, create_garment
from wardrobe.storage.database import Garment
from wardrobe.storage.files import FileStorage
from wardrobe.web.pages import STATIC_DIR
from wardrobe.web.pages import router as pages_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # get_session is meant for FastAPI's Depends; here we open a session ourselves.
    with Session(engine) as session:
        reset_stuck_garments(session)

    # Create objects (with brackets), not just name the classes. Loaded once for the whole server.
    remover = RembgBackgroundRemover("birefnet-general")
    recognizer = OllamaAttributeRecognizer("qwen2.5vl:3b")

    stop = Event()
    thread = Thread(target=run_worker, args=(engine, storage, remover, recognizer, stop), daemon=True)
    thread.start()

    yield  # the server runs here

    stop.set()
    thread.join()


app = FastAPI(lifespan=lifespan)

# Make folders reachable by URL: /static/... for CSS and JS, /cutouts/... for the cut-out photos.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# check_dir=False: on a fresh install the folder only appears with the first processed photo.
app.mount("/cutouts", StaticFiles(directory=storage.base_dir / "cutouts", check_dir=False), name="cutouts")
app.mount("/thumbnails", StaticFiles(directory=storage.base_dir / "thumbnails", check_dir=False), name="thumbnails")
app.include_router(pages_router)


@app.get("/garments")
def list_garments(session: Session = Depends(get_session)) -> list[Garment]:
    return session.exec(select(Garment)).all()


@app.get("/garments/{garment_id}")
def get_garment(garment_id: int, session: Session = Depends(get_session)) -> Garment:
    garment = session.get(Garment, garment_id)
    if garment is None:
        raise HTTPException(status_code=404, detail="Garment not found")
    return garment

@app.post("/garments")
def upload_garment(
    photo: UploadFile,
    session: Session = Depends(get_session),
    storage: FileStorage = Depends(get_storage),
) -> Garment:
    return create_garment(session, storage, photo.file.read())

@app.patch("/garments/{garment_id}")
def patch_garment(
    garment_id: int,
    correction: GarmentAttributes,
    session: Session = Depends(get_session),
) -> Garment:
    garment = session.get(Garment, garment_id)
    if garment is None:
        raise HTTPException(status_code=404, detail="Garment not found")
    return correct_garment(session, garment, correction)
