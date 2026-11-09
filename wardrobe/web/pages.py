"""HTML pages for the browser. The JSON API stays in api/main.py.

The pages use htmx: HTML attributes like hx-post or hx-get send a request and swap the returned
HTML fragment into the page, so the server stays the only place with logic and we need almost no JavaScript.
"""
from pathlib import Path
from typing import Annotated, get_args

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from sqlmodel import Session, func, select

from wardrobe.api.dependencies import get_session, get_storage
from wardrobe.models.attributes import Category, Color, Formality, GarmentAttributes, Material, Pattern
from wardrobe.services.garments import correct_garment, create_garment
from wardrobe.storage.database import Garment
from wardrobe.storage.files import FileStorage

# Paths relative to this file, so the server finds them no matter where it is started from.
WEB_DIR = Path(__file__).parent
STATIC_DIR = WEB_DIR / "static"

# The allowed values come straight from the schema, so the form can never offer a value the schema rejects.
OPTIONS = {
    "category": get_args(Category),
    "main_color": get_args(Color),
    "pattern": get_args(Pattern),
    "material": get_args(Material),
    "formality": get_args(Formality),
}

# Code and database stay English; only the screen shows German words.
LABELS = {
    "t-shirt": "T-Shirt", "shirt": "Hemd", "sweater": "Pullover", "jacket": "Jacke", "pants": "Hose",
    "shorts": "Shorts", "skirt": "Rock", "dress": "Kleid", "shoes": "Schuhe",
    "black": "Schwarz", "white": "Weiß", "gray": "Grau", "beige": "Beige", "brown": "Braun", "blue": "Blau",
    "green": "Grün", "red": "Rot", "yellow": "Gelb", "pink": "Rosa", "purple": "Lila", "multicolor": "Bunt",
    "solid": "Einfarbig", "striped": "Gestreift", "checked": "Kariert", "dotted": "Gepunktet", "graphic": "Print",
    "cotton": "Baumwolle", "denim": "Denim", "wool": "Wolle", "linen": "Leinen", "leather": "Leder",
    "synthetic": "Synthetik", "casual": "Lässig", "smart": "Schick", "formal": "Formell", "other": "Sonstiges",
    "pending": "Wartet", "processing": "In Arbeit", "done": "Fertig", "failed": "Fehler",
}

templates = Jinja2Templates(directory=WEB_DIR / "templates")
# Templates may only use the file name of a stored path: "data/cutouts/ab12.png" -> "ab12.png".
# Path().name also handles Windows paths with backslashes, which matters on the server PC.
templates.env.filters["filename"] = lambda path: Path(path).name
templates.env.filters["label"] = lambda value: LABELS.get(value, value)

# A router collects routes; main.py adds them to the app with include_router.
router = APIRouter()


def find_garment(session: Session, garment_id: int) -> Garment:
    garment = session.get(Garment, garment_id)
    if garment is None:
        raise HTTPException(status_code=404, detail="Garment not found")
    return garment


@router.get("/")
def gallery(
    request: Request,
    session: Session = Depends(get_session),
    category: str | None = None,
    main_color: str | None = None,
):
    # Newest first, so a fresh upload appears at the top. Empty filter values ("") mean "all".
    query = select(Garment).order_by(Garment.created_at.desc())
    if category:
        query = query.where(Garment.category == category)
    if main_color:
        query = query.where(Garment.main_color == main_color)
    garments = session.exec(query).all()
    # The dict holds the variables the template can use.
    return templates.TemplateResponse(
        request,
        "gallery.html",
        {"garments": garments, "options": OPTIONS, "category": category, "main_color": main_color},
    )


@router.post("/upload")
def upload(
    request: Request,
    photo: UploadFile,
    session: Session = Depends(get_session),
    storage: FileStorage = Depends(get_storage),
):
    garment = create_garment(session, storage, photo.file.read())
    total = session.exec(select(func.count()).select_from(Garment)).one()
    # Only the new icon goes back; htmx puts it at the start of the icon grid.
    return templates.TemplateResponse(request, "upload_result.html", {"garment": garment, "total": total})


@router.get("/items/{garment_id}/icon")
def icon(request: Request, garment_id: int, session: Session = Depends(get_session)):
    # Polled by icons that are still being processed, until they are done or failed.
    garment = find_garment(session, garment_id)
    return templates.TemplateResponse(request, "_icon.html", {"garment": garment})


@router.get("/items/{garment_id}")
def edit_form(request: Request, garment_id: int, session: Session = Depends(get_session)):
    garment = find_garment(session, garment_id)
    return templates.TemplateResponse(request, "edit.html", {"garment": garment, "options": OPTIONS})


@router.post("/items/{garment_id}")
def edit_save(
    garment_id: int,
    category: Annotated[str, Form()],
    main_color: Annotated[str, Form()],
    pattern: Annotated[str, Form()],
    brand: Annotated[str, Form()] = "",
    material: Annotated[str, Form()] = "",
    formality: Annotated[str, Form()] = "",
    session: Session = Depends(get_session),
):
    garment = find_garment(session, garment_id)
    try:
        # An empty select option means "unknown", which is None in the schema.
        correction = GarmentAttributes(
            category=category,
            main_color=main_color,
            pattern=pattern,
            brand=brand,
            material=material or None,
            formality=formality or None,
        )
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=error.errors()) from error
    correct_garment(session, garment, correction)
    # 303 tells the browser to load the gallery with GET, so a reload does not send the form again.
    return RedirectResponse("/", status_code=303)
