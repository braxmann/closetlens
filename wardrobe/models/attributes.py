from typing import Literal

from pydantic import BaseModel, field_validator

Category = Literal[
    "t-shirt", "shirt", "sweater", "jacket", "pants", "shorts", "skirt", "dress", "shoes", "other"
]
Color = Literal[
    "black", "white", "gray", "beige", "brown", "blue", "green", "red", "yellow", "pink", "purple",
    "multicolor", "other",
]
Pattern = Literal["solid", "striped", "checked", "dotted", "graphic", "other"]
Material = Literal["cotton", "denim", "wool", "linen", "leather", "synthetic", "other"]
Formality = Literal["casual", "smart", "formal"]

# Words a model writes as text when it means "no brand".
EMPTY_BRANDS = {"", "null", "none", "unknown", "n/a"}


class GarmentAttributes(BaseModel):
    category: Category
    main_color: Color
    pattern: Pattern
    brand: str | None = None
    # None = cannot be judged from the photo. "other" = recognized, but not in the list.
    material: Material | None = None
    formality: Formality | None = None

    @field_validator("brand")
    @classmethod
    def empty_brand_to_none(cls, value: str | None) -> str | None:
        if value is None or value.strip().lower() in EMPTY_BRANDS:
            return None
        return value.strip()
