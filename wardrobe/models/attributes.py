from typing import Literal

from pydantic import BaseModel

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


class GarmentAttributes(BaseModel):
    category: Category
    main_color: Color
    pattern: Pattern
    brand:str|None = None
    # None = cannot be judged from the photo. "other" = recognized, but not in the list.
    material: Material | None = None
    formality: Formality | None = None
