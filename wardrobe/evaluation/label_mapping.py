"""Mapping from the dataset's labels to our own taxonomy (wardrobe/models/attributes.py).

Rule: a dataset value that does not clearly belong to one of our classes is left out of the mapping,
and products with that value are not used for the evaluation. Mapping it to "other" would make
"other" a catch-all class and blur the ground truth.

Numbers in the comments are product counts in products.parquet.
"""
from wardrobe.models.attributes import Category, Color, Formality

# articleType -> category
CATEGORY_MAP: dict[str, Category] = {
    "Tshirts": "t-shirt",  # 6815
    "Shirts": "shirt",  # 2911
    "Sweaters": "sweater",  # 277
    "Sweatshirts": "sweater",  # 282
    "Jackets": "jacket",  # 254
    "Rain Jacket": "jacket",  # 7
    "Blazers": "jacket",  # 8
    "Waistcoat": "jacket",  # 15
    "Nehru Jackets": "jacket",  # 5
    "Shrug": "jacket",  # 6
    # Denim is a material in our taxonomy, not a category, so jeans are pants.
    "Jeans": "pants",  # 533
    "Trousers": "pants",  # 472
    "Track Pants": "pants",  # 298
    "Capris": "pants",  # 149
    "Leggings": "pants",  # 162
    "Jeggings": "pants",  # 33
    "Lounge Pants": "pants",  # 61
    "Shorts": "shorts",  # 496
    "Lounge Shorts": "shorts",  # 33
    "Skirts": "skirt",  # 114
    "Dresses": "dress",  # 437
    "Nightdress": "dress",  # 187
    # "shoes" includes open footwear.
    "Casual Shoes": "shoes",  # 2830
    "Sports Shoes": "shoes",  # 2013
    "Formal Shoes": "shoes",  # 634
    "Heels": "shoes",  # 1318
    "Flats": "shoes",  # 498
    "Sandals": "shoes",  # 890
    "Sports Sandals": "shoes",  # 67
    "Flip Flops": "shoes",  # 902
}
# Excluded, because the label itself is ambiguous in our taxonomy:
# - "Tops" (1657): a mix of t-shirt-like and blouse-like garments.
# - "Tunics" (229), "Kurtis" (234), "Kurtas" (1844): long tops between shirt and dress.
# Excluded after the label audit (data/evaluation/audit):
# - "Jumpsuit" (16): has trouser or shorts legs, so it is no dress.
# - "Rompers" (12): baby bodysuits in this dataset.
# Excluded, because no class of ours fits: innerwear, socks, nightwear sets, sarees, salwar/churidar/patiala,
# dupatta, lehenga, kurta sets, clothing sets, swimwear, stockings, tights, tracksuits, belts, suspenders.

# baseColour -> main_color
COLOR_MAP: dict[str, Color] = {
    "Black": "black",  # 5956
    "White": "white",  # 4160
    "Off White": "white",  # 150
    "Grey": "gray",  # 2109
    "Grey Melange": "gray",  # 138
    "Charcoal": "gray",  # 188
    "Beige": "beige",  # 506
    "Khaki": "beige",  # 117
    "Skin": "beige",  # 37
    "Nude": "beige",  # 16
    "Brown": "brown",  # 1868
    "Coffee Brown": "brown",  # 13
    "Mushroom Brown": "brown",  # 12
    "Blue": "blue",  # 3765
    "Navy Blue": "blue",  # 1455
    "Green": "green",  # 1599
    "Olive": "green",  # 272
    "Sea Green": "green",  # 17
    "Lime Green": "green",  # 4
    "Fluorescent Green": "green",  # 3
    "Red": "red",  # 1733
    "Maroon": "red",  # 379
    "Burgundy": "red",  # 25
    "Yellow": "yellow",  # 615
    "Mustard": "yellow",  # 70
    "Pink": "pink",  # 1195
    "Rose": "pink",  # 8
    "Magenta": "pink",  # 96
    "Purple": "purple",  # 1068
    "Lavender": "purple",  # 106
    "Mauve": "purple",  # 6
    "Multi": "multicolor",  # 308
}
# Excluded, because two of our classes are equally plausible:
# Tan (83) and Taupe (9) beige/brown, Teal (76) and Turquoise Blue (50) blue/green,
# Rust (52) red/brown, Peach (128) pink/orange.
# Excluded, because no class of ours fits: Orange (355); Silver, Gold, Bronze, Copper, Metallic.
# Excluded after the label audit: Cream (279), mostly looks white, sometimes beige.

# usage -> formality
FORMALITY_MAP: dict[str, Formality] = {
    "Casual": "casual",  # 21137
    "Sports": "casual",  # 3586
    "Formal": "formal",  # 1685
}
# Excluded: "Smart Casual" (39, too few for a reliable number, so "smart" is not measured),
# "Party" (24, smart or formal), "Ethnic" (2978, a style, not a level of formality), "Travel" (1).
