import pytest

from wardrobe.models.attributes import GarmentAttributes


@pytest.mark.parametrize("brand", ["null", "None", " unknown ", ""])
def test_empty_brand_becomes_none(brand):
    attributes = GarmentAttributes(category="pants", main_color="blue", pattern="solid", brand=brand)
    assert attributes.brand is None


def test_real_brand_is_kept():
    attributes = GarmentAttributes(category="t-shirt", main_color="white", pattern="graphic", brand=" PLAY ")
    assert attributes.brand == "PLAY"
