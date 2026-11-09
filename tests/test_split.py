import pandas as pd

from wardrobe.evaluation.split import add_groups, build_test_set


def make_products(n_designs: int, colours: list[str], article_type: str = "Tshirts", start_id: int = 0):
    """Every design exists in every colour, so each design is one group of len(colours) products."""
    rows = []
    for design in range(n_designs):
        for colour in colours:
            rows.append({
                "id": start_id + len(rows),
                "productDisplayName": f"Brand Men {colour} Design{design} {article_type}",
                "articleType": article_type,
                "baseColour": colour,
                "usage": "Casual",
            })
    return pd.DataFrame(rows)


def test_colour_variants_form_one_group():
    products = add_groups(make_products(1, ["Navy Blue", "Red"]))
    assert products["group"].nunique() == 1


def test_groups_are_never_split():
    products = make_products(40, ["Black", "White", "Red"])
    split, _ = build_test_set(products, n_per_class=10)
    assert (split.groupby("group")["split"].nunique() == 1).all()


def test_at_most_one_product_per_group_and_label():
    products = make_products(40, ["Black", "White", "Red"])
    split, test_set = build_test_set(products, n_per_class=10)
    test_set = test_set.merge(split, on="id")
    assert (test_set.groupby(["attribute", "label", "group"]).size() == 1).all()
    assert (test_set["split"] == "test").all()


def test_every_label_gets_test_groups():
    products = pd.concat([
        make_products(100, ["Black"], "Tshirts"),
        make_products(30, ["Red"], "Tshirts", start_id=1000),
    ])
    _, test_set = build_test_set(products, n_per_class=10)
    colours = test_set[test_set["attribute"] == "main_color"]["label"].value_counts()
    assert colours["red"] == 10


def test_small_category_keeps_groups_for_training():
    products = pd.concat([
        make_products(200, ["Black"], "Tshirts"),
        make_products(20, ["Black"], "Skirts", start_id=1000),
    ])
    split, test_set = build_test_set(products, n_per_class=100, max_test_fraction=0.5)
    counts = test_set[test_set["attribute"] == "category"]["label"].value_counts()
    assert counts["t-shirt"] == 100
    assert counts["skirt"] == 10


def test_same_seed_same_test_set():
    products = make_products(40, ["Black", "White", "Red"])
    _, first = build_test_set(products, n_per_class=10, seed=1)
    _, second = build_test_set(products, n_per_class=10, seed=1)
    pd.testing.assert_frame_equal(first, second)
