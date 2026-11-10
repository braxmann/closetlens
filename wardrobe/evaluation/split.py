"""Split the dataset into test, validation and training groups and draw balanced samples.

Products that differ only in color (the same shirt in red and in blue) form a group. A group goes either
into the test split or into the training split, never both, so a model trained in M6 never sees a sibling
of a test product. Within one label, a test sample contains at most one product per group: siblings are
not independent examples and would make the confidence intervals look narrower than they are.
"""
import re

import numpy as np
import pandas as pd

from wardrobe.evaluation.label_mapping import CATEGORY_MAP, COLOR_MAP, FORMALITY_MAP

# Our attribute -> (dataset column, mapping)
ATTRIBUTE_MAPS = {
    "category": ("articleType", CATEGORY_MAP),
    "main_color": ("baseColour", COLOR_MAP),
    "formality": ("usage", FORMALITY_MAP),
}
# Color words in product names that are not part of any baseColour value.
EXTRA_COLOR_WORDS = {"gray", "multicolour", "multicoloured", "colour", "coloured"}


def add_labels(products: pd.DataFrame) -> pd.DataFrame:
    """Adds one column per attribute with our label, or NaN if the dataset value is not mapped."""
    products = products.copy()
    for attribute, (column, mapping) in ATTRIBUTE_MAPS.items():
        products[attribute] = products[column].map(mapping)
    return products


def add_groups(products: pd.DataFrame) -> pd.DataFrame:
    """Adds a "group" column: the product name without color words."""
    color_words = {word.lower() for value in products["baseColour"].dropna().unique() for word in value.split()}
    color_words |= EXTRA_COLOR_WORDS

    def group_key(row) -> str:
        if pd.isna(row.productDisplayName):
            return f"id-{row.id}"  # without a name, a product is its own group
        words = re.findall(r"[a-z0-9]+", row.productDisplayName.lower())
        return " ".join(word for word in words if word not in color_words)

    products = products.copy()
    products["group"] = [group_key(row) for row in products.itertuples()]
    return products


def split_groups(
    products: pd.DataFrame, test_groups_per_class: int, max_test_fraction: float, seed: int
) -> pd.Series:
    """"test" or "train" per group, stratified by every label of every attribute.

    For each label, groups containing it are moved to the test split until
    min(test_groups_per_class, max_test_fraction * groups with that label) test groups contain it.
    The fraction keeps enough groups of small labels for training. Groups already moved to the test split
    for an earlier label count towards the later ones, so later labels only add what is still missing.
    """
    rng = np.random.default_rng(seed)
    test_groups: set[str] = set()
    for attribute in ATTRIBUTE_MAPS:
        groups_by_label = products.dropna(subset=[attribute]).groupby(attribute)["group"].unique()
        for label in sorted(groups_by_label.index):
            groups = np.sort(groups_by_label[label])
            target = min(test_groups_per_class, int(len(groups) * max_test_fraction))
            missing = target - len(test_groups.intersection(groups))
            if missing > 0:
                train_groups = [group for group in groups if group not in test_groups]
                test_groups.update(rng.choice(train_groups, missing, replace=False))
    groups = products["group"].unique()
    return pd.Series(np.where(np.isin(groups, list(test_groups)), "test", "train"), index=groups)


def sample_attribute(
    products: pd.DataFrame, attribute: str, n_per_class: int, seed: int, split: str = "test"
) -> pd.DataFrame:
    """Up to n_per_class products of one split per label of one attribute, at most one per group and label.

    The red and the blue variant of one shirt may both be in the color sample, but not two red ones.
    """
    rng = np.random.default_rng(seed)
    candidates = products[(products["split"] == split) & products[attribute].notna()]
    one_per_group = candidates.groupby([attribute, "group"]).sample(1, random_state=rng)
    return pd.concat(
        pool.sample(min(n_per_class, len(pool)), random_state=rng) for _, pool in one_per_group.groupby(attribute)
    )


def build_test_set(
    products: pd.DataFrame, n_per_class: int = 100, max_test_fraction: float = 0.5, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (split, test_set).

    split: id, group, split ("test" or "train") for every product.
    test_set: attribute, id, label, one balanced sample per attribute (long format).
    """
    products = add_groups(add_labels(products))
    group_split = split_groups(products, n_per_class, max_test_fraction, seed)
    products["split"] = products["group"].map(group_split)

    return products[["id", "group", "split"]], draw_samples(products, n_per_class, seed, "test")


def draw_samples(products: pd.DataFrame, n_per_class: int, seed: int, split: str) -> pd.DataFrame:
    """attribute, id, label: one balanced sample per attribute from one split (long format)."""
    samples = []
    for attribute in ATTRIBUTE_MAPS:
        sample = sample_attribute(products, attribute, n_per_class, seed, split)
        samples.append(pd.DataFrame({"attribute": attribute, "id": sample["id"], "label": sample[attribute]}))
    return pd.concat(samples, ignore_index=True)


def build_validation_set(
    products: pd.DataFrame, split: pd.DataFrame, n_per_class: int = 50, max_fraction: float = 0.25, seed: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (split, validation_set): moves some training groups to "validation" and samples from them.

    Used to tune prompts and class texts, so the test set is looked at only once, for the final numbers.
    Test groups stay untouched. The validation sample is drawn the same way as the test sample, but smaller
    and with a lower fraction, so small labels keep most of their groups for training in M6.
    """
    products = add_labels(products).merge(split, on="id")
    train = products[products["split"] == "train"]
    validation_groups = split_groups(train, n_per_class, max_fraction, seed)
    validation_groups = validation_groups[validation_groups == "test"].index
    products.loc[products["group"].isin(validation_groups), "split"] = "validation"
    return products[["id", "group", "split"]], draw_samples(products, n_per_class, seed, "validation")
