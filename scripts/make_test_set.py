"""Freeze the test set: write split.parquet and test_set.parquet next to products.parquet.

The test set is created once and then never changed, so that all models (VLM variants in M4, own models
in M6 and M7) are measured on exactly the same products. An existing test set is only replaced with --force.

Run from the project root: .venv/bin/python -m scripts.make_test_set
"""
import argparse

from wardrobe.evaluation.dataset import PRODUCTS_PATH, load_products
from wardrobe.evaluation.split import build_test_set

SPLIT_PATH = PRODUCTS_PATH.with_name("split.parquet")
TEST_SET_PATH = PRODUCTS_PATH.with_name("test_set.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-per-class", type=int, default=100)
    parser.add_argument("--max-test-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="replace an existing test set")
    args = parser.parse_args()

    if TEST_SET_PATH.exists() and not args.force:
        raise SystemExit(f"{TEST_SET_PATH} already exists. The test set is frozen; use --force to replace it.")

    split, test_set = build_test_set(load_products(), args.n_per_class, args.max_test_fraction, args.seed)
    split.to_parquet(SPLIT_PATH, index=False)
    test_set.to_parquet(TEST_SET_PATH, index=False)

    groups = split.drop_duplicates("group")["split"].value_counts()
    print(f"Groups: {groups.get('test', 0)} test, {groups.get('train', 0)} train")
    for attribute, sample in test_set.groupby("attribute", sort=False):
        print(f"\n{attribute}: {len(sample)} products")
        print(sample["label"].value_counts().to_string())
    images = test_set["id"].nunique()
    print(f"\n{images} distinct images to send to the VLM (~{images * 1.7 / 60:.0f} min at 1.7 s each)")


if __name__ == "__main__":
    main()
