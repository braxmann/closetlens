"""Draw the validation set from the training groups: updates split.parquet and writes validation_set.parquet.

The validation set is for tuning prompts and class texts. The test set stays frozen and is only looked at
for the final numbers. An existing validation set is only replaced with --force.

Run from the project root: .venv/bin/python -m scripts.make_validation_set
"""
import argparse

import pandas as pd

from scripts.make_test_set import SPLIT_PATH, TEST_SET_PATH
from wardrobe.evaluation.dataset import load_products
from wardrobe.evaluation.split import build_validation_set

VALIDATION_SET_PATH = TEST_SET_PATH.with_name("validation_set.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-per-class", type=int, default=50)
    parser.add_argument("--max-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="replace an existing validation set")
    args = parser.parse_args()

    if VALIDATION_SET_PATH.exists() and not args.force:
        raise SystemExit(f"{VALIDATION_SET_PATH} already exists; use --force to replace it.")

    old_split = pd.read_parquet(SPLIT_PATH)
    # A rerun starts again from the test/train split.
    old_split["split"] = old_split["split"].replace("validation", "train")
    split, validation_set = build_validation_set(
        load_products(), old_split, args.n_per_class, args.max_fraction, args.seed
    )
    unchanged = split.merge(old_split, on="id", suffixes=("", "_old"))
    if not ((unchanged["split"] == "test") == (unchanged["split_old"] == "test")).all():
        raise SystemExit("The test split would change; aborting.")
    split.to_parquet(SPLIT_PATH, index=False)
    validation_set.to_parquet(VALIDATION_SET_PATH, index=False)

    groups = split.drop_duplicates("group")["split"].value_counts()
    print(f"Groups: {groups.get('test', 0)} test, {groups.get('validation', 0)} validation, {groups.get('train', 0)} train")
    for attribute, sample in validation_set.groupby("attribute", sort=False):
        print(f"\n{attribute}: {len(sample)} products")
        print(sample["label"].value_counts().to_string())
    print(f"\n{validation_set['id'].nunique()} distinct images")


if __name__ == "__main__":
    main()
