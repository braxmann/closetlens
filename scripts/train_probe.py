"""Train a linear probe per attribute on the cached features and measure it on the validation set.

Run from the project root: .venv/bin/python -m scripts.train_probe
Needs data/features/<model>.npz from scripts.cache_features.
"""
import argparse

import pandas as pd

from scripts.cache_features import features_path
from scripts.make_test_set import SPLIT_PATH
from scripts.make_validation_set import VALIDATION_SET_PATH
from wardrobe.evaluation.dataset import load_products
from wardrobe.evaluation.split import ATTRIBUTE_MAPS, add_labels
from wardrobe.training.features import load_features
from wardrobe.training.linear_probe import accuracy, rows_for, train_probe

# Zero-shot accuracy of Marqo on the validation set (M5, tuned class texts), the number to beat.
ZERO_SHOT = {"category": 0.96, "main_color": 0.80, "formality": 0.82}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="hf-hub:Marqo/marqo-fashionSigLIP")
    # Several values = grid search. Log scale: 0.01 -> 0.1 can matter as much as 10 -> 100.
    parser.add_argument("--C", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1, 10, 100])
    args = parser.parse_args()

    ids, vectors = load_features(features_path(args.model))
    products = add_labels(load_products()).merge(pd.read_parquet(SPLIT_PATH), on="id")
    train = products[products["split"] == "train"]
    validation_set = pd.read_parquet(VALIDATION_SET_PATH)

    for attribute in ATTRIBUTE_MAPS:
        # Training: every training product with a label for this attribute (not balanced: real class sizes).
        labeled = train.dropna(subset=[attribute])
        X_train, y_train = rows_for(ids, vectors, labeled["id"].to_numpy()), labeled[attribute].to_numpy()
        # Validation: the same balanced sample the zero-shot texts were tuned on.
        sample = validation_set[validation_set["attribute"] == attribute]
        X_val, y_val = rows_for(ids, vectors, sample["id"].to_numpy()), sample["label"].to_numpy()

        print(f"\n{attribute}: {len(y_train)} training products, zero-shot {ZERO_SHOT[attribute]:.2f} on validation")
        # The data stay the same, only the training is repeated for every C.
        for C in args.C:
            model = train_probe(X_train, y_train, C=C)
            print(f"  C={C:<7} train {accuracy(model, X_train, y_train):.3f}  validation {accuracy(model, X_val, y_val):.3f}")


if __name__ == "__main__":
    main()
