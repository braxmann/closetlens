"""Learning curve: validation accuracy of the linear probe with only n training examples per class.

Answers: how many labeled photos per class does a probe need to beat zero-shot?
Run from the project root: .venv/bin/python -m scripts.learning_curve
Writes data/evaluation/learning_curve.png.
"""
import matplotlib.pyplot as plt
import pandas as pd

from scripts.cache_features import features_path
from scripts.make_test_set import SPLIT_PATH
from scripts.make_validation_set import VALIDATION_SET_PATH
from scripts.run_recognizer import RESULTS_DIR
from scripts.train_probe import ZERO_SHOT
from wardrobe.evaluation.dataset import load_products
from wardrobe.evaluation.split import ATTRIBUTE_MAPS, add_labels
from wardrobe.training.features import load_features
from wardrobe.training.linear_probe import accuracy, rows_for, sample_per_class, train_probe

SIZES = [5, 10, 20, 50, 100, 200, 500]
# Small samples depend a lot on which examples were drawn, so every size is repeated with other seeds.
SEEDS = [0, 1, 2]
MODEL = "hf-hub:Marqo/marqo-fashionSigLIP"


def main() -> None:
    ids, vectors = load_features(features_path(MODEL))
    products = add_labels(load_products()).merge(pd.read_parquet(SPLIT_PATH), on="id")
    train = products[products["split"] == "train"]
    validation_set = pd.read_parquet(VALIDATION_SET_PATH)

    results = []  # one dict per (attribute, n, seed)
    for attribute in ATTRIBUTE_MAPS:
        labeled = train.dropna(subset=[attribute])
        sample = validation_set[validation_set["attribute"] == attribute]
        X_val, y_val = rows_for(ids, vectors, sample["id"].to_numpy()), sample["label"].to_numpy()

        for n in SIZES:
            for seed in SEEDS:
                tset = sample_per_class(labeled, attribute, n, seed)
                X_train, y_train = rows_for(ids, vectors, tset["id"].to_numpy()), tset[attribute].to_numpy()
                model = train_probe(X_train, y_train, C=1)
                results.append({"attribute": attribute, "n": n, "seed": seed, "validation": accuracy(model, X_val, y_val)})
    table = pd.DataFrame(results)
    # Mean and spread over the seeds, one row per attribute and size.
    summary = table.groupby(["attribute", "n"])["validation"].agg(["mean", "min", "max"])
    print(summary.round(3).to_string())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for attribute in ATTRIBUTE_MAPS:
        curve = summary.loc[attribute]
        line = ax.plot(curve.index, curve["mean"], marker="o", label=attribute)[0]
        # The band shows how much the result depends on which examples were drawn.
        ax.fill_between(curve.index, curve["min"], curve["max"], color=line.get_color(), alpha=0.2)
        # Zero-shot in the same color, dashed: where the curve crosses it, the probe starts to pay off.
        ax.axhline(ZERO_SHOT[attribute], color=line.get_color(), linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xticks(SIZES, [str(n) for n in SIZES])
    ax.set_xlabel("training examples per class")
    ax.set_ylabel("validation accuracy")
    ax.set_title("Linear probe on Marqo features (dashed: zero-shot)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = RESULTS_DIR / "learning_curve.png"
    fig.savefig(path, dpi=150)
    print(f"Written to {path}")


if __name__ == "__main__":
    main()
