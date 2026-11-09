"""Evaluate one recognizer run against the frozen test set and write data/evaluation/<run>_report.md.

Run from the project root: .venv/bin/python -m scripts.evaluate --run qwen2.5vl-3b
"""
import argparse

import pandas as pd

from scripts.make_audit_sheets import AUDIT_DIR
from scripts.make_test_set import TEST_SET_PATH
from scripts.run_recognizer import RESULTS_DIR
from wardrobe.evaluation.metrics import (
    answer_distribution, confusion, join, label_noise, load_predictions, per_class, summary,
)

AUDIT_RESULT_PATH = AUDIT_DIR / "audit_claude.csv"


def block(title: str, table: pd.DataFrame) -> str:
    return f"## {title}\n\n```\n{table.to_string(float_format=lambda value: f'{value:.2f}')}\n```\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="qwen2.5vl-3b", help="name of the result file in data/evaluation")
    args = parser.parse_args()

    test_set = pd.read_parquet(TEST_SET_PATH)
    predictions = load_predictions(RESULTS_DIR / f"{args.run}.jsonl")
    joined = join(test_set, predictions)
    tested = predictions[predictions["id"].isin(test_set["id"])]

    sections = [f"# Evaluation: {args.run}\n"]
    sections.append(block("Summary (accuracy and macro-F1 on answered cases)", summary(joined)))
    if AUDIT_RESULT_PATH.exists():
        sections.append(block("Label noise from the audit", label_noise(pd.read_csv(AUDIT_RESULT_PATH))))
    for attribute in test_set["attribute"].unique():
        sections.append(block(f"{attribute}: per class", per_class(joined, attribute)))
        sections.append(block(f"{attribute}: confusion matrix", confusion(joined, attribute)))
    for attribute in ["material", "pattern"]:
        sections.append(block(f"{attribute} (no ground truth): answers per category", answer_distribution(tested, test_set, attribute)))
    seconds = tested["seconds"]
    sections.append(
        f"## Runtime\n\n{len(tested)} images, {tested['error'].notna().sum()} errors, "
        f"median {seconds.median():.2f} s, mean {seconds.mean():.2f} s per image\n"
    )

    report = "\n".join(sections)
    output = RESULTS_DIR / f"{args.run}_report.md"
    output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Written to {output}")


if __name__ == "__main__":
    main()
