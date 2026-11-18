"""Evaluate one recognizer run against the frozen test set and write data/evaluation/<run>_report.md.

Run from the project root: .venv/bin/python -m scripts.evaluate --run qwen2.5vl-3b
With --set validation it reads <run>_validation.jsonl and writes <run>_validation_report.md.
"""
import argparse

import pandas as pd

from scripts.make_audit_sheets import AUDIT_DIR
from scripts.run_recognizer import SET_PATHS, result_path
from wardrobe.evaluation.metrics import (
    answer_distribution, confusion, join, label_noise, load_predictions, per_class, summary,
)

AUDIT_RESULT_PATH = AUDIT_DIR / "audit_claude.csv"


def block(title: str, table: pd.DataFrame) -> str:
    return f"## {title}\n\n```\n{table.to_string(float_format=lambda value: f'{value:.2f}')}\n```\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="qwen2.5vl-3b", help="name of the result file in data/evaluation")
    parser.add_argument("--set", choices=SET_PATHS, default="test")
    args = parser.parse_args()

    # Named test_set below, but it is the validation set with --set validation.
    test_set = pd.read_parquet(SET_PATHS[args.set])
    results = result_path(args.run, args.set)
    predictions = load_predictions(results)
    joined = join(test_set, predictions)
    tested = predictions[predictions["id"].isin(test_set["id"])]

    sections = [f"# Evaluation: {args.run} ({args.set} set)\n"]
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
    output = results.with_name(f"{results.stem}_report.md")
    output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Written to {output}")


if __name__ == "__main__":
    main()
