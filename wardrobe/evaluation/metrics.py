"""Compare a recognizer's answers (scripts/run_recognizer.py) with the frozen test set.

A model may answer null for attributes it cannot judge (material, formality). Null answers are not counted
as errors: coverage says how often the model answered, accuracy and macro-F1 are measured on the answered
cases only (selective classification).
"""
import json
import math
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

NULL = "null"  # how a missing answer appears in tables


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % confidence interval of a proportion; better than the normal approximation near 0 and 1."""
    if n == 0:
        return math.nan, math.nan
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - half_width, center + half_width


def load_predictions(path: Path) -> pd.DataFrame:
    """One row per image: id, seconds, error and one column per predicted attribute."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows = [{"id": r["id"], "seconds": r["seconds"], "error": r.get("error"), **r.get("prediction", {})} for r in records]
    # A rerun appends a newer answer for the same image; the last one counts.
    return pd.DataFrame(rows).drop_duplicates("id", keep="last")


def join(test_set: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """The test set (attribute, id, label) plus the model's answer for that attribute in "prediction".

    Starts from the test set, so answers for images that are no longer in it are ignored.
    """
    joined = test_set.merge(predictions, on="id", how="left")
    missing = joined["seconds"].isna().sum()
    if missing:
        raise ValueError(f"{missing} test rows have no answer yet; finish the run first")
    joined["prediction"] = [row[row["attribute"]] if pd.notna(row[row["attribute"]]) else NULL for _, row in joined.iterrows()]
    return joined[["attribute", "id", "label", "prediction", "seconds", "error"]]


def summary(joined: pd.DataFrame) -> pd.DataFrame:
    """One row per attribute: coverage, accuracy with 95 % interval and macro-F1 on the answered cases."""
    rows = []
    for attribute, group in joined.groupby("attribute", sort=False):
        answered = group[group["prediction"] != NULL]
        correct = int((answered["prediction"] == answered["label"]).sum())
        low, high = wilson_interval(correct, len(answered))
        labels = sorted(group["label"].unique())
        rows.append({
            "attribute": attribute,
            "n": len(group),
            "coverage": len(answered) / len(group),
            "accuracy": correct / len(answered) if len(answered) else math.nan,
            "ci_low": low,
            "ci_high": high,
            # Only over the true classes: answering e.g. "other" lowers the recall of the true class.
            "macro_f1": f1_score(answered["label"], answered["prediction"], labels=labels, average="macro", zero_division=0),
        })
    return pd.DataFrame(rows).set_index("attribute")


def per_class(joined: pd.DataFrame, attribute: str) -> pd.DataFrame:
    """Precision, recall (with 95 % interval) and F1 per true class of one attribute."""
    group = joined[joined["attribute"] == attribute]
    answered = group[group["prediction"] != NULL]
    labels = sorted(group["label"].unique())
    precision, recall, f1, support = precision_recall_fscore_support(
        answered["label"], answered["prediction"], labels=labels, zero_division=0
    )
    table = pd.DataFrame({"n": support, "precision": precision, "recall": recall, "f1": f1}, index=labels)
    intervals = [wilson_interval(round(r * n), n) for r, n in zip(recall, support)]
    table["recall_ci_low"], table["recall_ci_high"] = zip(*intervals)
    table["coverage"] = [(answered["label"] == label).sum() / (group["label"] == label).sum() for label in labels]
    return table


def confusion(joined: pd.DataFrame, attribute: str) -> pd.DataFrame:
    """Rows: true class, columns: answer, including "null" and answers outside the true classes."""
    group = joined[joined["attribute"] == attribute]
    return pd.crosstab(group["label"], group["prediction"], rownames=["true"], colnames=["answer"])


def label_noise(audit: pd.DataFrame) -> pd.DataFrame:
    """Share of wrong and ambiguous labels per attribute from a label audit (scripts/make_audit_sheets.py).

    Mapping errors are left out: they were fixed in label_mapping.py after the audit.
    """
    rows = []
    for attribute in ["category", "main_color", "formality"]:
        labeled = audit[attribute].notna().sum()
        flags = audit["wrong"].fillna("").str.split()
        wrong = flags.map(lambda entries: attribute in entries).sum()
        ambiguous = flags.map(lambda entries: f"{attribute}:ambiguous" in entries).sum()
        rows.append({"attribute": attribute, "audited": labeled, "wrong": wrong / labeled, "ambiguous": ambiguous / labeled})
    return pd.DataFrame(rows).set_index("attribute")


def answer_distribution(predictions: pd.DataFrame, test_set: pd.DataFrame, attribute: str) -> pd.DataFrame:
    """Share of each answer for an attribute without ground truth (material, pattern), per true category."""
    categories = test_set[test_set["attribute"] == "category"][["id", "label"]]
    table = categories.merge(predictions, on="id")
    return pd.crosstab(table["label"], table[attribute].fillna(NULL), normalize="index", rownames=["category"])
