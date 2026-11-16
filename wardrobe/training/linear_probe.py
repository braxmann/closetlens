"""Linear probe (M6): a logistic regression on top of the frozen embeddings, one per attribute.

The embedder stays unchanged; only one weight per class and embedding dimension is learned from the labeled
training products.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

PROBES_DIR = Path("data/models")


def probes_path(model: str) -> Path:
    """data/models/probes_<model>.joblib: the probes only fit the embedder whose features they were trained on."""
    return PROBES_DIR / f"probes_{model.removeprefix('hf-hub:').replace('/', '_')}.joblib"


def save_probes(path: Path, model: str, probes: dict[str, LogisticRegression]) -> None:
    """Stores the trained probes together with the name of the embedder they belong to.

    joblib is pickle for objects with large numpy arrays inside; scikit-learn models are saved this way.
    Pickle files can run code when loaded, so only load files you wrote yourself.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "probes": probes}, path)


def load_probes(path: Path) -> tuple[str, dict[str, LogisticRegression]]:
    """(embedder name, attribute -> probe) as written by save_probes."""
    data = joblib.load(path)
    return data["model"], data["probes"]


def rows_for(ids: np.ndarray, vectors: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    """The feature rows of the wanted product ids, in the order of wanted."""
    position = {int(id): row for row, id in enumerate(ids)}
    return vectors[[position[int(id)] for id in wanted]]


def train_probe(X: np.ndarray, y: np.ndarray, C: float = 0.25) -> LogisticRegression:
    return LogisticRegression(C=C, max_iter=1000,class_weight="balanced").fit(X,y)

def accuracy(model: LogisticRegression, X: np.ndarray, y: np.ndarray) -> float:
    """Share of rows whose predicted label equals the true label."""
    predictions = model.predict(X)
    return (predictions == y).mean()



def sample_per_class(labeled: pd.DataFrame, attribute: str, n: int, seed: int) -> pd.DataFrame:
    """Up to n random rows per label of attribute. Labels with fewer rows keep all of them.

    Used for the learning curve: how good is the probe with only n training examples per class?
    """
    shuffled = labeled.sample(frac=1, random_state=seed)  # sample returns a new table, labeled stays unchanged
    return shuffled.groupby(attribute).head(n)
