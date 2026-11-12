"""Linear probe (M6): a logistic regression on top of the frozen embeddings, one per attribute.

The embedder stays unchanged; only one weight per class and embedding dimension is learned from the labeled
training products.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression



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
