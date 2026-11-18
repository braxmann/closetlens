"""Size and speed measurements for comparing model variants (M7)."""
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


def model_size_mb(path: Path) -> float:
    return path.stat().st_size / 1e6  # MB = 10^6 bytes, as Hugging Face counts


def measure_latency(
    predict: Callable[[Any], Any], images: list, warmup: int = 5, repeats: int = 50
) -> dict[str, float]:
    """Time predict on single, already preprocessed images; returns milliseconds.

    On a GPU, predict itself must wait for the result (e.g. return logits.cpu()),
    otherwise only the launch of the computation is timed.
    """
    # The first calls are slow (memory allocation, loading kernels) and would distort the result.
    for _ in range(warmup):
        predict(images[0])

    timings = []
    for i in range(repeats):
        image = images[i % len(images)]  # cycle through the images, so repeats may exceed len(images)
        start = time.perf_counter()
        predict(image)
        timings.append((time.perf_counter() - start) * 1000)

    # Median and 90th percentile instead of the mean: a single slow outlier would pull the mean up.
    return {
        "median_ms": statistics.median(timings),
        "p90_ms": float(np.percentile(timings, 90)),
    }
