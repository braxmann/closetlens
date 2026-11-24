"""Measure size, latency and accuracy of model variants for M7 (Verkleinern).

Run from the project root: .venv/bin/python -m scripts.benchmark
Writes data/evaluation/m7_benchmark.csv (one row per variant and device).
"""
import copy
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch

from scripts.make_validation_set import VALIDATION_SET_PATH
from scripts.run_recognizer import RESULTS_DIR
from wardrobe.compression.export import export_onnx, onnx_predict, quantize_int8
from wardrobe.evaluation.benchmark import measure_latency, model_size_mb
from wardrobe.evaluation.dataset import load_images
from wardrobe.models.finetuned import flatten_on_white, load_finetuned
from wardrobe.training.finetune import MultiHeadClassifier, eval_transform

# The model the app uses (M6): Marqo with the last 2 blocks fine-tuned, one head per attribute.
CHECKPOINT = Path("data/models/finetune_marqo-last2-ls.pt")
MODELS_DIR = Path("data/models/m7")
GPU = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None
DEVICES = ["cpu", GPU] if GPU else ["cpu"]
DTYPES = {"fp32": torch.float32, "fp16": torch.float16}
BATCH_SIZE = 64


def predict_classes(
    predict: Callable[[torch.Tensor], dict[str, torch.Tensor] | dict[str, np.ndarray]], inputs: torch.Tensor
) -> dict[str, np.ndarray]:
    """Class numbers per attribute for all inputs, in batches (one image at a time would take minutes).

    Takes any predict function (PyTorch or ONNX), so every variant is scored by the same code.
    """
    predictions: dict[str, list[np.ndarray]] = {}
    for start in range(0, len(inputs), BATCH_SIZE):
        for attribute, logits in predict(inputs[start:start + BATCH_SIZE]).items():
            predictions.setdefault(attribute, []).append(np.asarray(logits).argmax(axis=1))
    return {attribute: np.concatenate(parts) for attribute, parts in predictions.items()}


def make_predict(
    model: MultiHeadClassifier, device: str, dtype: torch.dtype
) -> Callable[[torch.Tensor], dict[str, torch.Tensor]]:
    """predict for measure_latency: one preprocessed batch in, the logits per attribute out."""
    def predict(x: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.inference_mode():
            # The input needs the model's dtype, otherwise the matrix multiplication fails.
            # .cpu() waits for the result; on mps/cuda the call would otherwise return before the work is done.
            # .float(): numpy has no bfloat16 and argmax does not care, so fp16 logits become fp32 here.
            return {attribute: logits.float().cpu() for attribute, logits in model(x.to(device, dtype)).items()}
    return predict


def accuracy(
    predictions: dict[str, np.ndarray], classes: dict[str, list[str]], ids: list[int], validation: pd.DataFrame
) -> dict[str, float]:
    """Share of validation rows whose label matches the predicted class name, per attribute."""
    row_of = {id: row for row, id in enumerate(ids)}
    result = {}
    for key, rows in validation.groupby("attribute"):
        attribute = str(key)
        predicted = [classes[attribute][int(predictions[attribute][row_of[id]])] for id in rows["id"]]
        result[attribute] = float((pd.Series(predicted, index=rows.index) == rows["label"]).mean())
    return result


def measure(
    predict: Callable, inputs: torch.Tensor, score: bool, classes: dict[str, list[str]], ids: list[int],
    validation: pd.DataFrame,
) -> dict[str, float]:
    """Latency on single images and, if score is set, accuracy on the whole validation set."""
    # unsqueeze(0): a batch of one, like a single upload in the app.
    latency = measure_latency(predict, [inputs[i].unsqueeze(0) for i in range(20)])
    scores = accuracy(predict_classes(predict, inputs), classes, ids, validation) if score else {}
    return {**latency, **scores}


def main() -> None:
    # load_finetuned returns the model and the class names per attribute (output i of a head = i-th name).
    model, classes = load_finetuned(CHECKPOINT, "cpu")
    parameters = sum(p.numel() for p in model.parameters())
    print(f"{parameters / 1e6:.1f} M parameters")

    # Validation set, not the test set: the test set stays untouched until the final comparison.
    # One row per (image, attribute); an image can have labels for several attributes.
    validation = pd.read_parquet(VALIDATION_SET_PATH)
    ids = validation["id"].drop_duplicates().tolist()
    images = load_images(ids)
    # Preprocess once, outside the timing: the same for every variant. On white like in the app (flatten_on_white).
    transform = eval_transform()
    inputs = torch.stack([cast(torch.Tensor, transform(flatten_on_white(images[id]))) for id in ids])
    print(f"{len(ids)} validation images")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    def report(row: dict) -> None:
        rows.append(row)
        print(", ".join(f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}"
                        for key, value in row.items()), flush=True)

    # PyTorch: fp32 and fp16 on every device.
    for name, dtype in DTYPES.items():
        # Size: the whole model incl. heads, saved in this dtype.
        path = MODELS_DIR / f"{name}.pt"
        torch.save(copy.deepcopy(model).to(dtype).state_dict(), path)
        size = model_size_mb(path)

        for device in DEVICES:
            # A fresh copy per variant: converting fp32 -> fp16 -> fp32 in place would keep the rounding of fp16.
            # Moved once, before timing; inside predict it would be measured on every call.
            variant = copy.deepcopy(model).to(device=device, dtype=dtype)
            try:
                # Accuracy only on the fastest device: the same weights give (nearly) the same answers everywhere,
                # and 783 images in fp16 on the CPU would take minutes.
                result = measure(make_predict(variant, device, dtype), inputs, device == DEVICES[-1],
                                 classes, ids, validation)
            except Exception as error:  # noqa: BLE001 - e.g. an operation without fp16 support on this device: a result, not a bug
                print(f"{name} {device}: failed ({error})")
                continue
            report({"variant": name, "device": device, "size_mb": size, **result})

    # ONNX: the fp32 export (to check the export itself) and its int8 quantization, on the CPU.
    # int8 in ONNX Runtime is mainly a CPU technique; the GPU providers do not speed up dynamic int8.
    fp32_onnx = MODELS_DIR / "fp32.onnx"
    int8_onnx = MODELS_DIR / "int8_dynamic.onnx"
    export_onnx(model, inputs[:2], fp32_onnx, output_names=list(classes))
    quantize_int8(fp32_onnx, int8_onnx)
    for name, path in [("onnx_fp32", fp32_onnx), ("onnx_int8_dynamic", int8_onnx)]:
        result = measure(onnx_predict(path), inputs, True, classes, ids, validation)
        report({"variant": name, "device": "cpu", "size_mb": model_size_mb(path), **result})

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "m7_benchmark.csv", index=False)
    print()
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
