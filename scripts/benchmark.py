"""Measure size, latency and accuracy of model variants for M7 (Verkleinern).

Measures every model whose files exist here: the fine-tuned Marqo (M6), the distillation teacher
(Marqo + linear probes) and every student from scripts/distill.py. Each in PyTorch fp32/fp16 and as ONNX
fp32, int8 dynamic and int8 static.

Run from the project root: .venv/bin/python -m scripts.benchmark [--models marqo-probe mobilenet-distill]
Writes data/evaluation/m7_benchmark.csv (one row per model, variant and device).
"""
import argparse
import copy
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import open_clip
import pandas as pd
import timm
import torch
from torch import nn

from scripts.make_test_set import SPLIT_PATH
from scripts.make_validation_set import VALIDATION_SET_PATH
from scripts.run_recognizer import RESULTS_DIR
from wardrobe.compression.export import export_onnx, onnx_predict, quantize_int8, quantize_int8_static
from wardrobe.evaluation.benchmark import measure_latency, model_size_mb
from wardrobe.evaluation.dataset import load_images
from wardrobe.models.finetuned import flatten_on_white, load_finetuned
from wardrobe.training.finetune import MultiHeadClassifier, eval_transform
from wardrobe.training.linear_probe import PROBES_DIR, load_probes, probes_path

# The model the app uses (M6): Marqo with the last 2 blocks fine-tuned, one head per attribute (only on the PC).
CHECKPOINT = Path("data/models/finetune_marqo-last2-ls.pt")
# The distillation teacher: original Marqo + linear probes (train_probe.py --save). Students: scripts/distill.py.
TEACHER_MODEL = "hf-hub:Marqo/marqo-fashionSigLIP"
CALIBRATION_IMAGES = 100
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


class L2Normalize(nn.Module):
    """Scales every embedding to length 1, like OpenClipEmbedder(normalize=True) did for the probe's features."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Written out instead of F.normalize: that one exports with an "Expand" node on which ONNX Runtime's
        # shape inference (needed before quantizing) crashes.
        return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True)


def load_teacher() -> tuple[MultiHeadClassifier, dict[str, list[str]]]:
    """The distillation teacher as one PyTorch model: original Marqo image tower + the linear probes as heads.

    A logistic regression is exactly a linear layer (score = X @ coef.T + intercept), so its weights can be copied
    into nn.Linear. Then the teacher can be timed, exported and quantized like every other model.
    """
    _, probes = load_probes(probes_path(TEACHER_MODEL))
    clip_model, _, _ = open_clip.create_model_and_transforms(TEACHER_MODEL)
    backbone = nn.Sequential(cast(nn.Module, clip_model.visual), L2Normalize())
    classes = {attribute: list(probe.classes_) for attribute, probe in probes.items()}
    model = MultiHeadClassifier(backbone, feature_dim=768, classes=classes)
    with torch.no_grad():
        for attribute, probe in probes.items():
            weight, bias = probe.coef_, probe.intercept_
            if weight.shape[0] == 1:
                # Two classes: scikit-learn stores one row "how much class 1"; [0, row] gives the same answers.
                weight, bias = np.vstack([np.zeros_like(weight), weight]), np.concatenate([[0.0], bias])
            model.heads[attribute].weight.copy_(torch.from_numpy(weight))
            model.heads[attribute].bias.copy_(torch.from_numpy(bias))
    return model.eval(), classes


def load_student(path: Path) -> tuple[MultiHeadClassifier, dict[str, list[str]]]:
    """A student from scripts/distill.py, rebuilt as it was trained, then its trained weights loaded."""
    checkpoint = torch.load(path, map_location="cpu")
    backbone = timm.create_model(checkpoint["student"], pretrained=False, num_classes=0)
    feature_dim = checkpoint["state_dict"]["heads." + next(iter(checkpoint["classes"])) + ".weight"].shape[1]
    model = MultiHeadClassifier(backbone, feature_dim=feature_dim, classes=checkpoint["classes"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint["classes"]


def available_models() -> dict[str, Callable[[], tuple[MultiHeadClassifier, dict[str, list[str]]]]]:
    """Every model whose files exist on this machine, by name. Loaded only when its turn comes (memory)."""
    models: dict[str, Callable[[], tuple[MultiHeadClassifier, dict[str, list[str]]]]] = {}
    if CHECKPOINT.exists():
        models["marqo-finetuned"] = lambda: load_finetuned(CHECKPOINT, "cpu")
    if probes_path(TEACHER_MODEL).exists():
        models["marqo-probe"] = load_teacher
    for path in sorted(PROBES_DIR.glob("student_*.pt")):
        # path=path: bind the current path now; a plain lambda would see only the last one of the loop.
        models[path.stem.removeprefix("student_")] = lambda path=path: load_student(path)
    return models


def calibration_inputs(transform, n: int = CALIBRATION_IMAGES) -> list[np.ndarray]:
    """n training images (never validation images) for static quantization, one batch of one each."""
    split = pd.read_parquet(SPLIT_PATH)
    train_ids = split[split["split"] == "train"]["id"].sample(n=n, random_state=0).tolist()
    images = load_images(train_ids)
    return [cast(torch.Tensor, transform(flatten_on_white(images[id]))).unsqueeze(0).numpy() for id in train_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", help="only these models (default: all available)")
    args = parser.parse_args()

    # Validation set, not the test set: the test set stays untouched until the final comparison.
    # One row per (image, attribute); an image can have labels for several attributes.
    validation = pd.read_parquet(VALIDATION_SET_PATH)
    ids = validation["id"].drop_duplicates().tolist()
    images = load_images(ids)
    # Preprocess once, outside the timing: the same for every variant. On white like in the app (flatten_on_white).
    # All models get the same preprocessing (Marqo's own differs only in bicubic instead of bilinear resizing).
    transform = eval_transform()
    inputs = torch.stack([cast(torch.Tensor, transform(flatten_on_white(images[id]))) for id in ids])
    calibration = calibration_inputs(transform)
    print(f"{len(ids)} validation images, {len(calibration)} calibration images")

    rows = []

    def report(row: dict) -> None:
        rows.append(row)
        print(", ".join(f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}"
                        for key, value in row.items()), flush=True)

    models = available_models()
    for model_name, load in models.items():
        if args.models and model_name not in args.models:
            continue
        model, classes = load()
        parameters = sum(p.numel() for p in model.parameters())
        print(f"\n{model_name}: {parameters / 1e6:.1f} M parameters", flush=True)
        folder = MODELS_DIR / model_name
        folder.mkdir(parents=True, exist_ok=True)

        # PyTorch: fp32 and fp16 on every device.
        for name, dtype in DTYPES.items():
            # Size: the whole model incl. heads, saved in this dtype.
            path = folder / f"{name}.pt"
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
                    print(f"{model_name} {name} {device}: failed ({error})")
                    continue
                report({"model": model_name, "variant": name, "device": device, "size_mb": size, **result})

        # ONNX on the CPU: the fp32 export (to check the export itself) and two int8 quantizations.
        # int8 in ONNX Runtime is mainly a CPU technique; the GPU providers do not speed it up.
        fp32_onnx = folder / "fp32.onnx"
        export_onnx(model, inputs[:2], fp32_onnx, output_names=list(classes))
        quantized = {"onnx_fp32": fp32_onnx}
        for name, quantize in [
            ("onnx_int8_dynamic", lambda out: quantize_int8(fp32_onnx, out)),
            ("onnx_int8_static", lambda out: quantize_int8_static(fp32_onnx, out, calibration)),
        ]:
            try:
                quantize(folder / f"{name}.onnx")
                quantized[name] = folder / f"{name}.onnx"
            except Exception as error:  # noqa: BLE001 - report and go on with the other variants
                print(f"{model_name} {name}: quantization failed ({error})")
        for name, path in quantized.items():
            result = measure(onnx_predict(path), inputs, True, classes, ids, validation)
            report({"model": model_name, "variant": name, "device": "cpu", "size_mb": model_size_mb(path), **result})

    table = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "m7_benchmark.csv"
    if args.models and csv_path.exists():
        # Only some models measured: keep the other models' rows from earlier runs.
        earlier = pd.read_csv(csv_path)
        table = pd.concat([earlier[~earlier["model"].isin(table["model"])], table], ignore_index=True)
    table.to_csv(csv_path, index=False)
    print()
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
