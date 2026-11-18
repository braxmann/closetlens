"""Measure size and latency of model variants for M7 (Verkleinern).

Run from the project root: .venv/bin/python -m scripts.benchmark
"""
import copy
from pathlib import Path

import open_clip
import torch

from wardrobe.evaluation.benchmark import measure_latency, model_size_mb
from wardrobe.evaluation.dataset import load_images, load_products

MODEL_NAME = "hf-hub:Marqo/marqo-fashionSigLIP"
MODELS_DIR = Path("data/models/m7")
DEVICES = ["cpu", "mps"]
DTYPES = {"fp32": torch.float32, "fp16": torch.float16}


def main() -> None:
    # Returns (model, train_transform, eval_transform); only the image tower and the eval transform are needed.
    model, _, eval_transform = open_clip.create_model_and_transforms(MODEL_NAME)
    visual = model.visual.eval()
    parameters = sum(p.numel() for p in visual.parameters())
    print(f"{parameters / 1e6:.1f} M parameters")

    # Size: only the weights of the image tower, the same part the fine-tuned recognizer uses.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, dtype in DTYPES.items():
        path = MODELS_DIR / f"{name}.pt"
        torch.save(copy.deepcopy(visual).to(dtype).state_dict(), path)
        print(f"{name}: {model_size_mb(path):.1f} MB")

    # Preprocess once, outside the timing: resize + normalize is the same for every variant.
    ids = load_products()["id"].head(20).tolist()
    images = load_images(ids)
    inputs = [eval_transform(image).unsqueeze(0) for image in images.values()]  # unsqueeze(0): a batch of one

    for device in DEVICES:
        for name, dtype in DTYPES.items():
            # A fresh copy per variant: converting fp32 -> fp16 -> fp32 in place would keep the rounding of fp16.
            # Moved once, before timing; inside predict it would be measured on every call.
            variant = copy.deepcopy(visual).to(device=device, dtype=dtype)

            def predict(x: torch.Tensor) -> torch.Tensor:
                with torch.inference_mode():
                    # The input needs the model's dtype, otherwise the matrix multiplication fails.
                    # .cpu() waits for the result; on mps/cuda the call would otherwise return before the work is done.
                    return variant(x.to(device=device, dtype=dtype)).cpu()

            try:
                latency = measure_latency(predict, inputs)
                print(f"{device} {name}: {latency['median_ms']:.1f} ms median, {latency['p90_ms']:.1f} ms p90")
            except Exception as error:  # e.g. an operation without fp16 support on this device: a result, not a bug
                print(f"{device} {name}: failed ({error})")


if __name__ == "__main__":
    main()
