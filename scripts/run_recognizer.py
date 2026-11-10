"""Run an attribute recognizer over every image of the frozen test set and store its answers.

Each answer is appended as one JSON line to data/evaluation/<run>.jsonl as soon as it arrives, so an
interrupted run continues where it stopped when started again.

Run from the project root:
    .venv/bin/python -m scripts.run_recognizer --model qwen2.5vl:3b
    .venv/bin/python -m scripts.run_recognizer --recognizer zero-shot --model hf-hub:Marqo/marqo-fashionSigLIP
"""
import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd

from wardrobe.evaluation.dataset import load_images
from wardrobe.models.attribute_recognizer import AttributeRecognizer, OllamaAttributeRecognizer
from scripts.make_test_set import TEST_SET_PATH

RESULTS_DIR = Path("data/evaluation")


def make_recognizer(kind: str, model: str) -> AttributeRecognizer:
    if kind == "ollama":
        return OllamaAttributeRecognizer(model)
    # Imported here so a VLM run does not need torch.
    from wardrobe.models.embedder import OpenClipEmbedder
    from wardrobe.models.zero_shot import ZeroShotAttributeRecognizer
    return ZeroShotAttributeRecognizer(OpenClipEmbedder(model))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recognizer", choices=["ollama", "zero-shot"], default="ollama")
    parser.add_argument("--model", default="qwen2.5vl:3b", help="Ollama model or open_clip model name")
    parser.add_argument("--run", help="name of the result file, defaults to the model name")
    args = parser.parse_args()

    # "hf-hub:Marqo/marqo-fashionSigLIP" -> "marqo-fashionSigLIP", "qwen2.5vl:3b" -> "qwen2.5vl-3b"
    run = args.run or re.sub(r"[:/]", "-", args.model.removeprefix("hf-hub:").split("/")[-1])
    output = RESULTS_DIR / f"{run}.jsonl"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ids = sorted(pd.read_parquet(TEST_SET_PATH)["id"].unique())
    done = set()
    if output.exists():
        done = {json.loads(line)["id"] for line in output.read_text(encoding="utf-8").splitlines()}
    todo = [id for id in ids if id not in done]
    print(f"{len(ids)} test images, {len(done)} already done, {len(todo)} to go -> {output}", flush=True)

    recognizer = make_recognizer(args.recognizer, args.model)
    images = load_images(todo)
    with output.open("a", encoding="utf-8") as file:
        for number, id in enumerate(todo, start=1):
            record = {"id": int(id), "model": args.model}
            start = time.perf_counter()
            try:
                record["prediction"] = recognizer.recognize_attributes(images[id]).model_dump()
            except Exception as error:  # a broken answer is a result too, not a reason to stop the run
                record["error"] = f"{type(error).__name__}: {error}"
            record["seconds"] = round(time.perf_counter() - start, 3)
            file.write(json.dumps(record) + "\n")
            file.flush()
            if number % 100 == 0 or number == len(todo):
                print(f"{number}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
