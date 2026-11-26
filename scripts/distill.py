"""Knowledge distillation (M7): train a small student model with the linear probe as teacher.

The teacher is the linear probe on Marqo features (train_probe.py --save). Its logits for every training image
come from the cached features in seconds, so the big Marqo model never runs during training.
The student is a small pretrained timm model with one head per attribute (the MultiHeadClassifier from M6).

Run from the project root:
    .venv/bin/python -m scripts.distill --alpha 0.0 --run mobilenet-labels      # without teacher (baseline)
    .venv/bin/python -m scripts.distill --alpha 0.7 --run mobilenet-distill     # with teacher
    .venv/bin/python -m scripts.distill --limit 512 --epochs 1                  # quick check that everything works
Writes data/evaluation/distill_<run>.csv (one row per epoch) and the best model to data/models/student_<run>.pt.
"""
import argparse
import math
import time

import numpy as np
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader

from scripts.cache_features import features_path
from scripts.finetune import load_image_bytes
from scripts.make_test_set import SPLIT_PATH
from scripts.make_validation_set import VALIDATION_SET_PATH
from scripts.run_recognizer import RESULTS_DIR
from wardrobe.evaluation.dataset import load_products
from wardrobe.evaluation.split import ATTRIBUTE_MAPS, add_labels
from wardrobe.training.distill import multi_head_loss
from wardrobe.training.features import load_features
from wardrobe.training.finetune import (
    MISSING, GarmentDataset, MultiHeadClassifier, count_parameters, encode_labels, eval_transform, train_transform,
)
from wardrobe.training.linear_probe import PROBES_DIR, load_probes, probes_path, rows_for

TEACHER_MODEL = "hf-hub:Marqo/marqo-fashionSigLIP"


class TeacherDataset(GarmentDataset):
    """GarmentDataset plus the teacher's logits for every image, computed beforehand."""

    def __init__(self, images, labels, teacher_logits: dict[str, np.ndarray], transform):
        super().__init__(images, labels, transform)
        self.teacher_logits = teacher_logits

    def __getitem__(self, i: int):
        image, labels = super().__getitem__(i)
        teacher = {attribute: torch.from_numpy(logits[i]) for attribute, logits in self.teacher_logits.items()}
        return image, labels, teacher


def teacher_logits(probes: dict, ids: np.ndarray) -> dict[str, np.ndarray]:
    """Raw scores of the linear probes for the given products, one array (n_images, n_classes) per attribute."""
    feature_ids, vectors = load_features(features_path(TEACHER_MODEL))
    X = rows_for(feature_ids, vectors, ids)
    result = {}
    for attribute, probe in probes.items():
        scores = probe.decision_function(X).astype(np.float32)
        if scores.ndim == 1:
            # Two classes (formality): scikit-learn returns only one score d per image, "how much class 1".
            # [0, d] gives the same probabilities under softmax as the probe's sigmoid(d).
            scores = np.stack([np.zeros_like(scores), scores], axis=1)
        result[attribute] = scores
    return result


def evaluate(model, loader, device) -> dict[str, float]:
    """Accuracy per attribute on the validation set (only images with a label for that attribute)."""
    model.eval()
    hits: dict[str, list[int]] = {}
    with torch.inference_mode():
        for images, labels in loader:
            outputs = model(images.to(device))
            for attribute, logits in outputs.items():
                target = labels[attribute].to(device)
                known = target != MISSING
                correct, counted = hits.setdefault(attribute, [0, 0])
                hits[attribute] = [correct + int((logits.argmax(dim=1)[known] == target[known]).sum()),
                                   counted + int(known.sum())]
    return {attribute: correct / max(counted, 1) for attribute, (correct, counted) in hits.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--student", default="mobilenetv3_large_100", help="any timm model name")
    parser.add_argument("--run", default="mobilenet-distill", help="name for the result files")
    parser.add_argument("--alpha", type=float, default=0.7, help="weight of the teacher; 0 = labels only")
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="use only this many training images (quick test)")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)

    # --- Data: the same training products and validation sample as probe and fine-tuning ---
    products = add_labels(load_products()).merge(pd.read_parquet(SPLIT_PATH), on="id")
    train = products[products["split"] == "train"].sort_values("id")
    if args.limit:
        train = train.sample(n=args.limit, random_state=0).sort_values("id")
    validation_set = pd.read_parquet(VALIDATION_SET_PATH)
    val_ids = np.sort(validation_set["id"].unique())

    # The classes come from the teacher: output i of a student head must mean the same class as column i of the
    # teacher's logits. (Taking them from the training rows would break with --limit, when a class is missing.)
    _, probes = load_probes(probes_path(TEACHER_MODEL))
    classes = {attribute: list(probes[attribute].classes_) for attribute in ATTRIBUTE_MAPS}
    train_labels = {a: encode_labels(train[a].to_numpy(), names) for a, names in classes.items()}
    val_labels = {}
    for attribute, names in classes.items():
        sample = validation_set[validation_set["attribute"] == attribute].set_index("id")["label"]
        val_labels[attribute] = encode_labels(pd.Series(val_ids).map(sample).to_numpy(), names)

    # The teacher has an answer for every training image, also where a label is MISSING (often formality).
    teacher = teacher_logits(probes, train["id"].to_numpy())

    print(f"Loading {len(train)} training and {len(val_ids)} validation images ...", flush=True)
    # Note: the teacher saw the unaugmented image, the student sees a random crop/flip of it. Close enough,
    # and it saves running the teacher in every epoch.
    train_data = TeacherDataset(load_image_bytes(train["id"].to_numpy()), train_labels, teacher, train_transform())
    val_data = GarmentDataset(load_image_bytes(val_ids), val_labels, eval_transform())
    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers, persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_data, shuffle=True, drop_last=True, **loader_options)
    val_loader = DataLoader(val_data, shuffle=False, **loader_options)

    # --- Student: pretrained on ImageNet; num_classes=0 removes its own classifier, it then returns features ---
    # It gets the same preprocessing as Marqo (values in [-1, 1]); all its weights learn, so it adapts to that.
    backbone = timm.create_model(args.student, pretrained=True, num_classes=0)
    # The feature size is measured with one dummy image: backbone.num_features is not always what comes out
    # (MobileNetV3 says 960, but its last layer expands to 1280).
    with torch.no_grad():
        feature_dim = backbone.eval()(torch.zeros(1, 3, 224, 224)).shape[1]
    model = MultiHeadClassifier(backbone, feature_dim=feature_dim, classes=classes).to(device)
    _, total = count_parameters(model)
    print(f"Student {args.student}: {total / 1e6:.1f} M weights, all trained | alpha {args.alpha}, "
          f"T {args.temperature} | {device}")

    # All weights learn (the student is small and must adapt completely), so one learning rate for all.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * len(train_loader)
    warmup = max(1, steps // 20)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: step / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))
    )

    # --- Training ---
    history, best = [], -1.0
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = PROBES_DIR / f"student_{args.run}.pt"
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        model.train()
        total_loss, batches = 0.0, 0
        for images, labels, teacher_batch in train_loader:
            outputs = model(images.to(device))
            loss = multi_head_loss(
                outputs,
                {a: logits.to(device) for a, logits in teacher_batch.items()},
                {a: target.to(device) for a, target in labels.items()},
                args.temperature, args.alpha,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            batches += 1
        scores = evaluate(model, val_loader, device)
        seconds = time.perf_counter() - start

        row = {"epoch": epoch, "train_loss": total_loss / max(batches, 1),
               **{f"val_{a}": score for a, score in scores.items()}, "seconds": round(seconds)}
        history.append(row)
        print(f"epoch {epoch}: loss {row['train_loss']:.3f} | "
              + "  ".join(f"{a} {s:.3f}" for a, s in scores.items()) + f" | {seconds:.0f} s", flush=True)

        # Keep the epoch with the best mean validation accuracy, like in finetune.py.
        mean_score = float(np.mean(list(scores.values())))
        if mean_score > best:
            best = mean_score
            torch.save({"student": args.student, "classes": classes, "alpha": args.alpha,
                        "temperature": args.temperature, "epoch": epoch, "state_dict": model.state_dict()},
                       checkpoint_path)
        pd.DataFrame(history).to_csv(RESULTS_DIR / f"distill_{args.run}.csv", index=False)

    print(f"Best mean validation accuracy {best:.3f}, saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
