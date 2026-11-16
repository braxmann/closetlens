"""Partial fine-tuning of the Marqo image tower with one head per attribute, measured on the validation set.

Run from the project root (GPU strongly recommended):
    .venv/bin/python -m scripts.finetune                      # full run
    .venv/bin/python -m scripts.finetune --limit 512 --epochs 1   # quick check that everything works
Writes data/evaluation/finetune_<run>.csv (one row per epoch) and the best model to data/models/finetune_<run>.pt.
"""
import argparse
import math
import time

import numpy as np
import open_clip
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.make_test_set import SPLIT_PATH
from scripts.make_validation_set import VALIDATION_SET_PATH
from scripts.run_recognizer import RESULTS_DIR
from wardrobe.evaluation.dataset import PRODUCTS_PATH, load_products
from wardrobe.evaluation.split import ATTRIBUTE_MAPS, add_labels
from wardrobe.training.finetune import (
    MISSING, GarmentDataset, MultiHeadClassifier, balanced_weights, count_parameters, encode_labels,
    eval_transform, freeze_except_last_blocks, train_transform,
)
from wardrobe.training.linear_probe import PROBES_DIR

# Linear probe on the validation set (train_probe.py), the numbers to beat.
PROBE = {"category": 0.980, "main_color": 0.826, "formality": 0.940}


def load_image_bytes(ids: np.ndarray) -> list[bytes]:
    """The compressed images in the order of ids (decoding happens later, in the DataLoader workers)."""
    table = pq.read_table(PRODUCTS_PATH, columns=["id", "image"], filters=[("id", "in", ids.tolist())]).to_pandas()
    by_id = dict(zip(table["id"].astype(int), table["image"]))
    return [by_id[int(id)] for id in ids]


def run_epoch(model, loader, criteria, device, optimizer=None, scaler=None, scheduler=None):
    """One pass over the loader. With an optimizer it trains, without one it only measures.

    Returns (mean loss, {attribute: (correct, counted)}).
    """
    training = optimizer is not None
    # train() / eval() switch layers like dropout on and off; the ViT has hardly any, but it is the habit.
    model.train(training)
    total_loss, batches = 0.0, 0
    hits = {attribute: [0, 0] for attribute in criteria}
    # Without gradients when measuring: faster and much less VRAM.
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            # Mixed precision: most calculations in float16, which the RTX 2070's tensor cores run much faster
            # and which halves the memory for activations. Sensitive steps stay in float32 automatically.
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                outputs = model(images)
            loss = torch.zeros((), device=device)
            for attribute, criterion in criteria.items():
                target = labels[attribute].to(device)
                known = target != MISSING
                if known.any():  # a batch without any label for this attribute would give 0/0 = nan
                    loss = loss + criterion(outputs[attribute].float(), target)
                    predicted = outputs[attribute].argmax(dim=1)
                    hits[attribute][0] += int((predicted[known] == target[known]).sum())
                    hits[attribute][1] += int(known.sum())
            if training:
                optimizer.zero_grad(set_to_none=True)
                # The scaler multiplies the loss before backward(), so tiny float16 gradients do not round to zero,
                # and divides them back before the optimizer step.
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            total_loss += loss.item()
            batches += 1
    return total_loss / max(batches, 1), hits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="hf-hub:Marqo/marqo-fashionSigLIP")
    parser.add_argument("--run", default="marqo-last2", help="name for the result files")
    parser.add_argument("--unfreeze-blocks", type=int, default=2, help="how many of the 12 ViT blocks learn")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr-heads", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=2e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="use only this many training images (quick test)")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    # --- Data: the same training products and validation sample as the linear probe ---
    products = add_labels(load_products()).merge(pd.read_parquet(SPLIT_PATH), on="id")
    train = products[products["split"] == "train"].sort_values("id")
    if args.limit:
        train = train.sample(n=args.limit, random_state=0).sort_values("id")
    validation_set = pd.read_parquet(VALIDATION_SET_PATH)
    val_ids = np.sort(validation_set["id"].unique())

    # The classes are the labels that occur in training; their order fixes which output number means what.
    classes = {attribute: sorted(train[attribute].dropna().unique()) for attribute in ATTRIBUTE_MAPS}
    train_labels = {a: encode_labels(train[a].to_numpy(), names) for a, names in classes.items()}
    # Validation: per attribute only the products that are in that attribute's sample, everything else MISSING.
    val_labels = {}
    for attribute, names in classes.items():
        sample = validation_set[validation_set["attribute"] == attribute].set_index("id")["label"]
        val_labels[attribute] = encode_labels(pd.Series(val_ids).map(sample).to_numpy(), names)

    print(f"Loading {len(train)} training and {len(val_ids)} validation images ...", flush=True)
    train_data = GarmentDataset(load_image_bytes(train["id"].to_numpy()), train_labels, train_transform())
    val_data = GarmentDataset(load_image_bytes(val_ids), val_labels, eval_transform())
    # Workers decode and augment the next batches on the CPU while the GPU computes the current one.
    loader_options = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device == "cuda",
                          persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_data, shuffle=True, drop_last=True, **loader_options)
    val_loader = DataLoader(val_data, shuffle=False, **loader_options)

    # --- Model: only the image tower of Marqo; the text tower is not needed for classification ---
    clip_model, _, _ = open_clip.create_model_and_transforms(args.model)
    visual = clip_model.visual
    del clip_model
    freeze_except_last_blocks(visual, args.unfreeze_blocks)
    model = MultiHeadClassifier(visual, feature_dim=768, classes=classes).to(device)
    trainable, total = count_parameters(model)
    print(f"{trainable / 1e6:.1f} M of {total / 1e6:.1f} M weights are trained ({args.unfreeze_blocks} blocks + heads)")

    criteria = {a: nn.CrossEntropyLoss(weight=balanced_weights(train_labels[a], len(names)).to(device),
                                       ignore_index=MISSING)
                for a, names in classes.items()}
    # Two learning rates: the new heads start from random numbers and must move a lot; the pretrained blocks
    # only need a nudge, a large step would destroy what they already know ("catastrophic forgetting").
    backbone_parameters = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": model.heads.parameters(), "lr": args.lr_heads},
        {"params": backbone_parameters, "lr": args.lr_backbone},
    ], weight_decay=0.01)
    # Learning rate schedule: warm up over the first 5 % of the steps, then fall along a cosine curve to zero.
    steps = args.epochs * len(train_loader)
    warmup = max(1, steps // 20)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: step / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))
    )
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    # --- Training ---
    history, best = [], -1.0
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PROBES_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = PROBES_DIR / f"finetune_{args.run}.pt"
    for epoch in range(1, args.epochs + 1):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        train_loss, train_hits = run_epoch(model, train_loader, criteria, device, optimizer, scaler, scheduler)
        val_loss, val_hits = run_epoch(model, val_loader, criteria, device)
        seconds = time.perf_counter() - start

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        for attribute in classes:
            row[f"train_{attribute}"] = train_hits[attribute][0] / max(train_hits[attribute][1], 1)
            row[f"val_{attribute}"] = val_hits[attribute][0] / max(val_hits[attribute][1], 1)
        row["seconds"] = round(seconds)
        row["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else 0
        history.append(row)

        scores = "  ".join(f"{a} {row[f'val_{a}']:.3f} (probe {PROBE[a]:.3f})" for a in classes)
        print(f"epoch {epoch}: loss train {train_loss:.3f} val {val_loss:.3f} | {scores} | "
              f"{seconds:.0f} s, {row['peak_vram_gb']} GB VRAM", flush=True)

        # Keep the epoch with the best mean validation accuracy, not simply the last one.
        mean_score = np.mean([row[f"val_{a}"] for a in classes])
        if mean_score > best:
            best = mean_score
            torch.save({"model": args.model, "classes": classes, "unfreeze_blocks": args.unfreeze_blocks,
                        "epoch": epoch, "state_dict": model.state_dict()}, checkpoint_path)
        pd.DataFrame(history).to_csv(RESULTS_DIR / f"finetune_{args.run}.csv", index=False)

    print(f"Best mean validation accuracy {best:.3f}, saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
