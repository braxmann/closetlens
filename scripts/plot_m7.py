"""Plot size against accuracy for every model variant of M7 (from data/evaluation/m7_benchmark.csv).

One panel per attribute, x = size in MB (log scale), y = validation accuracy. Each model is one line through its
variants, from the largest (fp32) to the smallest (int8): the slope shows what every saved MB costs.

Run from the project root: .venv/bin/python -m scripts.plot_m7
Writes docs/m7_curve.png (committed, used in the README).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, NullLocator

from scripts.run_recognizer import RESULTS_DIR

OUTPUT = Path("docs/m7_curve.png")
ATTRIBUTES = ["category", "main_color", "formality"]
# Color follows the model, always the same one, even when a model is missing (validated palette, light mode).
MODELS = {
    "marqo-finetuned": ("Marqo feingetunt (93 M)", "#6250d6"),
    "marqo-probe": ("Lehrer: Marqo + Probe (93 M)", "#2a78d6"),
    "mobilenet-distill": ("Schüler mit Lehrer (4,2 M)", "#eb6834"),
    "mobilenet-labels": ("Schüler ohne Lehrer (4,2 M)", "#1baf7a"),
}
# Shape follows the variant, so identity never depends on color alone.
VARIANTS = {
    "fp32": ("fp32", "o"),
    "fp16": ("fp16", "s"),
    "onnx_int8_static": ("int8 statisch", "D"),
    "onnx_int8_dynamic": ("int8 dynamisch", "^"),
}
# Below this the panel is cut off; broken variants (dynamic int8 on MobileNet: ~0.1) sit on the edge with their value.
Y_FLOOR = 0.70
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_points() -> pd.DataFrame:
    """One row per model and variant that has an accuracy (PyTorch on the GPU, ONNX on the CPU)."""
    table = pd.read_csv(RESULTS_DIR / "m7_benchmark.csv").dropna(subset=["category"])
    # onnx_fp32 is the same model as fp32 (same answers, same size); it only checked the export.
    return table[table["variant"].isin(VARIANTS)]


def main() -> None:
    points = load_points()
    fig, axes = plt.subplots(1, len(ATTRIBUTES), figsize=(15, 5), sharex=True, facecolor=SURFACE)
    for ax, attribute in zip(axes, ATTRIBUTES):
        ax.set_facecolor(SURFACE)
        for model, (model_label, color) in MODELS.items():
            rows = points[points["model"] == model].sort_values("size_mb", ascending=False)
            if rows.empty:
                continue
            # The line connects only working variants; a broken one would drag it to the floor.
            working = rows[rows[attribute] >= Y_FLOOR]
            ax.plot(working["size_mb"], working[attribute], color=color, linewidth=2, zorder=1)
            for row in rows.itertuples():
                value = getattr(row, attribute)
                _, marker = VARIANTS[row.variant]
                y = max(value, Y_FLOOR)
                ax.scatter(row.size_mb, y, color=color, marker=marker, s=64, zorder=2,
                           edgecolors=SURFACE, linewidths=2)  # surface ring where points overlap

        # Broken variants share one spot on the floor, so they get one note listing all their values.
        broken = points[points[attribute] < Y_FLOOR]
        if not broken.empty:
            note = ", ".join(f"{MODELS[row.model][0].split(' (')[0]} {getattr(row, attribute):.2f}".replace(".", ",")
                             for row in broken.itertuples())
            ax.annotate(f"int8 dynamisch: {note}", (broken["size_mb"].min(), Y_FLOOR), xytext=(8, 6),
                        textcoords="offset points", fontsize=8, color=MUTED)
        ax.set_xscale("log")
        # Plain numbers instead of 10^1, 10^2.
        ax.xaxis.set_major_locator(FixedLocator([5, 10, 20, 50, 100, 200, 400]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(lambda value, _: f"{value:g}")
        ax.set_ylim(Y_FLOOR - 0.01, 1.0)
        ax.set_title(attribute, color=INK, loc="left", fontsize=12)
        ax.set_xlabel("Größe in MB (logarithmisch)", color=MUTED)
        ax.grid(color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    axes[0].set_ylabel("Genauigkeit (Validierungsset)", color=MUTED)

    # Two legends: color = model, shape = variant.
    model_handles = [plt.Line2D([], [], color=color, linewidth=2, label=label)
                     for model, (label, color) in MODELS.items() if model in set(points["model"])]
    variant_handles = [plt.Line2D([], [], color=MUTED, marker=marker, linestyle="", markersize=7, label=label)
                       for label, marker in VARIANTS.values()]
    fig.legend(handles=model_handles + variant_handles, loc="lower center", ncol=4, frameon=False,
               labelcolor=INK, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Was kostet jedes eingesparte MB an Genauigkeit? (Werte unter 0,70 am unteren Rand)",
                 color=INK, x=0.06, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=150, facecolor=SURFACE)
    print(f"Written to {OUTPUT}")


if __name__ == "__main__":
    main()
