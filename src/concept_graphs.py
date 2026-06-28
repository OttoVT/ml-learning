"""Generate the three explanatory graphs from the source ChatGPT plan.

This is the cleaned, runnable version of the script in
``plan/chatgpt-source-plan.md``. It writes PNGs into ``assets/generated/``
(relative to the repo root) instead of the sandbox path used in the original.

Run it:

    pip install -r requirements.txt
    python src/concept_graphs.py

Then open the PNGs under ``assets/generated/``.

The three graphs:
1. Cross-entropy loss vs. predicted probability of the true class.
2. Precision / recall / F1 trade-off as the decision threshold moves.
3. Cosine similarity vs. the angle between two vectors.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # render to files without needing a display
import matplotlib.pyplot as plt

# Write next to the repo, not the sandbox path from the original script.
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "assets" / "generated"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def cross_entropy_curve() -> Path:
    """Cross-entropy = -log(p) for the probability p of the correct class."""
    p = np.linspace(0.001, 0.999, 500)
    ce = -np.log(p)
    plt.figure(figsize=(7, 4.5))
    plt.plot(p, ce)
    plt.xlabel("Predicted probability assigned to the true class")
    plt.ylabel("Cross-entropy loss")
    plt.title("Cross-entropy strongly punishes confident wrong predictions")
    plt.grid(True, alpha=0.3)
    path = OUT_DIR / "cross_entropy_curve.png"
    plt.savefig(path, bbox_inches="tight", dpi=160)
    plt.close()
    return path


def precision_recall_f1_tradeoff() -> Path:
    """Illustrative precision/recall/F1 curves as the decision threshold moves."""
    thresholds = np.linspace(0.01, 0.99, 99)
    # Illustrative smooth curves, not from a real model.
    precision = 0.45 + 0.5 * thresholds**0.7
    recall = 0.95 - 0.75 * thresholds**1.2
    f1 = 2 * precision * recall / (precision + recall)
    plt.figure(figsize=(7, 4.5))
    plt.plot(thresholds, precision, label="Precision")
    plt.plot(thresholds, recall, label="Recall")
    plt.plot(thresholds, f1, label="F1")
    plt.xlabel("Decision threshold")
    plt.ylabel("Metric value")
    plt.title("Precision, recall, and F1 usually trade off")
    plt.legend()
    plt.grid(True, alpha=0.3)
    path = OUT_DIR / "precision_recall_f1_tradeoff.png"
    plt.savefig(path, bbox_inches="tight", dpi=160)
    plt.close()
    return path


def cosine_similarity_curve() -> Path:
    """Cosine similarity as a function of the angle between two vectors."""
    angles = np.linspace(0, 180, 361)
    cos_sim = np.cos(np.deg2rad(angles))
    plt.figure(figsize=(7, 4.5))
    plt.plot(angles, cos_sim)
    plt.xlabel("Angle between two vectors, degrees")
    plt.ylabel("Cosine similarity")
    plt.title("Cosine similarity measures direction, not magnitude")
    plt.grid(True, alpha=0.3)
    path = OUT_DIR / "cosine_similarity_curve.png"
    plt.savefig(path, bbox_inches="tight", dpi=160)
    plt.close()
    return path


def main() -> None:
    paths = [
        cross_entropy_curve(),
        precision_recall_f1_tradeoff(),
        cosine_similarity_curve(),
    ]
    print("Generated graphs:")
    for p in paths:
        print(p.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
