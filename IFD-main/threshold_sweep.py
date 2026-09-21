"""
threshold_sweep.py
===================
Finds a better decision threshold than the default 0.5 using predicted
probabilities you already have — no retraining required.

Your current figures show AUC (~0.72-0.74) noticeably higher than F1
(~0.24-0.42). Since AUC is threshold-free and F1 is evaluated at a fixed
0.5 cutoff, that gap is a strong signal the model's *ranking* is fine and
0.5 is simply the wrong operating point for your class balance.

-------------------------------------------------------------------------
INPUT YOU NEED TO PROVIDE
-------------------------------------------------------------------------
This script needs the ground-truth labels and predicted *probabilities*
(not just the 0/1 predictions) for one round/scenario, saved as either:

  (a) a .csv with columns  y_true, y_score
  (b) a .npz with arrays   y_true, y_score

To generate an .npz from a training run, set two environment variables
before running train.py:

    SAVE_PROBS_PATH   path to the output .npz file
    SAVE_PROBS_ROUND  (optional) round number to capture; if omitted,
                      the file is overwritten every round and the last
                      round's probabilities are kept

Example (PowerShell):
    $env:SAVE_PROBS_PATH = "probs_clean.npz"
    $env:SAVE_PROBS_ROUND = "50"
    python train.py --num-clients 10 --num-rounds 50

Example (bash):
    SAVE_PROBS_PATH=probs_clean.npz SAVE_PROBS_ROUND=50 \\
        python train.py --num-clients 10 --num-rounds 50

Only client "0" writes the file (no race conditions in distributed
evaluate). The file is always written with the schema:
    np.savez(path, y_true=<int labels>, y_score=<float proba>)

Do this once per scenario you want a tuned threshold for (Clean,
SignFlip-20%, etc.) since the optimal threshold can differ across them.

-------------------------------------------------------------------------
WHAT IT DOES
-------------------------------------------------------------------------
1. Sweeps thresholds from 0.10 to 0.99 (90 steps, one per 0.01).
2. Reports precision/recall/F1/accuracy at each.
3. Recommends two thresholds:
     - the one that MAXIMIZES F1
     - the one giving the HIGHEST RECALL subject to a precision floor
       you choose (default 0.30) — usually the more defensible choice
       for a security paper, since it's framed as a cost trade-off
       rather than "we picked whatever maximized a number."
4. Saves a precision-recall curve (with F1 iso-contours, matching your
   Fig 8 style) marking the default 0.5 point vs. both recommendations.
5. Saves the full sweep table as CSV so you can put exact numbers in
   the paper or pick a different threshold yourself.

-------------------------------------------------------------------------
USAGE
-------------------------------------------------------------------------
    python threshold_sweep.py --input probs_clean.npz \\
        --precision-floor 0.30 --out-dir ./threshold_results \\
        --label "Clean"

    python threshold_sweep.py --input probs_signflip20.npz \\
        --precision-floor 0.30 --out-dir ./threshold_results \\
        --label "SignFlip-20%"

Run once per scenario. Compare the recommended thresholds across runs —
if they're similar, one global threshold is defensible; if they differ a
lot, report per-scenario thresholds (see step 5 in the earlier plan).
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.6,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


def load_scores(path):
    if path.endswith(".npz"):
        d = np.load(path)
        return np.asarray(d["y_true"]), np.asarray(d["y_score"])
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
        return df["y_true"].to_numpy(), df["y_score"].to_numpy()
    else:
        raise ValueError("Input must be .npz (arrays y_true, y_score) or .csv (columns y_true, y_score)")


def sweep(y_true, y_score, n_steps=90):
    thresholds = np.linspace(0.10, 0.99, n_steps)
    rows = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        rows.append((t, precision, recall, f1, acc))
    return pd.DataFrame(rows, columns=["threshold", "precision", "recall", "f1", "accuracy"])


def recommend(df, precision_floor):
    best_f1_row = df.loc[df["f1"].idxmax()]

    feasible = df[df["precision"] >= precision_floor]
    if len(feasible) > 0:
        best_recall_row = feasible.loc[feasible["recall"].idxmax()]
    else:
        best_recall_row = None  # no threshold reaches the requested precision floor

    return best_f1_row, best_recall_row


def plot_pr_curve(df, default_row, best_f1_row, best_recall_row, label, out_path):
    fig, ax = plt.subplots(figsize=(4.2, 4.0))

    # F1 iso-contours, same idea as your Fig 8
    p = np.linspace(0.001, 1, 300)
    for f1_level in np.arange(0.1, 1.0, 0.1):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (f1_level * p) / (2 * p - f1_level)
        r = np.where((r > 0) & (r <= 1), r, np.nan)
        ax.plot(r, p, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    ax.plot(df["recall"], df["precision"], color="#4C72B0", linewidth=1.5, label="PR curve (all thresholds)")
    ax.scatter([default_row["recall"]], [default_row["precision"]], color="black", marker="x", s=60,
               label=f"Default t=0.50 (F1={default_row['f1']:.3f})", zorder=5)
    ax.scatter([best_f1_row["recall"]], [best_f1_row["precision"]], color="#C44E52", marker="o", s=60,
               label=f"Best F1 t={best_f1_row['threshold']:.2f} (F1={best_f1_row['f1']:.3f})", zorder=5)
    if best_recall_row is not None:
        ax.scatter([best_recall_row["recall"]], [best_recall_row["precision"]], color="#55A868", marker="^", s=60,
                   label=f"Best recall @ precision floor, t={best_recall_row['threshold']:.2f}", zorder=5)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Threshold sweep — {label}", fontsize=10)
    ax.legend(loc="lower left", fontsize=6.5, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.replace(".pdf", ".png"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help=".npz (y_true, y_score) or .csv (y_true, y_score columns)")
    ap.add_argument("--precision-floor", type=float, default=0.30,
                     help="minimum acceptable precision when maximizing recall (default 0.30)")
    ap.add_argument("--out-dir", default="./threshold_results")
    ap.add_argument("--label", default="scenario", help="name for titles/filenames, e.g. 'Clean' or 'SignFlip-20%'")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    y_true, y_score = load_scores(args.input)

    df = sweep(y_true, y_score)
    best_f1_row, best_recall_row = recommend(df, args.precision_floor)

    default_pred = (y_score >= 0.5).astype(int)
    default_row = pd.Series({
        "threshold": 0.5,
        "precision": df.loc[(df["threshold"] - 0.5).abs().idxmin(), "precision"],
        "recall": df.loc[(df["threshold"] - 0.5).abs().idxmin(), "recall"],
        "f1": f1_score(y_true, default_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, default_pred),
    })

    safe_label = args.label.replace(" ", "_").replace("%", "pct")
    csv_path = os.path.join(args.out_dir, f"sweep_{safe_label}.csv")
    plot_path = os.path.join(args.out_dir, f"pr_curve_{safe_label}.pdf")
    df.to_csv(csv_path, index=False)
    plot_pr_curve(df, default_row, best_f1_row, best_recall_row, args.label, plot_path)

    print(f"\n=== {args.label} ===")
    print(f"Default   (t=0.50): precision={default_row['precision']:.3f}  recall={default_row['recall']:.3f}  "
          f"F1={default_row['f1']:.3f}  accuracy={default_row['accuracy']:.3f}")
    print(f"Best F1   (t={best_f1_row['threshold']:.2f}): precision={best_f1_row['precision']:.3f}  "
          f"recall={best_f1_row['recall']:.3f}  F1={best_f1_row['f1']:.3f}  accuracy={best_f1_row['accuracy']:.3f}")
    if best_recall_row is not None:
        print(f"Best recall @ precision>={args.precision_floor:.2f} (t={best_recall_row['threshold']:.2f}): "
              f"precision={best_recall_row['precision']:.3f}  recall={best_recall_row['recall']:.3f}  "
              f"F1={best_recall_row['f1']:.3f}  accuracy={best_recall_row['accuracy']:.3f}")
    else:
        print(f"No threshold reaches precision >= {args.precision_floor:.2f} — try lowering the floor.")
    print(f"\nWrote {csv_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
