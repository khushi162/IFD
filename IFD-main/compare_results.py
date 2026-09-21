#!/usr/bin/env python3
"""
compare_results.py — Build a comparison table of your model vs baselines.

Reads all result JSONs (model + baselines) and prints the final-round
metrics (AUC-ROC, F1, Recall, Precision, Accuracy, Loss) for every
experiment, grouped by experiment label.

Usage:
    python compare_results.py
    python compare_results.py --output results/comparison_table.csv

Output:
    - Console: formatted table
    - File (optional): CSV for import into Excel / LaTeX
"""

import argparse
import csv
import json
import os
import sys


# ============================================================================
# Configuration
# ============================================================================

RESULTS_DIR   = "./results"
BASELINES_DIR = os.path.join(RESULTS_DIR, "baselines")

# Your model results live directly in RESULTS_DIR
MODEL_NAME = "CascadeRouter"

# Baselines to include in the table
BASELINES = ["b1_fedavg", "b2_krum"]

BASELINE_LABELS = {
    "b1_fedavg": "FedAvg",
    "b2_krum":   "Krum",
}

# Canonical experiment order
EXPERIMENTS = [
    "CleanRun_Simple",
    "Attack_SignFlip_10pct",
    "Attack_SignFlip_20pct",
    "Attack_SignFlip_40pct",
    "Attack_LabelFlip_10pct",
    "Attack_LabelFlip_20pct",
    "Attack_LabelFlip_40pct",
    "Attack_ModelReplace_10pct",
    "Attack_ModelReplace_20pct",
    "Attack_ModelReplace_40pct",
]

METRICS = ["auc", "f1", "recall", "precision", "accuracy", "loss"]

METRIC_LABELS = {
    "auc":       "AUC-ROC",
    "f1":        "F1",
    "recall":    "Recall",
    "precision": "Precision",
    "accuracy":  "Accuracy",
    "loss":      "Loss",
}


# ============================================================================
# Loader
# ============================================================================

def load_final_metrics(path: str) -> dict:
    """
    Extract the final-round metrics from a result JSON.

    metrics_distributed holds per-round aggregated client metrics:
        {metric_name: [[round, value], ...]}

    losses_distributed holds per-round loss:
        [[round, loss], ...]

    Returns a dict with keys matching METRICS, or None values if missing.
    """
    result = {m: None for m in METRICS}

    if not os.path.isfile(path):
        return result

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  WARNING: could not parse {path}: {e}", file=sys.stderr)
        return result

    # metrics_distributed: {metric: [[round, value], ...]}
    md = data.get("metrics_distributed", {})
    for metric in ["auc", "f1", "recall", "precision", "accuracy"]:
        series = md.get(metric, [])
        if series:
            # Take the last round's value
            result[metric] = series[-1][1]

    # loss from losses_distributed
    ld = data.get("losses_distributed", [])
    if ld:
        result["loss"] = ld[-1][1]

    return result


def collect_all() -> dict:
    """
    Returns:
        {experiment_label: {model_name: {metric: value}}}
    """
    table = {}

    for exp in EXPERIMENTS:
        table[exp] = {}

        # Your model
        model_path = os.path.join(RESULTS_DIR, f"{exp}.json")
        table[exp][MODEL_NAME] = load_final_metrics(model_path)

        # Baselines
        for bl in BASELINES:
            bl_path = os.path.join(BASELINES_DIR, bl, f"{exp}.json")
            label = BASELINE_LABELS[bl]
            table[exp][label] = load_final_metrics(bl_path)

    return table


# ============================================================================
# Formatting
# ============================================================================

def fmt(value) -> str:
    if value is None:
        return "—"
    return f"{value:.4f}"


def print_table(table: dict) -> None:
    models = [MODEL_NAME] + [BASELINE_LABELS[b] for b in BASELINES]
    col_w  = 13
    exp_w  = 30

    # One sub-table per metric for readability
    for metric in METRICS:
        header_label = METRIC_LABELS[metric]
        print(f"\n{'=' * (exp_w + col_w * len(models) + 3)}")
        print(f"  {header_label}")
        print(f"{'=' * (exp_w + col_w * len(models) + 3)}")

        # Header row
        header = f"{'Experiment':<{exp_w}}" + "".join(
            f"{m:>{col_w}}" for m in models
        )
        print(header)
        print("-" * len(header))

        for exp, model_results in table.items():
            row = f"{exp:<{exp_w}}"
            values = []
            for m in models:
                v = model_results.get(m, {}).get(metric)
                values.append(v)

            # Highlight best value (highest for all except loss, lowest for loss)
            numeric = [(i, v) for i, v in enumerate(values) if v is not None]
            if numeric:
                if metric == "loss":
                    best_idx = min(numeric, key=lambda x: x[1])[0]
                else:
                    best_idx = max(numeric, key=lambda x: x[1])[0]
            else:
                best_idx = -1

            for i, v in enumerate(values):
                cell = fmt(v)
                if i == best_idx and v is not None:
                    cell = f"*{cell}*"  # mark best
                row += f"{cell:>{col_w}}"

            print(row)


def write_csv(table: dict, out_path: str) -> None:
    models = [MODEL_NAME] + [BASELINE_LABELS[b] for b in BASELINES]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header
        header = ["Experiment", "Metric"] + models
        writer.writerow(header)

        for exp, model_results in table.items():
            for metric in METRICS:
                row = [exp, METRIC_LABELS[metric]]
                for m in models:
                    v = model_results.get(m, {}).get(metric)
                    row.append(fmt(v))
                writer.writerow(row)

    print(f"\nCSV saved to: {out_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare CascadeRouter vs baseline results."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save comparison as CSV (e.g. results/comparison.csv)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=RESULTS_DIR,
        help=f"Root results directory (default: {RESULTS_DIR})",
    )
    args = parser.parse_args()

    # Allow overriding RESULTS_DIR at runtime
    global RESULTS_DIR, BASELINES_DIR
    RESULTS_DIR   = args.results_dir
    BASELINES_DIR = os.path.join(RESULTS_DIR, "baselines")

    print("Loading results...")
    table = collect_all()
    print_table(table)

    if args.output:
        write_csv(table, args.output)


if __name__ == "__main__":
    main()
