"""
plot_results.py — Generate all meaningful graphs from simulation results.

Expected outputs in paper/:
  fig1_clean_metrics.png        — Clean baseline: all 5 metrics over rounds
  fig2_signflip_auc.png         — AUC vs rounds for SignFlip at 10/20/40%
  fig3_labelflip_auc.png        — AUC vs rounds for LabelFlip at 10/20/40%
  fig4_attack_final_auc_bar.png — Final-round AUC bar chart across all attacks
  fig5_attack_final_f1_bar.png  — Final-round F1 bar chart across all attacks
  fig6_ablation_auc.png         — Ablation: final AUC per layer removed (clean vs attacked)
  fig7_ablation_f1.png          — Ablation: final F1 per layer removed (clean vs attacked)
  fig8_precision_recall.png     — Precision vs Recall scatter across all scenarios
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "results"
PAPER_DIR   = "paper"
os.makedirs(PAPER_DIR, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def load(name: str):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        print(f"  [SKIP] {name} — file not found")
        return None
    with open(path) as f:
        d = json.load(f)
    metrics = d.get("metrics_distributed", {})
    if not metrics.get("auc"):
        print(f"  [SKIP] {name} — empty / incomplete run")
        return None
    return d


def extract(d, key):
    """Return (rounds, values) for a metric key."""
    pairs = d["metrics_distributed"].get(key, [])
    return [p[0] for p in pairs], [p[1] for p in pairs]


def final(d, key):
    """Last recorded value for a metric."""
    _, vals = extract(d, key)
    return vals[-1] if vals else float("nan")


def save(fname):
    out = os.path.join(PAPER_DIR, fname)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved -> {out}")


# ── fig 1: clean baseline — all 5 metrics ────────────────────────────────────

def fig1_clean_metrics():
    d = load("CleanRun_Simple.json")
    if not d:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Fig 1 — Clean Baseline (No Adversaries)", fontsize=14, fontweight="bold")

    # Left: AUC + F1
    ax = axes[0]
    for key, label, marker, color in [
        ("auc", "AUC",      "o", "royalblue"),
        ("f1",  "F1-Score", "s", "darkorange"),
    ]:
        r, v = extract(d, key)
        ax.plot(r, v, marker=marker, color=color, label=label, linewidth=1.8, markersize=3)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Score")
    ax.set_title("AUC & F1-Score")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Precision / Recall / Accuracy
    ax = axes[1]
    for key, label, marker, color in [
        ("accuracy",  "Accuracy",  "o", "seagreen"),
        ("precision", "Precision", "s", "purple"),
        ("recall",    "Recall",    "^", "crimson"),
    ]:
        r, v = extract(d, key)
        ax.plot(r, v, marker=marker, color=color, label=label, linewidth=1.8, markersize=3)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Score")
    ax.set_title("Precision, Recall & Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save("fig1_clean_metrics.png")


# ── fig 2: sign-flip AUC over rounds ─────────────────────────────────────────

def fig2_signflip_auc():
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Fig 2 — Robustness Under Sign-Flip Attack (AUC)", fontsize=14, fontweight="bold")

    d = load("CleanRun_Simple.json")
    if d:
        r, v = extract(d, "auc")
        ax.plot(r, v, label="Clean (0% adv)", color="royalblue", linewidth=2)

    colors = ["gold", "darkorange", "darkred"]
    for pct, color in zip([10, 20, 40], colors):
        d = load(f"Attack_SignFlip_{pct}pct.json")
        if not d:
            continue
        r, v = extract(d, "auc")
        ax.plot(r, v, label=f"SignFlip {pct}% adv", color=color, linewidth=1.8, linestyle="--")

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    save("fig2_signflip_auc.png")


# ── fig 3: label-flip AUC over rounds ────────────────────────────────────────

def fig3_labelflip_auc():
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Fig 3 — Robustness Under Label-Flip Attack (AUC)", fontsize=14, fontweight="bold")

    d = load("CleanRun_Simple.json")
    if d:
        r, v = extract(d, "auc")
        ax.plot(r, v, label="Clean (0% adv)", color="royalblue", linewidth=2)

    colors = ["gold", "darkorange", "darkred"]
    for pct, color in zip([10, 20, 40], colors):
        d = load(f"Attack_LabelFlip_{pct}pct.json")
        if not d:
            continue
        r, v = extract(d, "auc")
        ax.plot(r, v, label=f"LabelFlip {pct}% adv", color=color, linewidth=1.8, linestyle="--")

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    save("fig3_labelflip_auc.png")


# ── fig 4 & 5: final-round bar charts across all attack configs ───────────────

def fig4_fig5_attack_bar():
    configs = [
        ("CleanRun_Simple",        "Clean\n(0% adv)",   "royalblue"),
        ("Attack_SignFlip_10pct",  "SignFlip\n10%",     "gold"),
        ("Attack_SignFlip_20pct",  "SignFlip\n20%",     "darkorange"),
        ("Attack_SignFlip_40pct",  "SignFlip\n40%",     "darkred"),
        ("Attack_LabelFlip_10pct", "LabelFlip\n10%",   "lightgreen"),
        ("Attack_LabelFlip_20pct", "LabelFlip\n20%",   "seagreen"),
        ("Attack_LabelFlip_40pct", "LabelFlip\n40%",   "darkgreen"),
    ]

    for metric, ylabel, figname, title in [
        ("auc", "AUC (final round)",      "fig4_attack_final_auc_bar.png",
         "Fig 4 — Final-Round AUC: Clean vs All Attack Scenarios"),
        ("f1",  "F1-Score (final round)", "fig5_attack_final_f1_bar.png",
         "Fig 5 — Final-Round F1: Clean vs All Attack Scenarios"),
    ]:
        labels, values, colors = [], [], []
        for fname, label, color in configs:
            d = load(f"{fname}.json")
            if d:
                labels.append(label)
                values.append(final(d, metric))
                colors.append(color)

        fig, ax = plt.subplots(figsize=(11, 6))
        fig.suptitle(title, fontsize=13, fontweight="bold")
        bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6, width=0.55)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
        save(figname)


# ── fig 6 & 7: ablation — final metric per layer, clean vs attacked ───────────

def fig6_fig7_ablation():
    layer_labels = ["Full Model\n(baseline)", "No Layer 1\n(L1 off)",
                    "No Layer 2\n(L2 off)", "No Layer 3\n(L3 off)"]

    for metric, ylabel, figname, title in [
        ("auc", "AUC (final round)",      "fig6_ablation_auc.png",
         "Fig 6 — Ablation Study: Final AUC (Layer Removal)"),
        ("f1",  "F1-Score (final round)", "fig7_ablation_f1.png",
         "Fig 7 — Ablation Study: Final F1 (Layer Removal)"),
    ]:
        clean_vals, attacked_vals = [], []

        d_clean = load("CleanRun_Simple.json")
        d_atk   = load("Attack_SignFlip_20pct.json")
        clean_vals.append(final(d_clean, metric) if d_clean else float("nan"))
        attacked_vals.append(final(d_atk, metric)   if d_atk   else float("nan"))

        for layer in ["NoL1", "NoL2", "NoL3"]:
            dc = load(f"Ablation_{layer}_Clean.json")
            da = load(f"Ablation_{layer}_SignFlip20pct.json")
            clean_vals.append(final(dc, metric) if dc else float("nan"))
            attacked_vals.append(final(da, metric) if da else float("nan"))

        x = np.arange(len(layer_labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(title, fontsize=13, fontweight="bold")
        b1 = ax.bar(x - width / 2, clean_vals,    width, label="Clean",
                    color="royalblue",  edgecolor="black", linewidth=0.6)
        b2 = ax.bar(x + width / 2, attacked_vals, width, label="SignFlip 20% adv",
                    color="darkorange", edgecolor="black", linewidth=0.6)

        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                if not np.isnan(h):
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                            f"{h:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(layer_labels)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        save(figname)


# ── fig 8: precision–recall scatter across all major scenarios ────────────────

def fig8_precision_recall():
    scenarios = [
        ("CleanRun_Simple",        "Clean",          "o", "royalblue",   100),
        ("Attack_SignFlip_10pct",  "SignFlip 10%",   "s", "gold",         80),
        ("Attack_SignFlip_20pct",  "SignFlip 20%",   "s", "darkorange",   80),
        ("Attack_SignFlip_40pct",  "SignFlip 40%",   "s", "darkred",      80),
        ("Attack_LabelFlip_10pct", "LabelFlip 10%", "^", "lightgreen",   80),
        ("Attack_LabelFlip_20pct", "LabelFlip 20%", "^", "seagreen",     80),
        ("Attack_LabelFlip_40pct", "LabelFlip 40%", "^", "darkgreen",    80),
        ("Ablation_NoL1_Clean",    "No L1 (clean)", "D", "mediumpurple", 70),
        ("Ablation_NoL2_Clean",    "No L2 (clean)", "D", "orchid",       70),
        ("Ablation_NoL3_Clean",    "No L3 (clean)", "D", "plum",         70),
    ]

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.suptitle("Fig 8 — Precision vs Recall (Final Round, All Scenarios)",
                 fontsize=13, fontweight="bold")

    for fname, label, marker, color, size in scenarios:
        d = load(f"{fname}.json")
        if not d:
            continue
        p = final(d, "precision")
        r = final(d, "recall")
        ax.scatter(r, p, marker=marker, color=color, s=size, label=label,
                   edgecolors="black", linewidths=0.5, zorder=3)

    # iso-F1 curves
    recall_grid = np.linspace(0.01, 1.0, 300)
    for f1_val in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        precision_curve = f1_val * recall_grid / (2 * recall_grid - f1_val)
        mask = (precision_curve >= 0) & (precision_curve <= 1)
        ax.plot(recall_grid[mask], precision_curve[mask],
                color="lightgrey", linewidth=0.8, linestyle="--")
        idx = np.where(mask)[0]
        if len(idx):
            mid = idx[len(idx) // 2]
            ax.annotate(f"F1={f1_val:.1f}",
                        (recall_grid[mid], precision_curve[mid]),
                        fontsize=7, color="grey", ha="center")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)
    save("fig8_precision_recall.png")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Generating plots ===\n")

    print("Fig 1: Clean baseline metrics")
    fig1_clean_metrics()

    print("Fig 2: SignFlip AUC over rounds")
    fig2_signflip_auc()

    print("Fig 3: LabelFlip AUC over rounds")
    fig3_labelflip_auc()

    print("Fig 4 & 5: Attack final-round bar charts")
    fig4_fig5_attack_bar()

    print("Fig 6 & 7: Ablation bar charts")
    fig6_fig7_ablation()

    print("Fig 8: Precision-Recall scatter")
    fig8_precision_recall()

    print(f"\nDone! All figures saved to ./{PAPER_DIR}/")
