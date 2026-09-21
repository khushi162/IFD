"""
plot_seed_sweep2.py
===================
Generate all graphs for the seed_sweep2 experiment suite.

Seeds: 45-55 (11 seeds), 22 experiments each.
Incomplete runs (Attack_SignFlip_40pct, Ablation_NoL2_ModelReplace20pct,
and a few others) are silently skipped per-seed using only seeds that have
>=50 rounds for that experiment.

Output figures (all saved to results/seed_sweep2/figures/):
  fig1_clean_baseline.png        — clean run: mean±std of all 5 metrics over 50 rounds
  fig2_attack_auc_heatmap.png    — final-round AUC heatmap: attack type × adversary %
  fig3_attack_f1_heatmap.png     — final-round F1 heatmap: attack type × adversary %
  fig4_signflip_auc_rounds.png   — AUC over rounds: clean vs SignFlip 10/20% (mean±std)
  fig5_labelflip_auc_rounds.png  — AUC over rounds: clean vs LabelFlip 10/20/40%
  fig6_modelreplace_auc_rounds.png — AUC over rounds: clean vs ModelReplace 10/20/40%
  fig7_ablation_auc_bar.png      — ablation final AUC: layer × condition grouped bar
  fig8_ablation_f1_bar.png       — ablation final F1: layer × condition grouped bar
  fig9_ablation_recall_bar.png   — ablation final Recall: layer × condition grouped bar
  fig10_seed_variance.png        — per-seed final AUC for clean + SignFlip20 + LabelFlip20
  fig11_precision_recall_scatter.png — Precision vs Recall scatter, all scenarios
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ─────────────────────────────────────────────────────────────────────
SWEEP_DIR  = os.path.dirname(os.path.abspath(__file__))
FIG_DIR    = os.path.join(SWEEP_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SEEDS = [f"seed_{s}" for s in range(45, 56)]   # seed_45 … seed_55
METRICS = ["auc", "f1", "precision", "recall", "accuracy"]
N_ROUNDS = 50

# ── Helpers ───────────────────────────────────────────────────────────────────

def load(seed: str, exp: str) -> dict | None:
    """Load a JSON for one seed/experiment. Returns None if missing or <50 rounds."""
    fp = os.path.join(SWEEP_DIR, seed, f"{exp}.json")
    if not os.path.exists(fp):
        return None
    with open(fp) as f:
        d = json.load(f)
    md = d.get("metrics_distributed", {})
    if len(md.get("auc", [])) < N_ROUNDS:
        return None
    return d


def final(d: dict, metric: str) -> float:
    """Last recorded value for a metric."""
    pairs = d["metrics_distributed"].get(metric, [])
    return pairs[-1][1] if pairs else float("nan")


def rounds_values(d: dict, metric: str):
    """Return (rounds_array, values_array) for a metric."""
    pairs = d["metrics_distributed"].get(metric, [])
    if not pairs:
        return np.array([]), np.array([])
    r, v = zip(*pairs)
    return np.array(r), np.array(v)


def collect_final(exp: str, metric: str) -> list[float]:
    """Collect final-round metric values across all seeds that completed this exp."""
    vals = []
    for seed in SEEDS:
        d = load(seed, exp)
        if d is not None:
            vals.append(final(d, metric))
    return vals


def collect_curves(exp: str, metric: str) -> np.ndarray | None:
    """
    Return shape (n_complete_seeds, N_ROUNDS) array of per-round values,
    or None if no seeds have this experiment complete.
    """
    rows = []
    for seed in SEEDS:
        d = load(seed, exp)
        if d is None:
            continue
        _, v = rounds_values(d, metric)
        if len(v) == N_ROUNDS:
            rows.append(v)
    return np.array(rows) if rows else None


def plot_mean_std(ax, curves: np.ndarray, label: str, color: str, linestyle="-"):
    """Plot mean curve with ±1 std shading."""
    if curves is None or len(curves) == 0:
        return
    mean = curves.mean(axis=0)
    std  = curves.std(axis=0)
    xs   = np.arange(1, N_ROUNDS + 1)
    ax.plot(xs, mean, color=color, label=label, linewidth=1.8, linestyle=linestyle)
    ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.15)


def savefig(fname: str):
    out = os.path.join(FIG_DIR, fname)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved -> {out}")


# ── Fig 1: Clean baseline — all 5 metrics over 50 rounds ──────────────────────
def fig1_clean_baseline():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Clean Baseline — Mean ± Std over {len(SEEDS)} Seeds",
        fontsize=13, fontweight="bold"
    )

    style = [
        ("auc",       "AUC",       "royalblue",  "o", ax1),
        ("f1",        "F1",        "darkorange",  "s", ax1),
        ("precision", "Precision", "purple",      "^", ax2),
        ("recall",    "Recall",    "crimson",     "v", ax2),
        ("accuracy",  "Accuracy",  "seagreen",    "D", ax2),
    ]
    for metric, label, color, marker, ax in style:
        curves = collect_curves("CleanRun_Simple", metric)
        if curves is None:
            continue
        mean = curves.mean(axis=0)
        std  = curves.std(axis=0)
        xs   = np.arange(1, N_ROUNDS + 1)
        ax.plot(xs, mean, color=color, label=label, linewidth=1.8)
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.15)

    for ax, title in [(ax1, "AUC & F1"), (ax2, "Precision / Recall / Accuracy")]:
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.set_xlim(1, N_ROUNDS)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    savefig("fig1_clean_baseline.png")


# ── Fig 2 & 3: Attack final-round heatmaps ────────────────────────────────────
def fig2_fig3_attack_heatmaps():
    attacks = ["SignFlip", "LabelFlip", "ModelReplace"]
    ratios  = ["10pct", "20pct", "40pct"]

    for metric, figname, title in [
        ("auc", "fig2_attack_auc_heatmap.png",
         "Final-Round AUC — Attack Robustness (Mean over Seeds)"),
        ("f1",  "fig3_attack_f1_heatmap.png",
         "Final-Round F1 — Attack Robustness (Mean over Seeds)"),
    ]:
        data   = np.full((len(attacks), len(ratios)), np.nan)
        counts = np.zeros((len(attacks), len(ratios)), dtype=int)

        for i, atk in enumerate(attacks):
            for j, pct in enumerate(ratios):
                exp = f"Attack_{atk}_{pct}"
                vals = collect_final(exp, metric)
                if vals:
                    data[i, j]   = np.mean(vals)
                    counts[i, j] = len(vals)

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.suptitle(title, fontsize=12, fontweight="bold")

        im = ax.imshow(data, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        plt.colorbar(im, ax=ax, label=metric.upper())

        ax.set_xticks(range(len(ratios)))
        ax.set_xticklabels(["10% adv", "20% adv", "40% adv"])
        ax.set_yticks(range(len(attacks)))
        ax.set_yticklabels(attacks)

        for i in range(len(attacks)):
            for j in range(len(ratios)):
                if not np.isnan(data[i, j]):
                    ax.text(j, i, f"{data[i,j]:.3f}\n(n={counts[i,j]})",
                            ha="center", va="center", fontsize=9,
                            color="black" if data[i, j] > 0.4 else "white")

        savefig(figname)


# ── Fig 4-6: AUC over rounds for each attack type ─────────────────────────────
def fig_attack_rounds(attack_label: str, attack_key: str, ratios: list, figname: str):
    colors = ["gold", "darkorange", "darkred"]
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        f"AUC over Rounds — {attack_label} Attack (Mean ± Std over Seeds)",
        fontsize=12, fontweight="bold"
    )

    clean_curves = collect_curves("CleanRun_Simple", "auc")
    plot_mean_std(ax, clean_curves, "Clean (0% adv)", "royalblue")

    for pct_str, color in zip(ratios, colors):
        exp = f"Attack_{attack_key}_{pct_str}"
        curves = collect_curves(exp, "auc")
        pct_label = pct_str.replace("pct", "%")
        plot_mean_std(ax, curves, f"{attack_label} {pct_label} adv", color, linestyle="--")

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("AUC")
    ax.set_xlim(1, N_ROUNDS)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    savefig(figname)


# ── Fig 7-9: Ablation grouped bar charts ──────────────────────────────────────
def fig_ablation_bar(metric: str, ylabel: str, figname: str, title: str):
    layers     = ["NoL1", "NoL2", "NoL3"]
    conditions = [
        ("Clean",              "Clean",           "royalblue"),
        ("SignFlip20pct",      "SignFlip 20%",     "darkorange"),
        ("LabelFlip20pct",     "LabelFlip 20%",    "seagreen"),
        ("ModelReplace20pct",  "ModelReplace 20%", "crimson"),
    ]
    full_model_vals = collect_final("CleanRun_Simple", metric)
    full_model_mean = np.mean(full_model_vals) if full_model_vals else np.nan

    x     = np.arange(len(layers))
    n_c   = len(conditions)
    width = 0.18
    offsets = np.linspace(-(n_c - 1) / 2, (n_c - 1) / 2, n_c) * width

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    for offset, (cond_key, cond_label, color) in zip(offsets, conditions):
        vals = []
        for layer in layers:
            exp  = f"Ablation_{layer}_{cond_key}"
            data = collect_final(exp, metric)
            vals.append(np.mean(data) if data else np.nan)

        bars = ax.bar(x + offset, vals, width=width, color=color,
                      edgecolor="black", linewidth=0.6, label=cond_label)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.008,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    # Full-model reference line
    if not np.isnan(full_model_mean):
        ax.axhline(full_model_mean, color="black", linewidth=1.2,
                   linestyle=":", label=f"Full model ({full_model_mean:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(["No Layer 1", "No Layer 2", "No Layer 3"], fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    savefig(figname)


# ── Fig 10: Per-seed variance for key experiments ─────────────────────────────
def fig10_seed_variance():
    scenarios = [
        ("CleanRun_Simple",     "Clean",          "royalblue"),
        ("Attack_SignFlip_20pct",  "SignFlip 20%",   "darkorange"),
        ("Attack_LabelFlip_20pct", "LabelFlip 20%",  "seagreen"),
        ("Attack_ModelReplace_20pct", "ModelReplace 20%", "crimson"),
    ]
    seed_nums = [int(s.split("_")[1]) for s in SEEDS]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(
        "Per-Seed Final-Round AUC — Key Scenarios",
        fontsize=13, fontweight="bold"
    )

    for exp, label, color in scenarios:
        vals = []
        for seed in SEEDS:
            d = load(seed, exp)
            vals.append(final(d, "auc") if d else np.nan)
        ax.plot(seed_nums, vals, marker="o", color=color,
                label=label, linewidth=1.5, markersize=6)

    ax.set_xlabel("Random Seed")
    ax.set_ylabel("Final-Round AUC")
    ax.set_xticks(seed_nums)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    savefig("fig10_seed_variance.png")


# ── Fig 11: Precision-Recall scatter across all scenarios ─────────────────────
def fig11_precision_recall_scatter():
    scenarios = [
        ("CleanRun_Simple",           "Clean",              "o", "royalblue",  100),
        ("Attack_SignFlip_10pct",     "SignFlip 10%",        "s", "gold",        70),
        ("Attack_SignFlip_20pct",     "SignFlip 20%",        "s", "darkorange",  70),
        ("Attack_LabelFlip_10pct",    "LabelFlip 10%",       "^", "lightgreen",  70),
        ("Attack_LabelFlip_20pct",    "LabelFlip 20%",       "^", "seagreen",    70),
        ("Attack_LabelFlip_40pct",    "LabelFlip 40%",       "^", "darkgreen",   70),
        ("Attack_ModelReplace_10pct", "ModelReplace 10%",    "D", "plum",        70),
        ("Attack_ModelReplace_20pct", "ModelReplace 20%",    "D", "orchid",      70),
        ("Attack_ModelReplace_40pct", "ModelReplace 40%",    "D", "purple",      70),
        ("Ablation_NoL1_Clean",       "No L1 (clean)",       "P", "steelblue",   70),
        ("Ablation_NoL2_Clean",       "No L2 (clean)",       "P", "cornflowerblue", 70),
        ("Ablation_NoL3_Clean",       "No L3 (clean)",       "P", "deepskyblue", 70),
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle(
        "Precision vs Recall — Final Round, Mean over Seeds",
        fontsize=13, fontweight="bold"
    )

    for exp, label, marker, color, size in scenarios:
        ps = collect_final(exp, "precision")
        rs = collect_final(exp, "recall")
        if not ps or not rs:
            continue
        ax.scatter(np.mean(rs), np.mean(ps),
                   marker=marker, color=color, s=size,
                   edgecolors="black", linewidths=0.6,
                   label=f"{label}  P={np.mean(ps):.3f} R={np.mean(rs):.3f}",
                   zorder=3)

    # Iso-F1 curves
    rec_grid = np.linspace(0.01, 1.0, 300)
    for f1_val in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        with np.errstate(invalid="ignore", divide="ignore"):
            p_iso = f1_val * rec_grid / (2 * rec_grid - f1_val)
        mask = (p_iso >= 0) & (p_iso <= 1)
        ax.plot(rec_grid[mask], p_iso[mask], color="lightgrey",
                linewidth=0.8, linestyle="--")
        idx = np.where(mask)[0]
        if len(idx):
            mid = idx[len(idx) // 2]
            ax.annotate(f"F1={f1_val:.1f}",
                        (rec_grid[mid], p_iso[mid]),
                        fontsize=7, color="grey", ha="center")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.2)
    savefig("fig11_precision_recall_scatter.png")


# ── Fig 12: Summary table — mean ± std for all experiments ────────────────────
def fig12_summary_table():
    """Text-table figure: mean±std of AUC/F1/Precision/Recall for all 22 experiments."""
    experiments = [
        "CleanRun_Simple",
        "Attack_SignFlip_10pct", "Attack_SignFlip_20pct", "Attack_SignFlip_40pct",
        "Attack_LabelFlip_10pct", "Attack_LabelFlip_20pct", "Attack_LabelFlip_40pct",
        "Attack_ModelReplace_10pct", "Attack_ModelReplace_20pct", "Attack_ModelReplace_40pct",
        "Ablation_NoL1_Clean", "Ablation_NoL1_SignFlip20pct",
        "Ablation_NoL1_LabelFlip20pct", "Ablation_NoL1_ModelReplace20pct",
        "Ablation_NoL2_Clean", "Ablation_NoL2_SignFlip20pct",
        "Ablation_NoL2_LabelFlip20pct", "Ablation_NoL2_ModelReplace20pct",
        "Ablation_NoL3_Clean", "Ablation_NoL3_SignFlip20pct",
        "Ablation_NoL3_LabelFlip20pct", "Ablation_NoL3_ModelReplace20pct",
    ]
    cols = ["AUC", "F1", "Prec", "Rec", "n"]
    col_keys = ["auc", "f1", "precision", "recall"]

    rows = []
    for exp in experiments:
        row = [exp]
        for mk in col_keys:
            vals = collect_final(exp, mk)
            if vals:
                row.append(f"{np.mean(vals):.3f}±{np.std(vals):.3f}")
            else:
                row.append("—")
        n = sum(1 for s in SEEDS if load(s, exp) is not None)
        row.append(str(n))
        rows.append(row)

    fig, ax = plt.subplots(figsize=(16, 10))
    fig.suptitle(
        f"Seed Sweep Summary — Mean ± Std (seeds 45–55, up to {len(SEEDS)} per exp)",
        fontsize=12, fontweight="bold"
    )
    ax.axis("off")

    col_headers = ["Experiment"] + cols
    table = ax.table(
        cellText=rows,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.4)

    # Colour header row
    for j in range(len(col_headers)):
        table[0, j].set_facecolor("#2c5f9e")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternating row shading
    for i in range(1, len(rows) + 1):
        for j in range(len(col_headers)):
            table[i, j].set_facecolor("#f0f4f8" if i % 2 == 0 else "white")

    savefig("fig12_summary_table.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== Generating seed_sweep2 figures ===\n")

    print("Fig 1: Clean baseline mean±std over rounds")
    fig1_clean_baseline()

    print("Fig 2 & 3: Attack AUC/F1 heatmaps")
    fig2_fig3_attack_heatmaps()

    print("Fig 4: SignFlip AUC over rounds")
    fig_attack_rounds("SignFlip", "SignFlip", ["10pct", "20pct"], "fig4_signflip_auc_rounds.png")

    print("Fig 5: LabelFlip AUC over rounds")
    fig_attack_rounds("LabelFlip", "LabelFlip", ["10pct", "20pct", "40pct"], "fig5_labelflip_auc_rounds.png")

    print("Fig 6: ModelReplace AUC over rounds")
    fig_attack_rounds("ModelReplace", "ModelReplace", ["10pct", "20pct", "40pct"], "fig6_modelreplace_auc_rounds.png")

    print("Fig 7: Ablation AUC bar")
    fig_ablation_bar("auc",       "AUC (final round, mean over seeds)",
                     "fig7_ablation_auc_bar.png",
                     "Ablation Study — Final AUC (Layer Removal × Attack Condition)")

    print("Fig 8: Ablation F1 bar")
    fig_ablation_bar("f1",        "F1 (final round, mean over seeds)",
                     "fig8_ablation_f1_bar.png",
                     "Ablation Study — Final F1 (Layer Removal × Attack Condition)")

    print("Fig 9: Ablation Recall bar")
    fig_ablation_bar("recall",    "Recall (final round, mean over seeds)",
                     "fig9_ablation_recall_bar.png",
                     "Ablation Study — Final Recall (Layer Removal × Attack Condition)")

    print("Fig 10: Per-seed AUC variance")
    fig10_seed_variance()

    print("Fig 11: Precision-Recall scatter")
    fig11_precision_recall_scatter()

    print("Fig 12: Summary table")
    fig12_summary_table()

    print(f"\nDone. All figures saved to {FIG_DIR}")
