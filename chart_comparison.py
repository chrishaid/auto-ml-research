"""
Generate a comparison chart: Serial vs Island model evolution.

Usage: uv run chart_comparison.py results_serial.tsv results_island.tsv
"""

import re
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor": "#1e293b",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#e2e8f0",
    "axes.grid": True,
    "grid.color": "#334155",
    "grid.alpha": 0.5,
    "grid.linewidth": 0.5,
    "text.color": "#e2e8f0",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "font.family": "sans-serif",
    "font.size": 11,
    "legend.facecolor": "#1e293b",
    "legend.edgecolor": "#475569",
    "legend.fontsize": 10,
})

SERIAL_COLOR = "#f97316"   # orange
ISLAND_COLOR = "#06b6d4"   # cyan
SERIAL_FILL = "#f9731620"
ISLAND_FILL = "#06b6d420"
ACCENT = "#a78bfa"         # purple accent

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path):
    """Load results.tsv into a list of (elapsed_s, fitness, description) for successful evals."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["status"] == "ok":
                elapsed = float(row["elapsed_s"])
                fitness = float(row["fitness"])
                desc = row["description"]
                rows.append((elapsed, fitness, desc))
    return rows


def compute_cumulative_best(rows):
    """From rows, compute cumulative elapsed time, running best fitness, and best description."""
    if not rows:
        return [], [], ""
    cum_time = []
    best_fitness = []
    t = 0.0
    best = float("-inf")
    best_desc = ""
    for elapsed, fitness, desc in rows:
        t += elapsed
        if fitness > best:
            best = fitness
            best_desc = desc
        cum_time.append(t)
        best_fitness.append(best)
    return cum_time, best_fitness, best_desc


def fitness_to_rmse(fitness_vals):
    """Convert fitness (negated RMSE) back to RMSE."""
    return [-f for f in fitness_vals]


def parse_pipeline_config(desc):
    """Parse a pipeline description string into structured stages.

    Input:  '[I0] SimpleImputer_mean(strategy=mean) -> Winsorizer(lower_pct=5) -> LightGBM(n_estimators=220, ...)'
    Output: [('SimpleImputer_mean', {'strategy': 'mean'}), ('Winsorizer', {'lower_pct': '5'}), ...]
    """
    # Strip island prefix like '[I0] ' or '[I1] '
    desc = re.sub(r'^\[I\d+\]\s*', '', desc)

    stages = []
    for part in desc.split(" -> "):
        part = part.strip()
        match = re.match(r'^(\w+)(?:\((.+)\))?$', part)
        if match:
            name = match.group(1)
            params = {}
            if match.group(2):
                # Parse key=value pairs, handling nested parens like hidden_sizes=(128, 64)
                param_str = match.group(2)
                # Split on ', ' but not inside parens
                depth = 0
                current = ""
                pairs = []
                for ch in param_str:
                    if ch == '(':
                        depth += 1
                        current += ch
                    elif ch == ')':
                        depth -= 1
                        current += ch
                    elif ch == ',' and depth == 0:
                        pairs.append(current.strip())
                        current = ""
                    else:
                        current += ch
                if current.strip():
                    pairs.append(current.strip())
                for pair in pairs:
                    if '=' in pair:
                        k, v = pair.split('=', 1)
                        params[k.strip()] = v.strip()
            stages.append((name, params))
        else:
            stages.append((part, {}))
    return stages


def format_config_lines(stages):
    """Format parsed stages into display lines for the chart."""
    lines = []
    for name, params in stages:
        # Categorize the stage
        algo_names = {"RandomForest", "GradientBoosting", "ExtraTrees", "XGBoost",
                      "LightGBM", "Ridge", "Lasso", "ElasticNet", "SVR", "SVC",
                      "KNeighbors", "DecisionTree", "AdaBoost", "MLP",
                      "LogisticRegression"}
        preproc_names = {"StandardScaler", "MinMaxScaler", "RobustScaler", "PCA",
                         "PolynomialFeatures"}
        fs_names = {"SelectKBest", "passthrough"}
        prep_names = {"SimpleImputer_mean", "SimpleImputer_median",
                      "SimpleImputer_most_frequent", "KNNImputer",
                      "OutlierClipper", "Winsorizer"}

        if name in algo_names:
            tag = "ALGO"
        elif name in fs_names:
            tag = "FEAT"
        elif name in preproc_names:
            tag = "PREP"
        elif name in prep_names:
            tag = "DATA"
        else:
            tag = "    "

        lines.append((tag, name, params))
    return lines


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def draw_config_panel(ax, stages, color, title):
    """Draw a pipeline config panel with stage cards."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="600", color=color, pad=10)

    lines = format_config_lines(stages)

    tag_colors = {
        "DATA": "#22c55e",
        "PREP": "#3b82f6",
        "FEAT": "#a855f7",
        "ALGO": "#ef4444",
        "    ": "#64748b",
    }

    y = 0.94
    stage_height = 0.06
    param_height = 0.045

    for tag, name, params in lines:
        tag_col = tag_colors.get(tag, "#64748b")

        # Tag badge
        ax.text(0.0, y, f" {tag} ", fontsize=7, fontweight="bold", color="#0f172a",
                family="monospace", verticalalignment="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=tag_col, alpha=0.85,
                          edgecolor="none"))

        # Stage name
        ax.text(0.15, y, name, fontsize=10.5, fontweight="600", color="#f1f5f9",
                verticalalignment="center")
        y -= stage_height

        # Params
        for k, v in params.items():
            # Truncate long float values
            try:
                fv = float(v)
                if fv != int(fv):
                    v = f"{fv:.4g}"
            except (ValueError, OverflowError):
                pass
            ax.text(0.17, y, f"{k}", fontsize=8.5, color="#94a3b8",
                    verticalalignment="center", family="monospace")
            ax.text(0.58, y, f"{v}", fontsize=8.5, color="#cbd5e1",
                    verticalalignment="center", family="monospace")
            y -= param_height

        # Separator line
        y -= 0.01
        if y > 0.05:
            ax.axhline(y=y + 0.005, xmin=0.02, xmax=0.98,
                        color="#334155", linewidth=0.4, alpha=0.6)
        y -= 0.015


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} results_serial.tsv results_island.tsv")
        sys.exit(1)

    serial_path, island_path = sys.argv[1], sys.argv[2]

    serial_rows = load_results(serial_path)
    island_rows = load_results(island_path)

    serial_time, serial_best, serial_desc = compute_cumulative_best(serial_rows)
    island_time, island_best, island_desc = compute_cumulative_best(island_rows)

    serial_rmse = fitness_to_rmse(serial_best)
    island_rmse = fitness_to_rmse(island_best)

    serial_stages = parse_pipeline_config(serial_desc)
    island_stages = parse_pipeline_config(island_desc)

    # --- Stats ---
    serial_final = serial_rmse[-1] if serial_rmse else float("inf")
    island_final = island_rmse[-1] if island_rmse else float("inf")
    serial_evals = len(serial_rows)
    island_evals = len(island_rows)
    improvement_pct = (serial_final - island_final) / serial_final * 100 if serial_final > 0 else 0

    print(f"Serial: {serial_evals} evals, final RMSE = {serial_final:.4f}")
    print(f"Island: {island_evals} evals, final RMSE = {island_final:.4f}")
    print(f"Improvement: {improvement_pct:+.1f}%")
    print(f"Serial best: {serial_desc}")
    print(f"Island best: {island_desc}")

    # --- Figure: 2 rows ---
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.28, wspace=0.25,
                          left=0.05, right=0.97, top=0.93, bottom=0.04)

    fig.suptitle("Island Model vs Serial Evolution", fontsize=20, fontweight="bold",
                 color="#f8fafc", y=0.97)

    # -- Top left: Convergence over time --
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(serial_time, serial_rmse, color=SERIAL_COLOR, linewidth=2.2,
             label=f"Serial (1 island)", zorder=3)
    ax1.fill_between(serial_time, serial_rmse, serial_rmse[-1] if serial_rmse else 0,
                     color=SERIAL_FILL, zorder=1)

    ax1.plot(island_time, island_rmse, color=ISLAND_COLOR, linewidth=2.2,
             label=f"Island model (2 islands)", zorder=3)
    ax1.fill_between(island_time, island_rmse, island_rmse[-1] if island_rmse else 0,
                     color=ISLAND_FILL, zorder=1)

    # Best markers
    if serial_rmse:
        best_serial_idx = serial_rmse.index(min(serial_rmse))
        ax1.scatter([serial_time[best_serial_idx]], [serial_rmse[best_serial_idx]],
                    color=SERIAL_COLOR, s=80, zorder=5, edgecolors="#0f172a", linewidth=1.5)
    if island_rmse:
        best_island_idx = island_rmse.index(min(island_rmse))
        ax1.scatter([island_time[best_island_idx]], [island_rmse[best_island_idx]],
                    color=ISLAND_COLOR, s=80, zorder=5, edgecolors="#0f172a", linewidth=1.5)

    ax1.set_xlabel("Cumulative Compute Time (s)", fontsize=12, fontweight="500")
    ax1.set_ylabel("Best RMSE (lower is better)", fontsize=12, fontweight="500")
    ax1.set_title("Convergence Over Time", fontsize=13, fontweight="600", pad=12)
    ax1.legend(loc="upper right", framealpha=0.9)

    # Annotate final values
    if serial_rmse:
        ax1.annotate(f"{serial_final:.4f}", xy=(serial_time[-1], serial_rmse[-1]),
                     xytext=(-60, 15), textcoords="offset points",
                     color=SERIAL_COLOR, fontsize=11, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=SERIAL_COLOR, lw=1.2))
    if island_rmse:
        ax1.annotate(f"{island_final:.4f}", xy=(island_time[-1], island_rmse[-1]),
                     xytext=(-60, -20), textcoords="offset points",
                     color=ISLAND_COLOR, fontsize=11, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=ISLAND_COLOR, lw=1.2))

    # -- Top right: Summary stats --
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.set_title("Summary", fontsize=13, fontweight="600", pad=12)

    stats_text = [
        ("", "Serial", "Island"),
        ("Final RMSE", f"{serial_final:.4f}", f"{island_final:.4f}"),
        ("Evals", f"{serial_evals}", f"{island_evals}"),
        ("Wall time", f"{serial_time[-1]:.0f}s" if serial_time else "\u2014",
                      f"{island_time[-1]:.0f}s" if island_time else "\u2014"),
    ]

    y_start = 0.85
    row_height = 0.11
    for i, (label, v1, v2) in enumerate(stats_text):
        y = y_start - i * row_height
        weight = "bold" if i == 0 else "normal"
        size = 11 if i == 0 else 12
        ax2.text(0.05, y, label, fontsize=size, fontweight=weight, color="#94a3b8",
                 verticalalignment="center")
        ax2.text(0.55, y, v1, fontsize=size, fontweight=weight, color=SERIAL_COLOR,
                 verticalalignment="center", horizontalalignment="center")
        ax2.text(0.85, y, v2, fontsize=size, fontweight=weight, color=ISLAND_COLOR,
                 verticalalignment="center", horizontalalignment="center")
        if i > 0:
            ax2.axhline(y=y + row_height * 0.5, xmin=0.02, xmax=0.98,
                        color="#334155", linewidth=0.5)

    # Improvement callout
    if improvement_pct != 0:
        sign = "+" if improvement_pct > 0 else ""
        color = "#22c55e" if improvement_pct > 0 else "#ef4444"
        ax2.text(0.5, 0.35, f"{sign}{improvement_pct:.1f}%",
                 fontsize=32, fontweight="bold", color=color,
                 ha="center", va="center")
        ax2.text(0.5, 0.24, "RMSE improvement" if improvement_pct > 0 else "RMSE regression",
                 fontsize=10, color="#94a3b8", ha="center", va="center")

    # Legend for tags
    ax2.text(0.5, 0.10, "Stage tags:", fontsize=8, color="#64748b",
             ha="center", va="center")
    tag_legend = [("DATA", "#22c55e"), ("PREP", "#3b82f6"),
                  ("FEAT", "#a855f7"), ("ALGO", "#ef4444")]
    for j, (tag, tc) in enumerate(tag_legend):
        x = 0.1 + j * 0.23
        ax2.text(x, 0.04, f" {tag} ", fontsize=7, fontweight="bold", color="#0f172a",
                 family="monospace", va="center",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor=tc, alpha=0.85,
                           edgecolor="none"))

    # -- Bottom left: Serial best config --
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#1e293b")
    draw_config_panel(ax3, serial_stages, SERIAL_COLOR, f"Serial Best  (RMSE {serial_final:.4f})")

    # -- Bottom middle: Island best config --
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#1e293b")
    draw_config_panel(ax4, island_stages, ISLAND_COLOR, f"Island Best  (RMSE {island_final:.4f})")

    # -- Bottom right: Key differences --
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.axis("off")
    ax5.set_title("Key Differences", fontsize=13, fontweight="600", pad=12, color=ACCENT)

    # Extract algorithm names
    serial_algo = next((name for name, _ in serial_stages
                        if name in {"RandomForest", "GradientBoosting", "ExtraTrees",
                                    "XGBoost", "LightGBM", "Ridge", "Lasso", "ElasticNet",
                                    "SVR", "KNeighbors", "DecisionTree", "AdaBoost", "MLP"}),
                       "Unknown")
    island_algo = next((name for name, _ in island_stages
                        if name in {"RandomForest", "GradientBoosting", "ExtraTrees",
                                    "XGBoost", "LightGBM", "Ridge", "Lasso", "ElasticNet",
                                    "SVR", "KNeighbors", "DecisionTree", "AdaBoost", "MLP"}),
                       "Unknown")

    diffs = [
        ("Algorithm", serial_algo, island_algo),
        ("Data steps", str(sum(1 for n, _ in serial_stages if n in
            {"SimpleImputer_mean", "SimpleImputer_median", "SimpleImputer_most_frequent",
             "KNNImputer", "OutlierClipper", "Winsorizer"})),
            str(sum(1 for n, _ in island_stages if n in
            {"SimpleImputer_mean", "SimpleImputer_median", "SimpleImputer_most_frequent",
             "KNNImputer", "OutlierClipper", "Winsorizer"}))),
        ("Preprocessing", str(sum(1 for n, _ in serial_stages if n in
            {"StandardScaler", "MinMaxScaler", "RobustScaler", "PCA", "PolynomialFeatures"})),
            str(sum(1 for n, _ in island_stages if n in
            {"StandardScaler", "MinMaxScaler", "RobustScaler", "PCA", "PolynomialFeatures"}))),
        ("Feature sel.", "yes" if any(n == "SelectKBest" for n, _ in serial_stages) else "no",
                         "yes" if any(n == "SelectKBest" for n, _ in island_stages) else "no"),
        ("Total stages", str(len(serial_stages)), str(len(island_stages))),
    ]

    y = 0.85
    for label, v1, v2 in diffs:
        ax5.text(0.05, y, label, fontsize=10, color="#94a3b8", va="center")
        ax5.text(0.50, y, v1, fontsize=10, color=SERIAL_COLOR, va="center",
                 ha="center", fontweight="500")
        ax5.text(0.80, y, v2, fontsize=10, color=ISLAND_COLOR, va="center",
                 ha="center", fontweight="500")
        y -= 0.1
        ax5.axhline(y=y + 0.05, xmin=0.02, xmax=0.98, color="#334155",
                     linewidth=0.4, alpha=0.5)

    # Insight text
    ax5.text(0.5, 0.18,
             "Island diversity led to exploring\ndifferent algorithm families",
             fontsize=10, color="#94a3b8", ha="center", va="center",
             style="italic", linespacing=1.6)

    # Footer
    fig.text(0.5, 0.01, "California Housing  \u00b7  120s budget  \u00b7  RMSE (minimize)  \u00b7  autoresearch",
             ha="center", fontsize=9, color="#64748b", style="italic")

    out_path = "docs/serial_vs_island.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
