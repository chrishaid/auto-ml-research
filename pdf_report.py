#!/usr/bin/env python3
"""
Stakeholder PDF Report — AutoML Propensity Models
===================================================

Generates a polished PDF report for R&A Team, Data Engineering, and CTO.

Usage:
    uv run pdf_report.py

Requires: fpdf2, matplotlib, numpy, shap (all already installed).
"""

import os
import sys
import csv
import re
import warnings
import textwrap
from collections import defaultdict, Counter
from io import BytesIO
import base64

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

from fpdf import FPDF

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Import report.py functions for data collection
# ---------------------------------------------------------------------------

from report import (
    load_results, detect_runs, pick_run_for_problem, find_best_row,
    parse_description, extract_algorithm_name, extract_block_signature,
    compute_feature_importances, compute_shap_data, compute_evolution_data,
    rebuild_config_from_description,
)

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

BLUE    = (41, 98, 255)
NAVY    = (25, 42, 86)
TEAL    = (0, 150, 136)
ORANGE  = (255, 152, 0)
RED     = (244, 67, 54)
GREEN   = (76, 175, 80)
GRAY    = (120, 120, 120)
LIGHT   = (240, 244, 248)
WHITE   = (255, 255, 255)
BLACK   = (30, 30, 30)
ACCENT  = (63, 81, 181)

PROBLEM_COLORS = {
    "email-propensity": BLUE,
    "event-propensity": TEAL,
    "web-propensity":   ORANGE,
}

PROBLEM_ORDER = ["email-propensity", "event-propensity", "web-propensity"]
PROBLEM_LABELS = {
    "email-propensity": "Email Propensity",
    "event-propensity": "Event Propensity",
    "web-propensity":   "Web Propensity",
}

# ---------------------------------------------------------------------------
# PDF subclass with helpers
# ---------------------------------------------------------------------------

class StakeholderPDF(FPDF):
    """Custom PDF with headers, footers, and styling helpers."""

    def __init__(self):
        super().__init__(orientation="P", unit="mm", format="letter")
        self.set_auto_page_break(auto=True, margin=25)
        self._load_fonts()
        self.section_num = 0
        self.figure_num = 0

    def _load_fonts(self):
        """Register Unicode-capable fonts."""
        # Use Arial Unicode for full Unicode support (em dashes, etc.)
        font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
        bold_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        italic_path = "/System/Library/Fonts/Supplemental/Arial Italic.ttf"
        bold_italic_path = "/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf"
        self.add_font("Arial", "", font_path, uni=True)
        self.add_font("Arial", "B", bold_path, uni=True)
        self.add_font("Arial", "I", italic_path, uni=True)
        self.add_font("Arial", "BI", bold_italic_path, uni=True)

    # -- Header / Footer ---------------------------------------------------

    def header(self):
        if self.page_no() == 1:
            return  # cover page handled separately
        self.set_font("Arial", "I", 8)
        self.set_text_color(*GRAY)
        self.cell(0, 5, "AutoML Propensity Models — Stakeholder Report", align="L")
        self.cell(0, 5, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 7)
        self.set_text_color(*GRAY)
        self.cell(0, 5, "Confidential — R&A Analytics", align="C")

    # -- Layout helpers -----------------------------------------------------

    def section_title(self, title, level=1):
        """Print a section heading."""
        if level == 1:
            self.section_num += 1
            self.set_font("Arial", "B", 16)
            self.set_text_color(*NAVY)
            self.ln(4)
            self.cell(0, 10, f"{self.section_num}. {title}", new_x="LMARGIN", new_y="NEXT")
            y = self.get_y()
            self.set_draw_color(*BLUE)
            self.set_line_width(0.8)
            self.line(self.l_margin, y, self.l_margin + 80, y)
            self.ln(4)
        elif level == 2:
            self.set_font("Arial", "B", 13)
            self.set_text_color(*ACCENT)
            self.ln(2)
            self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
            self.ln(2)
        elif level == 3:
            self.set_font("Arial", "B", 11)
            self.set_text_color(*BLACK)
            self.ln(1)
            self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
            self.ln(1)

    def body_text(self, text, indent=0):
        """Print body paragraph text."""
        self.set_font("Arial", "", 10)
        self.set_text_color(*BLACK)
        x = self.l_margin + indent
        w = self.w - self.l_margin - self.r_margin - indent
        self.set_x(x)
        self.multi_cell(w, 5, text)
        self.ln(2)

    def callout_box(self, title, text, color=BLUE):
        """Render a colored callout/aside box."""
        self.ln(2)
        x = self.l_margin + 5
        w = self.w - self.l_margin - self.r_margin - 10
        # Check if we need a page break
        needed = 20 + len(text) // 80 * 5
        if self.get_y() + needed > self.h - 30:
            self.add_page()

        y_start = self.get_y()
        # Background
        self.set_fill_color(color[0], color[1], color[2])
        self.rect(x, y_start, 3, 0.1)  # placeholder, we'll extend after

        # Title
        self.set_x(x + 6)
        self.set_font("Arial", "B", 9)
        self.set_text_color(*color)
        self.cell(w - 6, 5, title, new_x="LMARGIN", new_y="NEXT")

        # Body
        self.set_x(x + 6)
        self.set_font("Arial", "", 9)
        self.set_text_color(80, 80, 80)
        self.multi_cell(w - 6, 4.5, text)
        self.ln(1)

        y_end = self.get_y()
        # Draw left border bar
        self.set_fill_color(color[0], color[1], color[2])
        self.rect(x, y_start, 2.5, y_end - y_start, "F")
        # Light background
        self.set_fill_color(color[0] // 4 + 191, color[1] // 4 + 191, color[2] // 4 + 191)
        self.rect(x + 2.5, y_start, w - 2.5, y_end - y_start, "F")
        # Re-draw text on top of background by repositioning — fpdf2 doesn't support z-order
        # So we draw background first, then overlay text
        # Actually, we need to draw background BEFORE text. Let's restructure:
        pass  # We'll use a simpler approach below

    def aside_box(self, title, text, color=BLUE):
        """Aside box with proper height measurement — render text, measure, draw bg underneath."""
        self.ln(2)
        x0 = self.l_margin + 4
        w = self.w - self.l_margin - self.r_margin - 8
        content_w = w - 8

        # Measure actual height by doing a dry-run render
        self.set_font("Arial", "B", 9)
        title_h = 5
        self.set_font("Arial", "", 9)
        # Count lines the body text will need
        body_lines = 1
        for line in text.split("\n"):
            line_w = self.get_string_width(line)
            body_lines += max(1, int(line_w / content_w) + 1)
        body_h = body_lines * 4.5
        h_box = title_h + body_h + 8  # padding top + bottom

        if self.get_y() + h_box > self.h - 30:
            self.add_page()

        y_start = self.get_y()

        # Draw background and accent bar
        r, g, b = color
        self.set_fill_color(r // 5 + 204, g // 5 + 204, b // 5 + 204)
        self.rect(x0, y_start, w, h_box, "F")
        self.set_fill_color(*color)
        self.rect(x0, y_start, 2.5, h_box, "F")

        # Title
        self.set_xy(x0 + 6, y_start + 3)
        self.set_font("Arial", "B", 9)
        self.set_text_color(*color)
        self.cell(content_w, 5, title)

        # Body
        self.set_xy(x0 + 6, y_start + 9)
        self.set_font("Arial", "", 9)
        self.set_text_color(60, 60, 60)
        self.multi_cell(content_w, 4.5, text)

        actual_end = self.get_y() + 3
        # If text ran past our estimate, extend the background
        if actual_end > y_start + h_box:
            h_box = actual_end - y_start
            # Redraw background (overdraw is fine in PDF)
            self.set_fill_color(r // 5 + 204, g // 5 + 204, b // 5 + 204)
            self.rect(x0, y_start, w, h_box, "F")
            self.set_fill_color(*color)
            self.rect(x0, y_start, 2.5, h_box, "F")
            # Re-render text on top
            self.set_xy(x0 + 6, y_start + 3)
            self.set_font("Arial", "B", 9)
            self.set_text_color(*color)
            self.cell(content_w, 5, title)
            self.set_xy(x0 + 6, y_start + 9)
            self.set_font("Arial", "", 9)
            self.set_text_color(60, 60, 60)
            self.multi_cell(content_w, 4.5, text)

        self.set_y(max(y_start + h_box, self.get_y()) + 2)

    def add_figure(self, fig, caption="", width=170):
        """Save matplotlib figure to temp file and embed in PDF."""
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=180, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
        buf.seek(0)

        # Write to temp file (fpdf2 needs a file path)
        tmp_path = f"/tmp/automl_fig_{id(fig)}.png"
        with open(tmp_path, "wb") as f:
            f.write(buf.read())

        # Check if we need page break
        if self.get_y() + 90 > self.h - 30:
            self.add_page()

        x = (self.w - width) / 2
        self.image(tmp_path, x=x, w=width)
        os.remove(tmp_path)

        if caption:
            self.figure_num += 1
            self.set_font("Arial", "I", 8)
            self.set_text_color(*GRAY)
            # Use multi_cell for wrapping, centered within margins
            caption_text = f"Figure {self.figure_num}: {caption}"
            cap_w = self.w - self.l_margin - self.r_margin - 10
            cap_x = self.l_margin + 5
            self.set_x(cap_x)
            self.multi_cell(cap_w, 4, caption_text, align="C")
        self.ln(3)

    def key_metric_row(self, items):
        """Render a row of key metric boxes. items = [(label, value, color), ...]"""
        n = len(items)
        box_w = (self.w - self.l_margin - self.r_margin - (n - 1) * 4) / n
        y = self.get_y()

        if y + 22 > self.h - 30:
            self.add_page()
            y = self.get_y()

        for i, (label, value, color) in enumerate(items):
            x = self.l_margin + i * (box_w + 4)
            # Box background
            self.set_fill_color(color[0] // 5 + 204, color[1] // 5 + 204, color[2] // 5 + 204)
            self.rect(x, y, box_w, 20, "F")
            # Top accent
            self.set_fill_color(*color)
            self.rect(x, y, box_w, 2, "F")
            # Value
            self.set_xy(x, y + 3)
            self.set_font("Arial", "B", 14)
            self.set_text_color(*color)
            self.cell(box_w, 8, str(value), align="C")
            # Label
            self.set_xy(x, y + 11)
            self.set_font("Arial", "", 8)
            self.set_text_color(*GRAY)
            self.cell(box_w, 5, label, align="C")

        self.set_y(y + 24)


# ---------------------------------------------------------------------------
# Chart generation functions
# ---------------------------------------------------------------------------

def fig_score_progression(all_data):
    """Create faceted score progression charts — one per problem, free y-axis scales."""
    n = len(all_data)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.2), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, d in zip(axes, all_data):
        name = d["name"]
        evo = d["evo_data"]
        gens = sorted(evo["gen_best"].keys())
        scores = [evo["gen_best"][g] for g in gens]
        color = tuple(c / 255 for c in PROBLEM_COLORS.get(name, BLUE))

        ax.plot(gens, scores, color=color, linewidth=2)
        ax.fill_between(gens, scores, alpha=0.08, color=color)

        # Mark final best
        if gens and scores:
            ax.scatter([gens[-1]], [scores[-1]], color=color, s=50, zorder=5)
            ax.annotate(f"{scores[-1]:.4f}", (gens[-1], scores[-1]),
                       textcoords="offset points", xytext=(-5, 8),
                       fontsize=8, color=color, fontweight="bold")

        # Free y-axis: zoom to the range of this problem's scores
        score_min = min(s for s in scores if s > 0) if scores else 0.8
        score_max = max(scores) if scores else 1.0
        margin = (score_max - score_min) * 0.15
        ax.set_ylim(score_min - margin, score_max + margin * 2)

        ax.set_xlabel("Generation", fontsize=9)
        ax.set_ylabel("AUC", fontsize=9)
        ax.set_title(PROBLEM_LABELS.get(name, name), fontsize=10, fontweight="bold", color=color)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(labelsize=7)

    fig.suptitle("AUC Convergence by Problem", fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    return fig


def fig_feature_importance(feature_names, importances, title, color=BLUE):
    """Horizontal bar chart of top 10 feature importances."""
    idx = np.argsort(importances)[::-1][:10]
    names = [feature_names[i] for i in idx]
    vals = importances[idx]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    c = tuple(v / 255 for v in color)
    bars = ax.barh(range(len(names) - 1, -1, -1), vals, color=c, alpha=0.85, height=0.65)
    ax.set_yticks(range(len(names) - 1, -1, -1))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Importance", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    for i, (bar, v) in enumerate(zip(bars, vals)):
        ax.text(v + max(vals) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=7, color=c)

    # Add right margin so bar labels don't clip outside canvas
    ax.set_xlim(right=max(vals) * 1.18)
    fig.tight_layout()
    return fig


def _abbreviate_feature_names(names, max_len=28):
    """Shorten long feature names for chart readability."""
    ABBREVS = {
        "TOTAL_PAGEVIEWS": "PGVIEWS",
        "TOTAL_AD_IMPRESSIONS": "AD_IMP",
        "TOTAL_AD_CLICKS": "AD_CLK",
        "TOTAL_FORM_SUBMITS": "FORM_SUB",
        "TOTAL_EVENT_REGISTERS": "EVT_REG",
        "TOTAL_EVENT_ATTENDS": "EVT_ATT",
        "TOTAL_EDU_REGISTERS": "EDU_REG",
        "TOTAL_EDU_COMPLETES": "EDU_CMP",
        "TOTAL_SURVEY_RESPONSES": "SURVEY",
        "ACTIVE_MONTHS": "ACT_MO",
        "BRANDS_ENGAGED": "BRANDS",
        "RECENT_3M_EVENTS": "REC_3M",
        "MID_3M_EVENTS": "MID_3M",
        "EARLY_6M_EVENTS": "EARLY_6M",
        "MONTHS_SINCE_LAST_WEB": "MO_SINCE_WEB",
        "MONTHS_SINCE_LAST_EMAIL": "MO_SINCE_EML",
        "MONTHS_SINCE_LAST_EVENT": "MO_SINCE_EVT",
        "NEW_BRAND_ENGAGEMENTS": "NEW_BRAND",
        "EMAIL_EVENTS_12M": "EML_12M",
        "WEB_EVENTS_12M": "WEB_12M",
        "EVENT_EVENTS_12M": "EVT_12M",
        "AD_EVENTS_12M": "AD_12M",
        "SPECIALTY_PRIMARY_GROUP": "SPECIALTY",
        "GRADUATION_YEAR": "GRAD_YR",
        "PERSON_TYPE": "PERS_TYPE",
        "IS_HOSPITALIST": "HOSP",
        "IS_EXECUTIVE": "EXEC",
        "IS_KEY_OPINION_LEADER": "KOL",
    }
    short = []
    for n in names:
        s = n
        for long, abbr in ABBREVS.items():
            s = s.replace(long, abbr)
        # Truncate if still too long
        if len(s) > max_len:
            s = s[:max_len - 2] + ".."
        short.append(s)
    return short


def fig_shap_beeswarm(shap_values, X_sample, feature_names, title, color=BLUE):
    """SHAP beeswarm-style summary plot."""
    import shap

    short_names = _abbreviate_feature_names(feature_names)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    shap.summary_plot(shap_values, X_sample, feature_names=short_names,
                      max_display=10, show=False, plot_size=None)
    plt.title(title, fontsize=11, fontweight="bold")
    fig = plt.gcf()
    fig.set_size_inches(7.5, 4)
    fig.tight_layout()
    return fig


def fig_algorithm_diversity(all_data):
    """Stacked bar chart showing algorithm diversity across generations for each problem."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    for ax, d in zip(axes, all_data):
        name = d["name"]
        evo = d["evo_data"]
        gen_algos = evo["gen_algorithms"]
        gens = sorted(gen_algos.keys())

        # Collect all algorithms
        all_algos = set()
        for g in gens:
            all_algos.update(gen_algos[g].keys())

        algo_colors = {
            "XGBoost": "#1565C0", "ExtraTrees": "#2E7D32", "RandomForest": "#4CAF50",
            "GradientBoosting": "#FF8F00", "LightGBM": "#7B1FA2", "AdaBoost": "#D84315",
            "DecisionTree": "#795548", "KNeighbors": "#00ACC1", "SVR": "#E91E63",
            "MLP": "#5C6BC0", "Ridge": "#8D6E63", "Lasso": "#78909C",
        }

        # Downsample generations
        if len(gens) > 40:
            step = len(gens) / 40
            gens = [gens[int(i * step)] for i in range(40)]

        bottom = np.zeros(len(gens))
        for algo in sorted(all_algos):
            counts = [gen_algos[g].get(algo, 0) for g in gens]
            c = algo_colors.get(algo, "#9E9E9E")
            ax.bar(range(len(gens)), counts, bottom=bottom, label=algo,
                   color=c, width=1.0, alpha=0.85)
            bottom += np.array(counts)

        ax.set_title(PROBLEM_LABELS.get(name, name), fontsize=9, fontweight="bold")
        ax.set_xlabel("Gen", fontsize=7)
        if ax == axes[0]:
            ax.set_ylabel("Evaluations", fontsize=7)
        ax.tick_params(labelsize=6)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=6,
              bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Algorithm Diversity Across Generations", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def fig_evolution_diagram():
    """Create a diagram showing how the genetic algorithm works."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.set_xlim(-0.2, 10.2)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Boxes
    boxes = [
        (0.5, 4.5, "Initial\nPopulation", BLUE),
        (2.5, 4.5, "Tournament\nSelection", TEAL),
        (4.5, 4.5, "Block-Swap\nCrossover", ORANGE),
        (6.5, 4.5, "Mutation", RED),
        (8.5, 4.5, "Evaluate\n& Replace", GREEN),
    ]

    for x, y, text, color in boxes:
        c = tuple(v / 255 for v in color)
        rect = mpatches.FancyBboxPatch((x, y - 0.6), 1.6, 1.2,
                                        boxstyle="round,pad=0.1",
                                        facecolor=c, alpha=0.2,
                                        edgecolor=c, linewidth=2)
        ax.add_patch(rect)
        ax.text(x + 0.8, y, text, ha="center", va="center", fontsize=8,
                fontweight="bold", color=c)

    # Arrows
    gray_n = tuple(v / 255 for v in GRAY)
    for i in range(4):
        x_start = boxes[i][0] + 1.6
        x_end = boxes[i + 1][0]
        y = boxes[i][1]
        ax.annotate("", xy=(x_end, y), xytext=(x_start, y),
                    arrowprops=dict(arrowstyle="->", color=gray_n, lw=1.5))

    # Loop back arrow — labeled so the arc is clearly a "repeat" loop
    ax.annotate("Repeat (100-200+ generations)",
                xy=(0.8, 3.7), xytext=(9.3, 3.7),
                arrowprops=dict(arrowstyle="->", color=gray_n, lw=1.5,
                               connectionstyle="arc3,rad=0.25"),
                fontsize=8, color=gray_n, style="italic", ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                         edgecolor=gray_n, alpha=0.8))

    # Pipeline block breakdown
    y_block = 1.5
    ax.text(0.5, y_block + 0.8, "Pipeline Blocks (each swappable):", fontsize=9,
            fontweight="bold", color=tuple(v / 255 for v in NAVY))

    block_items = [
        ("Preparation", "Imputation, outlier\nhandling", BLUE),
        ("Preprocessing", "Scaling, polynomial\nfeatures", TEAL),
        ("Feature Selection", "SelectKBest,\nPCA", ORANGE),
        ("Algorithm", "XGBoost, RF,\nExtraTrees, ...", GREEN),
    ]
    for i, (name, desc, color) in enumerate(block_items):
        x = 0.5 + i * 2.3
        c = tuple(v / 255 for v in color)
        rect = mpatches.FancyBboxPatch((x, y_block - 0.8), 2.0, 1.4,
                                        boxstyle="round,pad=0.08",
                                        facecolor=c, alpha=0.12,
                                        edgecolor=c, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 1.0, y_block + 0.15, name, ha="center", va="center",
                fontsize=7.5, fontweight="bold", color=c)
        ax.text(x + 1.0, y_block - 0.35, desc, ha="center", va="center",
                fontsize=6, color=(0.3, 0.3, 0.3))

    # Arrows between blocks
    for i in range(3):
        x1 = 0.5 + i * 2.3 + 2.0
        x2 = 0.5 + (i + 1) * 2.3
        if x2 > x1:
            ax.annotate("", xy=(x2, y_block), xytext=(x1, y_block),
                        arrowprops=dict(arrowstyle="->", color=(0.5, 0.5, 0.5), lw=1))

    fig.suptitle("Evolutionary AutoML Pipeline Architecture", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def fig_improvement_sources(all_data):
    """Pie/bar chart showing what type of genetic operation produced improvements."""
    # Analyze best_timeline for block-level changes
    labels_all = []

    for d in all_data:
        timeline = d["evo_data"]["best_timeline"]
        prev_sig = None
        for gen, score, desc in timeline:
            parsed = parse_description(desc)
            sig = extract_block_signature(parsed)
            if prev_sig is None:
                prev_sig = sig
                continue
            # Compare blocks
            changes = []
            block_names = ["Preparation", "Preprocessing", "Feature Selection", "Algorithm"]
            for i, bn in enumerate(block_names):
                if sig[i] != prev_sig[i]:
                    changes.append(bn)
            if not changes:
                labels_all.append("Hyperparameter\nTuning")
            elif len(changes) == 1:
                labels_all.append(f"{changes[0]}\nSwap")
            else:
                labels_all.append("Multi-Block\nSwap")
            prev_sig = sig

    counts = Counter(labels_all)
    if not counts:
        return None

    fig, ax = plt.subplots(figsize=(5, 3))
    items = counts.most_common()
    names = [x[0] for x in items]
    vals = [x[1] for x in items]

    colors_map = {
        "Hyperparameter\nTuning": tuple(v / 255 for v in BLUE),
        "Algorithm\nSwap": tuple(v / 255 for v in RED),
        "Preparation\nSwap": tuple(v / 255 for v in TEAL),
        "Preprocessing\nSwap": tuple(v / 255 for v in ORANGE),
        "Feature Selection\nSwap": tuple(v / 255 for v in GREEN),
        "Multi-Block\nSwap": tuple(v / 255 for v in ACCENT),
    }
    bar_colors = [colors_map.get(n, (0.5, 0.5, 0.5)) for n in names]

    ax.barh(range(len(names) - 1, -1, -1), vals, color=bar_colors, alpha=0.85, height=0.6)
    ax.set_yticks(range(len(names) - 1, -1, -1))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("# Improvements", fontsize=9)
    ax.set_title("Sources of Fitness Improvements", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    return fig


def fig_persona_waterfall(shap_values_row, feature_names, expected_value, title):
    """Create a waterfall-style bar chart for one individual."""
    short_names = _abbreviate_feature_names(feature_names)
    # Sort by absolute SHAP value
    idx = np.argsort(np.abs(shap_values_row))[::-1][:8]
    names = [short_names[i] for i in idx]
    vals = [shap_values_row[i] for i in idx]

    fig, ax = plt.subplots(figsize=(6, 2.8))
    # Red = pushes toward engagement (positive SHAP), blue = pushes away
    SHAP_RED = (0.84, 0.19, 0.15)   # matches shap library red
    SHAP_BLUE = (0.12, 0.38, 0.72)  # matches shap library blue
    colors = [SHAP_RED if v > 0 else SHAP_BLUE for v in vals]
    bars = ax.barh(range(len(names) - 1, -1, -1), vals, color=colors, alpha=0.8, height=0.6)
    ax.set_yticks(range(len(names) - 1, -1, -1))
    ax.set_yticklabels(names, fontsize=7)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("SHAP Value (impact on prediction)", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_mlops_architecture():
    """Create MLOps pipeline architecture diagram."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Data layer
    data_boxes = [
        (0.5, 5.5, "Snowflake\nData Warehouse", BLUE, 2.2),
        (3.2, 5.5, "Feature\nEngineering\n(dbt/SQL)", TEAL, 2.2),
        (5.9, 5.5, "Feature\nStore", ORANGE, 2.0),
    ]

    # Model layer
    model_boxes = [
        (0.5, 3.2, "AutoML\nEvolution\nEngine", RED, 2.2),
        (3.2, 3.2, "Model\nRegistry\n(MLflow)", GREEN, 2.2),
        (5.9, 3.2, "Model\nValidation\n& Testing", ACCENT, 2.0),
    ]

    # Serving layer
    serve_boxes = [
        (0.5, 1.0, "Batch Scoring\n(Airflow/dbt)", NAVY, 2.2),
        (3.2, 1.0, "Score\nDelivery\n(Snowflake)", BLUE, 2.2),
        (5.9, 1.0, "Monitoring\n& Drift\nDetection", TEAL, 2.0),
    ]

    def draw_boxes(boxes, y_label, label):
        for x, y, text, color, w in boxes:
            c = tuple(v / 255 for v in color)
            rect = mpatches.FancyBboxPatch((x, y - 0.7), w, 1.4,
                                            boxstyle="round,pad=0.1",
                                            facecolor=c, alpha=0.15,
                                            edgecolor=c, linewidth=1.8)
            ax.add_patch(rect)
            ax.text(x + w / 2, y, text, ha="center", va="center",
                    fontsize=7, fontweight="bold", color=c)

    draw_boxes(data_boxes, 5.5, "Data Layer")
    draw_boxes(model_boxes, 3.2, "Model Layer")
    draw_boxes(serve_boxes, 1.0, "Serving Layer")

    # Layer labels
    for y, label, color in [(5.5, "DATA LAYER", BLUE), (3.2, "MODEL LAYER", RED), (1.0, "SERVING LAYER", NAVY)]:
        c = tuple(v / 255 for v in color)
        ax.text(8.8, y, label, fontsize=9, fontweight="bold", color=c,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=c, alpha=0.08, edgecolor=c))

    # Arrows between layers
    for x_offset in [1.6, 4.3, 6.9]:
        ax.annotate("", xy=(x_offset, 4.4), xytext=(x_offset, 4.8),
                    arrowprops=dict(arrowstyle="->", color=(0.5, 0.5, 0.5), lw=1.2))
        ax.annotate("", xy=(x_offset, 2.1), xytext=(x_offset, 2.5),
                    arrowprops=dict(arrowstyle="->", color=(0.5, 0.5, 0.5), lw=1.2))

    # Horizontal arrows within layers
    for y in [5.5, 3.2, 1.0]:
        ax.annotate("", xy=(3.2, y), xytext=(2.7, y),
                    arrowprops=dict(arrowstyle="->", color=(0.6, 0.6, 0.6), lw=1))
        ax.annotate("", xy=(5.9, y), xytext=(5.4, y),
                    arrowprops=dict(arrowstyle="->", color=(0.6, 0.6, 0.6), lw=1))

    # Feedback loop — from Monitoring back to AutoML Engine
    red_n = tuple(v / 255 for v in RED)
    ax.annotate("Drift detected\n-> retrain",
                xy=(1.6, 3.9), xytext=(6.9, 1.7),
                arrowprops=dict(arrowstyle="->", color=red_n,
                               lw=1.8, linestyle="dashed",
                               connectionstyle="arc3,rad=-0.3"),
                fontsize=7, color=red_n, ha="center", style="italic",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                         edgecolor=red_n, alpha=0.9))

    fig.suptitle("Recommended MLOps Architecture", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ---------------------------------------------------------------------------
# Individual persona examples
# ---------------------------------------------------------------------------

def get_persona_examples(shap_values, X_sample, feature_names, expected_value, problem_name,
                         original_data=None, original_names=None):
    """Find illustrative individuals from SHAP data, with original feature values."""
    preds = expected_value + shap_values.sum(axis=1)
    short_names = _abbreviate_feature_names(feature_names)

    def _build_persona(idx, label, persona_type):
        raw_features = {}
        if original_data is not None and original_names is not None:
            for i, col in enumerate(original_names):
                if i < original_data.shape[1]:
                    raw_features[col] = original_data[idx, i]
        return {
            "label": label,
            "description": _describe_persona_from_shap(shap_values[idx], short_names, persona_type, problem_name),
            "shap_row": shap_values[idx],
            "pred": float(preds[idx]),
            "raw_features": raw_features,
            "sample_idx": idx,
        }

    personas = []
    personas.append(_build_persona(np.argmax(preds), "High-Propensity Individual", "high"))
    personas.append(_build_persona(np.argmin(preds), "Low-Propensity Individual", "low"))
    median_idx = np.argmin(np.abs(preds - np.median(preds)))
    personas.append(_build_persona(median_idx, "Typical Individual", "typical"))
    return personas


def _describe_persona_from_shap(shap_row, feature_names, persona_type, problem_name):
    """Describe an individual by their top SHAP drivers — always interpretable."""
    channel = problem_name.split("-")[0]

    # Top 3 features by absolute SHAP value
    idx = np.argsort(np.abs(shap_row))[::-1][:3]
    drivers = []
    for i in idx:
        name = feature_names[i]
        val = shap_row[i]
        direction = "increases" if val > 0 else "decreases"
        drivers.append(f"{name} ({direction} score by {abs(val):.2f})")

    if persona_type == "high":
        intro = f"This individual has the highest predicted {channel} engagement in our sample."
    elif persona_type == "low":
        intro = f"This individual has the lowest predicted {channel} engagement in our sample."
    else:
        intro = f"This individual sits near the median predicted {channel} engagement."

    return f"{intro} Key drivers: {', '.join(drivers)}."


# ---------------------------------------------------------------------------
# Main report generation
# ---------------------------------------------------------------------------

def _resolve_transformed_feature_names(best_row, problem):
    """Rebuild the pipeline and get actual transformed feature names.

    This is critical for pipelines with PolynomialFeatures, which expand
    the feature space from ~20 to 200+ interaction terms. Without this,
    SHAP plots just show 'feature_0', 'feature_1', etc.
    """
    from prepare import load_data, split_data, auto_preprocess
    from pipeline import build_sklearn_pipeline
    from search_space import get_registry

    X, y = load_data(problem)
    X = auto_preprocess(X)
    X_train, X_val, y_train, y_val = split_data(X, y)
    original_names = list(X_train.columns)

    X_train_np = X_train.values if hasattr(X_train, "values") else X_train
    y_train_np = y_train.values if hasattr(y_train, "values") else y_train

    desc = best_row["description"]
    parsed = parse_description(desc)
    config = rebuild_config_from_description(parsed["full_desc"])

    registry = get_registry()
    pipe = build_sklearn_pipeline(config, registry, problem["task"])
    pipe.fit(X_train_np, y_train_np)

    # Walk through transform steps and track feature names
    from sklearn.pipeline import Pipeline
    current_names = original_names[:]
    for step_name, step in pipe.named_steps.items():
        if step_name == "algorithm":
            break
        if hasattr(step, "get_feature_names_out"):
            try:
                out_names = step.get_feature_names_out(current_names)
                current_names = list(out_names)
            except Exception:
                try:
                    out_names = step.get_feature_names_out()
                    current_names = list(out_names)
                except Exception:
                    pass
        elif hasattr(step, "get_support"):
            mask = step.get_support()
            current_names = [n for n, m in zip(current_names, mask) if m]

    return current_names, original_names


def _shorten_poly_name(name):
    """Shorten a PolynomialFeatures name like 'total_pageviews total_ad_clicks' to something readable."""
    # PolynomialFeatures names use spaces: "x0 x1" means x0 * x1, "x0^2" means x0 squared
    parts = name.split(" ")
    if len(parts) == 1:
        # Single feature or squared term
        return name.replace("^2", " (squared)")
    elif len(parts) == 2:
        return f"{parts[0]} x {parts[1]}"
    else:
        return " x ".join(parts)


def collect_all_problem_data():
    """Load and analyze all 3 propensity problems."""
    all_rows = load_results()
    runs = detect_runs(all_rows)
    print(f"Loaded {len(all_rows)} result rows across {len(runs)} runs.", file=sys.stderr)

    problem_paths = [f"problems/{p}.toml" for p in
                     ["email_propensity", "event_propensity", "web_propensity"]]

    all_data = []
    for path in problem_paths:
        if not os.path.exists(path):
            print(f"  Skipping {path} (not found)", file=sys.stderr)
            continue

        from prepare import load_problem
        problem = load_problem(path)
        name = problem["name"]
        direction = problem["direction"]

        print(f"\nAnalyzing {name}...", file=sys.stderr)

        start, end = pick_run_for_problem(all_rows, runs, problem)
        rows = all_rows[start:end]

        best_row = find_best_row(rows, direction)
        if not best_row:
            print(f"  No successful pipelines for {name}", file=sys.stderr)
            continue

        # Feature importances
        feature_names, importances, imp_type = compute_feature_importances(best_row, problem)

        # SHAP data
        shap_data = compute_shap_data(best_row, problem)

        # Also load original (pre-transform) validation data for persona feature tables
        original_val_data = None
        original_val_names = None
        try:
            from prepare import load_data, split_data, auto_preprocess
            X_raw, y_raw = load_data(problem)
            X_raw = auto_preprocess(X_raw)
            X_train_raw, X_val_raw, _, _ = split_data(X_raw, y_raw)
            original_val_names = list(X_val_raw.columns)
            original_val_np = X_val_raw.values if hasattr(X_val_raw, "values") else X_val_raw
            # Use same RNG + indices as compute_shap_data to match rows
            n_shap = min(500, original_val_np.shape[0])
            shap_indices = np.random.RandomState(42).choice(original_val_np.shape[0], n_shap, replace=False)
            original_val_data = original_val_np[shap_indices]
        except Exception as e:
            print(f"  Warning: could not load original val data: {e}", file=sys.stderr)

        # If SHAP feature names are generic (feature_0, feature_1, ...),
        # resolve them to actual transformed names
        if shap_data is not None:
            shap_values, X_sample, shap_feat_names, expected_value = shap_data
            if shap_feat_names and shap_feat_names[0].startswith("feature_"):
                print(f"  Resolving transformed feature names for {name}...", file=sys.stderr)
                try:
                    resolved_names, orig_names = _resolve_transformed_feature_names(best_row, problem)
                    if len(resolved_names) == len(shap_feat_names):
                        resolved_names = [_shorten_poly_name(n) for n in resolved_names]
                        shap_data = (shap_values, X_sample, resolved_names, expected_value)
                        print(f"    Resolved {len(resolved_names)} feature names", file=sys.stderr)
                    else:
                        print(f"    Name count mismatch: {len(resolved_names)} vs {len(shap_feat_names)}",
                              file=sys.stderr)
                except Exception as e:
                    print(f"    Warning: could not resolve names: {e}", file=sys.stderr)

        # Evolution data
        evo_data = compute_evolution_data(rows, direction)

        # Stats
        ok_rows = [r for r in rows if r["status"] == "ok"]
        err_rows = [r for r in rows if r["status"] == "error"]
        gens = set()
        for r in rows:
            try:
                gens.add(int(r["generation"]))
            except (ValueError, KeyError):
                pass

        parsed = parse_description(best_row["description"])

        all_data.append({
            "name": name,
            "problem": problem,
            "best_row": best_row,
            "rows": rows,
            "score": float(best_row["score"]),
            "n_evals": len(rows),
            "n_ok": len(ok_rows),
            "n_errors": len(err_rows),
            "max_gen": max(gens) if gens else 0,
            "feature_names": feature_names,
            "importances": importances,
            "imp_type": imp_type,
            "shap_data": shap_data,
            "original_val_data": original_val_data,
            "original_val_names": original_val_names,
            "evo_data": evo_data,
            "best_pipeline_desc": parsed["full_desc"],
            "best_alg": extract_algorithm_name(parsed.get("algorithm", "")),
        })

    return all_data


def build_pdf(all_data):
    """Build the full stakeholder PDF report."""
    pdf = StakeholderPDF()

    # ======================================================================
    # COVER PAGE
    # ======================================================================
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Arial", "B", 28)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 15, "Propensity Model Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Arial", "", 14)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 8, "Email, Event & Web Engagement Scoring", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # Accent line
    y = pdf.get_y()
    pdf.set_draw_color(*BLUE)
    pdf.set_line_width(1.2)
    pdf.line(pdf.w / 2 - 40, y, pdf.w / 2 + 40, y)
    pdf.ln(12)

    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(*BLACK)
    pdf.cell(0, 7, "Prepared for: R&A Team  |  Data Engineering  |  CTO Office",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "April 2026", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(25)

    # Key results preview
    items = []
    for d in all_data:
        label = PROBLEM_LABELS.get(d["name"], d["name"])
        color = PROBLEM_COLORS.get(d["name"], BLUE)
        items.append((label, f"{d['score']:.4f} AUC", color))
    if items:
        pdf.key_metric_row(items)

    pdf.ln(10)
    pdf.set_font("Arial", "I", 9)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 5, "Generated by AutoML Evolutionary Framework — Automated Model Selection & Optimization",
             align="C", new_x="LMARGIN", new_y="NEXT")

    # ======================================================================
    # EXECUTIVE SUMMARY
    # ======================================================================
    pdf.add_page()
    pdf.section_title("Executive Summary")

    pdf.body_text(
        "We built three propensity models — email, event, and web engagement — using an "
        "evolutionary AutoML system that automatically discovers optimal machine learning pipelines. "
        "The short version: all three models perform well (AUC 0.90+), and the patterns they reveal "
        "are genuinely actionable."
    )

    pdf.body_text(
        "The big insight across all three models is the same, and it's probably not surprising: "
        "recency matters more than volume. Someone who did anything in the last 3 months is dramatically more "
        "likely to engage again than someone who was active 6+ months ago — regardless of how much they "
        "engaged historically. This has direct implications for how we allocate outreach resources."
    )

    pdf.aside_box(
        "The Bottom Line",
        "All three models achieve AUC > 0.90, which means they correctly rank an engaged individual "
        "above a non-engaged one roughly 90% of the time. That's strong enough for production scoring, "
        "prioritization, and audience segmentation. The evolutionary optimization process explored "
        "thousands of pipeline configurations to find these results — something that would take "
        "a data scientist weeks to do manually.",
        TEAL
    )

    # Model results summary
    pdf.section_title("Model Results at a Glance", level=2)
    items = []
    for d in all_data:
        items.append((
            PROBLEM_LABELS.get(d["name"], d["name"]),
            f"{d['score']:.4f} AUC",
            PROBLEM_COLORS.get(d["name"], BLUE),
        ))
    pdf.key_metric_row(items)
    pdf.ln(4)

    for d in all_data:
        name = PROBLEM_LABELS.get(d["name"], d["name"])
        pdf.set_font("Arial", "B", 10)
        pdf.set_text_color(*PROBLEM_COLORS.get(d["name"], BLUE))
        pdf.cell(60, 5, f"{name}:")
        pdf.set_font("Arial", "", 10)
        pdf.set_text_color(*BLACK)
        pdf.cell(0, 5,
                 f"AUC {d['score']:.4f}  |  {d['max_gen']} generations  |  "
                 f"{d['n_evals']} evaluations  |  Best: {d['best_alg']}",
                 new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Score progression chart
    fig = fig_score_progression(all_data)
    pdf.add_figure(fig, "AUC convergence for each propensity model (each panel has its own y-axis scale to show detail). Higher is better. Most improvement happens in the first 20-30 generations; later generations refine hyperparameters for diminishing gains.")

    # ======================================================================
    # INDIVIDUAL MODEL DEEP DIVES
    # ======================================================================
    for d in all_data:
        _write_model_section(pdf, d)

    # ======================================================================
    # MLOps RECOMMENDATIONS
    # ======================================================================
    pdf.add_page()
    pdf.section_title("MLOps Pipeline Recommendations")

    pdf.body_text(
        "Getting a model into a Jupyter notebook is the easy part — keeping it alive and useful "
        "in production is where the real work begins. Here's how I'd recommend we operationalize "
        "these propensity scores in our existing data infrastructure."
    )

    pdf.section_title("Architecture Overview", level=2)
    fig = fig_mlops_architecture()
    pdf.add_figure(fig, "Recommended production architecture. Data flows left-to-right within each layer; a drift-detection feedback loop (dashed) triggers automated retraining when feature distributions shift.")

    pdf.section_title("Data Layer", level=3)
    pdf.body_text(
        "The feature engineering SQL already lives in well-structured CTEs against our "
        "PERSON_BRAND_ENGAGEMENT_MONTHLY tables. The natural next step is to formalize these "
        "as dbt models — one feature-engineering model per propensity type. This gives us "
        "version control, lineage tracking, and testability for free."
    )

    pdf.aside_box(
        "Why dbt?",
        "We're already running SQL against Snowflake for feature engineering. dbt just wraps "
        "that SQL in a framework that handles dependencies, testing, and documentation. The "
        "migration from our current CTEs to dbt models is nearly zero-effort — it's basically "
        "copy-paste plus a YAML config.",
        TEAL
    )

    pdf.section_title("Model Layer", level=3)
    pdf.body_text(
        "The evolutionary search produces an optimized sklearn pipeline (serializable via joblib). "
        "We should version these in a model registry — MLflow is the natural choice since it "
        "integrates with sklearn out of the box. Each retraining cycle produces a new model "
        "version with tracked metrics, parameters, and artifacts."
    )

    pdf.body_text(
        "Retraining cadence: monthly, triggered by the dbt feature refresh. The evolutionary "
        "search takes ~20 minutes per problem, so the full three-problem sweep runs in about "
        "an hour. Not free, but not expensive either — and it automatically adapts to distribution "
        "shifts in the data."
    )

    pdf.section_title("Serving Layer", level=3)
    pdf.body_text(
        "For propensity scores, batch inference is the right pattern (not real-time). Airflow "
        "orchestrates: (1) dbt runs to refresh features, (2) model loads from registry, (3) "
        "batch scoring writes results to a Snowflake table, (4) downstream consumers "
        "(marketing tools, BI dashboards, CRM) read from that table."
    )

    pdf.section_title("Monitoring & Drift Detection", level=3)
    pdf.body_text(
        "The key thing to watch is feature distribution drift. If RECENT_3M_EVENTS suddenly "
        "shifts (say, due to a tracking change or a marketing campaign spike), the model's "
        "predictions will degrade before the AUC metric shows it. We need to catch drift "
        "before it reaches the output scores."
    )

    pdf.body_text(
        "Population Stability Index (PSI) is the standard metric here. PSI compares the "
        "distribution of a feature (or score) between the training population and the "
        "current scoring population by binning values into deciles and measuring the "
        "divergence. The interpretation is straightforward:"
    )
    pdf.body_text(
        "  - PSI < 0.1: No significant drift. Model is stable.\n"
        "  - PSI 0.1-0.25: Moderate drift. Investigate but likely OK.\n"
        "  - PSI > 0.25: Significant drift. Retrain recommended."
    )
    pdf.body_text(
        "We should compute PSI weekly on (1) the top-5 input features for each model, "
        "(2) the output score distribution itself, and (3) the target rate if ground truth "
        "is available. For our propensity models, the features to watch are RECENT_3M_EVENTS, "
        "ACTIVE_MONTHS, and MONTHS_SINCE_LAST_* — these are the most volatile since they "
        "shift naturally with time and are heavily weighted by all three models. A Snowflake "
        "stored procedure that runs after each scoring batch can compute PSI and log results "
        "to a monitoring table. If any feature exceeds PSI 0.25, trigger an automated "
        "retraining run."
    )

    pdf.aside_box(
        "Production Checklist",
        "1. Formalize feature SQL as dbt models with tests\n"
        "2. Set up MLflow for model versioning and experiment tracking\n"
        "3. Build Airflow DAG: dbt refresh -> model scoring -> output table\n"
        "4. Add weekly PSI checks on top-5 features + score distribution (threshold: 0.25)\n"
        "5. Implement automatic retraining trigger when PSI exceeds threshold\n"
        "6. Create Snowflake views for downstream consumers\n"
        "7. Set up alerting (Slack/email) for drift events and retraining outcomes",
        GREEN
    )

    # ======================================================================
    # HOW THE MODELS EVOLVED
    # ======================================================================
    pdf.add_page()
    pdf.section_title("How the Models Evolved")

    pdf.body_text(
        "The evolutionary approach borrows from biology — literally. We start with a random "
        "population of ML pipelines, each with different combinations of data preparation, "
        "preprocessing, feature selection, and algorithms. Then we let natural selection do "
        "its thing: the best-performing pipelines reproduce (with mutations and recombination), "
        "and the worst ones die off."
    )

    fig = fig_evolution_diagram()
    pdf.add_figure(fig, "The evolutionary loop: a population of ML pipelines competes via tournament selection; winners recombine (block-swap crossover) and mutate. Each pipeline is four modular blocks (bottom row) that can be independently swapped between parents.")

    pdf.body_text(
        "What makes this more interesting than a standard hyperparameter search (like grid "
        "search or Bayesian optimization) is the block-swap crossover. Each pipeline has four "
        "modular blocks, and crossover can swap entire blocks between parent pipelines. So if "
        "one pipeline discovered that Winsorizer + StandardScaler is great preprocessing, and "
        "another discovered that XGBoost with max_depth=4 is the best algorithm — crossover "
        "can combine those insights in one step."
    )

    # Improvement sources
    fig = fig_improvement_sources(all_data)
    if fig:
        pdf.add_figure(fig, "Breakdown of which genetic operations produced new best-scoring pipelines across all three problems. Hyperparameter tuning accounts for the majority of incremental gains, but block swaps (algorithm, preprocessing) deliver the largest single-step jumps.")

    pdf.aside_box(
        "Convergence Patterns",
        "All three problems converged on XGBoost as the dominant algorithm — which is "
        "consistent with what we see across the ML competition landscape. The early "
        "generations showed genuine diversity (RandomForest, ExtraTrees, GradientBoosting all "
        "competitive), but XGBoost's adaptive learning rate and regularization consistently won "
        "out. The interesting variation was in preprocessing: web propensity benefited from "
        "PolynomialFeatures (interaction terms between features), while email and event did not.",
        ACCENT
    )

    # Algorithm diversity chart
    fig = fig_algorithm_diversity(all_data)
    pdf.add_figure(fig, "Stacked bars show how many evaluations each algorithm received per generation. Early generations are diverse (multiple colors); later generations converge almost entirely to XGBoost (dark blue), confirming it as the dominant model family for all three problems.")

    # ======================================================================
    # APPENDIX A: METHODS
    # ======================================================================
    pdf.add_page()
    pdf.section_title("Appendix A: Methods & Possible Improvements")

    pdf.section_title("Current Approach: Evolutionary AutoML", level=2)
    pdf.body_text(
        "The system uses a steady-state island-model genetic algorithm with tournament "
        "selection (k=3). Population size is 20 with 10 offspring per generation. Crossover "
        "(probability 0.7) swaps pipeline blocks independently with 50% probability per block. "
        "Mutation (probability 0.3) applies one of five operators: hyperparameter perturbation, "
        "algorithm swap, preparation modification, preprocessing modification, or feature "
        "selection modification."
    )

    pdf.body_text(
        "Fitness is evaluated via 5-fold cross-validation AUC on a train/validation split "
        "(80/20). Each pipeline has a 30-second timeout to prevent runaway computations. The "
        "full search runs for 20 minutes per problem with a minimum of 2,000 evaluations."
    )

    pdf.section_title("Traditional Statistical Models", level=2)
    pdf.body_text(
        "Could we add OLS regression or logistic regression to the search space? Absolutely — "
        "and we should. Logistic regression with proper regularization (L1/L2 via ElasticNet) "
        "is a strong baseline for binary classification, and its coefficients are directly "
        "interpretable. The evolutionary framework already supports it as an algorithm block; "
        "we just need to add LogisticRegression to the registry with appropriate hyperparameter "
        "ranges (C, penalty type, solver)."
    )

    pdf.body_text(
        "That said, I wouldn't expect logistic regression to beat XGBoost on these problems. "
        "The feature interactions we see in the data (particularly for web propensity) suggest "
        "nonlinear relationships that linear models would miss. But having a logit baseline "
        "is valuable for interpretability and as a regulatory-friendly fallback."
    )

    pdf.section_title("Bayesian Optimization", level=2)
    pdf.body_text(
        "Bayesian optimization (e.g., via Optuna or SMAC) would be a natural complement to "
        "the evolutionary search. Where evolution excels at structural search (which blocks "
        "to combine), Bayesian methods are more sample-efficient for continuous hyperparameter "
        "tuning. A hybrid approach — evolution for architecture, Bayesian for hyperparameters — "
        "could converge faster."
    )

    pdf.aside_box(
        "Hybrid Approach",
        "The most promising improvement would be a two-phase search: (1) evolutionary search "
        "for pipeline structure (which preprocessing, which algorithm), then (2) Bayesian "
        "optimization to fine-tune the winning pipeline's hyperparameters. This gets us the "
        "best of both worlds — broad structural exploration followed by focused refinement.",
        ORANGE
    )

    pdf.section_title("Deep Learning & Neural Approaches", level=2)
    pdf.body_text(
        "For tabular data like ours, deep learning is typically not the right tool. Recent "
        "benchmarks (Grinsztajn et al., 2022) consistently show that tree-based methods "
        "outperform neural networks on tabular datasets, especially with < 100K rows. That "
        "said, there are a few scenarios where neural approaches could add value:"
    )

    pdf.body_text(
        "- TabNet (Arik & Pfister, 2019) uses attention mechanisms for built-in feature "
        "selection and can sometimes match XGBoost while providing attention-based "
        "interpretability.\n"
        "- Autoencoders for representation learning — if we had richer engagement sequences "
        "(not just monthly aggregates), an LSTM or transformer could learn temporal patterns "
        "that our hand-crafted features miss.\n"
        "- Neural Architecture Search (NAS) could extend our evolutionary approach to include "
        "neural network architectures alongside tree-based methods."
    )

    pdf.section_title("Ensemble Methods", level=2)
    pdf.body_text(
        "The evolutionary search finds a single best pipeline, but the population at "
        "convergence often contains several strong and diverse models. Stacking the top-5 "
        "pipelines (using a logistic regression meta-learner) would likely squeeze out another "
        "0.5-1% AUC improvement. This is the single lowest-effort improvement we could make."
    )

    # ======================================================================
    # APPENDIX B: DETAILED EVOLUTION DATA
    # ======================================================================
    pdf.add_page()
    pdf.section_title("Appendix B: Detailed Evolution Statistics")

    for d in all_data:
        name = PROBLEM_LABELS.get(d["name"], d["name"])
        color = PROBLEM_COLORS.get(d["name"], BLUE)

        pdf.section_title(name, level=2)

        # Stats table
        pdf.set_font("Arial", "", 9)
        pdf.set_text_color(*BLACK)

        stats = [
            ("Best AUC", f"{d['score']:.6f}"),
            ("Best Algorithm", d["best_alg"]),
            ("Total Evaluations", str(d["n_evals"])),
            ("Successful Evaluations", str(d["n_ok"])),
            ("Errors/Timeouts", str(d["n_errors"])),
            ("Generations", str(d["max_gen"])),
            ("Error Rate", f"{d['n_errors'] / d['n_evals'] * 100:.1f}%"),
            ("Best Pipeline", d["best_pipeline_desc"][:80] + ("..." if len(d["best_pipeline_desc"]) > 80 else "")),
        ]

        for label, value in stats:
            pdf.set_font("Arial", "B", 9)
            pdf.cell(50, 5, label + ":")
            pdf.set_font("Arial", "", 9)
            pdf.cell(0, 5, value, new_x="LMARGIN", new_y="NEXT")

        pdf.ln(3)

        # Best score timeline
        timeline = d["evo_data"]["best_timeline"]
        if timeline:
            pdf.set_font("Arial", "B", 9)
            pdf.cell(0, 5, "Improvement Timeline:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Arial", "", 8)
            for gen, score, desc in timeline[:15]:  # Top 15 improvements
                parsed = parse_description(desc)
                alg = extract_algorithm_name(parsed.get("algorithm", ""))
                pdf.cell(0, 4, f"  Gen {gen:>4}: {score:.6f}  ({alg})",
                         new_x="LMARGIN", new_y="NEXT")
            if len(timeline) > 15:
                pdf.cell(0, 4, f"  ... and {len(timeline) - 15} more improvements",
                         new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)

    # ======================================================================
    # Final page
    # ======================================================================
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Arial", "B", 18)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 12, "Questions? Let's Talk.", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(*BLACK)
    pdf.cell(0, 7, "These models are only as useful as the decisions they inform.",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "The next step is figuring out which propensity scores to operationalize first",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "and how they integrate with your team's existing workflows.",
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(15)
    pdf.set_font("Arial", "I", 10)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 7, "R&A Analytics  |  April 2026", align="C", new_x="LMARGIN", new_y="NEXT")

    return pdf


def _format_feature_val(val):
    """Format a feature value for display."""
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        if val == int(val) and abs(val) < 1e6:
            return str(int(val))
        if abs(val) < 0.01 or abs(val) > 1e5:
            return f"{val:.2e}"
        return f"{val:.2f}"
    return str(val)


def _render_persona_summary_table(pdf, personas, original_names, color):
    """Render a side-by-side comparison table of all 3 personas."""
    if not personas or not personas[0].get("raw_features"):
        return

    # Pick the most important features to show (top 8 by name recognition)
    PRIORITY_FEATURES = [
        "RECENT_3M_EVENTS", "ACTIVE_MONTHS", "BRANDS_ENGAGED",
        "MONTHS_SINCE_LAST_EMAIL", "MONTHS_SINCE_LAST_EVENT", "MONTHS_SINCE_LAST_WEB",
        "TOTAL_PAGEVIEWS", "TOTAL_AD_CLICKS", "TOTAL_FORM_SUBMITS",
        "TOTAL_EVENT_REGISTERS", "TOTAL_EVENT_ATTENDS",
        "TOTAL_EDU_REGISTERS", "TOTAL_EDU_COMPLETES",
        "MID_3M_EVENTS", "EARLY_6M_EVENTS",
        "EMAIL_EVENTS_12M", "WEB_EVENTS_12M", "EVENT_EVENTS_12M", "AD_EVENTS_12M",
        "NEW_BRAND_ENGAGEMENTS",
        "AGE", "GRADUATION_YEAR",
    ]

    # Find features present in raw data
    available = [f for f in PRIORITY_FEATURES if f in personas[0]["raw_features"]][:10]
    if not available:
        return

    pdf.ln(2)
    pdf.section_title("Persona Comparison", level=3)

    # Table header
    col_w_feat = 48
    col_w_val = 38
    row_h = 5

    # Check page space
    needed = row_h * (len(available) + 2) + 10
    if pdf.get_y() + needed > pdf.h - 30:
        pdf.add_page()

    # Header row
    pdf.set_font("Arial", "B", 8)
    pdf.set_fill_color(*color)
    pdf.set_text_color(*WHITE)
    pdf.cell(col_w_feat, row_h, "Feature", border=1, fill=True)
    for p in personas:
        label = p["label"].replace(" Individual", "")
        pdf.cell(col_w_val, row_h, label, border=1, fill=True, align="C")
    pdf.ln()

    # Data rows
    for i, feat in enumerate(available):
        if i % 2 == 0:
            pdf.set_fill_color(245, 247, 250)
        else:
            pdf.set_fill_color(*WHITE)

        pdf.set_font("Arial", "", 8)
        pdf.set_text_color(*BLACK)
        pdf.cell(col_w_feat, row_h, feat, border=1, fill=True)
        for p in personas:
            val = p["raw_features"].get(feat, "")
            pdf.cell(col_w_val, row_h, _format_feature_val(val), border=1, fill=True, align="C")
        pdf.ln()

    pdf.ln(3)


def _render_persona_feature_table(pdf, raw_features, color):
    """Render a compact feature table for one individual."""
    PRIORITY = [
        "RECENT_3M_EVENTS", "MID_3M_EVENTS", "EARLY_6M_EVENTS",
        "ACTIVE_MONTHS", "BRANDS_ENGAGED",
        "MONTHS_SINCE_LAST_EMAIL", "MONTHS_SINCE_LAST_EVENT", "MONTHS_SINCE_LAST_WEB",
        "TOTAL_PAGEVIEWS", "TOTAL_AD_CLICKS", "TOTAL_FORM_SUBMITS",
        "TOTAL_EVENT_REGISTERS", "TOTAL_EVENT_ATTENDS",
        "EMAIL_EVENTS_12M", "WEB_EVENTS_12M", "EVENT_EVENTS_12M",
        "NEW_BRAND_ENGAGEMENTS", "AGE",
    ]
    features = [(k, raw_features[k]) for k in PRIORITY if k in raw_features][:8]
    if not features:
        return

    pdf.set_font("Arial", "", 8)
    pdf.set_text_color(*GRAY)
    line = "  ".join(f"{k}: {_format_feature_val(v)}" for k, v in features)
    pdf.multi_cell(0, 4, f"Raw feature values: {line}")
    pdf.ln(1)


def _write_model_section(pdf, d):
    """Write a detailed section for one propensity model."""
    name = PROBLEM_LABELS.get(d["name"], d["name"])
    color = PROBLEM_COLORS.get(d["name"], BLUE)
    channel = d["name"].split("-")[0].title()

    pdf.add_page()
    pdf.section_title(f"{name} Model")

    # Overview
    if d["name"] == "email-propensity":
        pdf.body_text(
            f"The email propensity model predicts which individuals will engage with email "
            f"communications in the coming month — opens, clicks, the whole funnel. With an "
            f"AUC of {d['score']:.4f}, it's reliably separating the engaged from the dormant."
        )
        pdf.body_text(
            "The story here is overwhelmingly about recency. RECENT_3M_EVENTS dominates "
            "the model's decision-making — someone who's been active in email over the last "
            "quarter is the strongest signal we have. That might sound obvious, but it means "
            "our targeting should weight recent engagement much more heavily than lifetime "
            "engagement volume. A user who opened 3 emails last month is worth more outreach "
            "than someone who opened 100 emails two years ago."
        )
        pdf.aside_box(
            "Actionable Insight — Email",
            "Focus email campaigns on the recently-active segment (last 3 months). For "
            "re-engagement campaigns targeting dormant users, set realistic expectations — "
            "the model suggests their probability of re-engaging drops sharply after 3+ months "
            "of inactivity. Consider shorter, more frequent touches rather than occasional blasts.",
            color
        )

    elif d["name"] == "event-propensity":
        pdf.body_text(
            f"Event propensity is a harder prediction problem — only about 3.5% of our "
            f"population attends events in any given month, making this a severely imbalanced "
            f"classification task. Despite that, the model achieves AUC {d['score']:.4f}, "
            f"which means it's finding real signal in the data."
        )
        pdf.body_text(
            "Prior event registrations are the dominant predictor (no surprise — past behavior "
            "is the best predictor of future behavior). But the cross-channel signals are "
            "interesting: email engagement and web activity both contribute to event propensity. "
            "People who are active across multiple channels are more likely to show up at events."
        )
        pdf.aside_box(
            "Actionable Insight — Events",
            "The strongest lever for event attendance is targeting people who've registered "
            "before, especially recently. But the cross-channel signal is the real opportunity: "
            "people who are active on email AND web are significantly more likely to attend "
            "events, even if they haven't attended one before. Use email and web engagement "
            "scores as a secondary filter for event promotion targeting.",
            color
        )

    elif d["name"] == "web-propensity":
        pdf.body_text(
            f"Web propensity was the most interesting model to build. Initially, the AUC was "
            f"only 0.80 — disappointing compared to the other two. Turns out, 91% of the "
            f"original population had only passive ad impressions (no actual pageviews or "
            f"clicks). After filtering to users with genuine web activity and narrowing the "
            f"target definition, AUC jumped to {d['score']:.4f}."
        )
        pdf.body_text(
            "This model also discovered something the others didn't: feature interactions "
            "matter. The evolutionary search selected PolynomialFeatures as a preprocessing "
            "step, expanding the 20-odd input features to 231 interaction terms. The model "
            "uses combinations of features (e.g., recency * brand diversity) that neither "
            "email nor event models needed."
        )
        pdf.aside_box(
            "Actionable Insight — Web",
            "Web engagement prediction benefits from understanding how features interact. "
            "A user who is both recently active AND engaged across multiple brands is "
            "disproportionately likely to continue engaging — more than either factor alone "
            "would predict. For web content targeting, consider multi-signal audience "
            "definitions rather than single-threshold rules.",
            color
        )

    # Feature importance chart
    pdf.section_title("What Drives the Model", level=2)
    fig = fig_feature_importance(
        d["feature_names"], d["importances"],
        f"Top Features — {name}", color
    )
    pdf.add_figure(fig, f"Top 10 features for {name} ranked by {d['imp_type']} importance. Longer bars = stronger influence on the model's predictions. The top 2-3 features typically account for the majority of predictive power.")

    # SHAP analysis
    if d["shap_data"] is not None:
        shap_values, X_sample, feature_names_shap, expected_value = d["shap_data"]

        pdf.section_title("SHAP Analysis — How Individual Predictions Work", level=2)
        pdf.body_text(
            "SHAP (SHapley Additive exPlanations) decomposes each individual's prediction into "
            "per-feature contributions. The beeswarm plot below shows every person in our sample "
            "as a dot — one dot per person per feature. A dot's horizontal position shows how much "
            "that feature pushed that person's prediction up (right) or down (left). Dot color "
            "indicates the person's actual feature value: red means a high value for that feature, "
            "blue means low. So if you see a cluster of red dots pushed to the right on "
            "RECENT_3M_EVENTS, that means people with many recent events get a strong boost in "
            "predicted propensity — exactly what you'd expect."
        )
        pdf.body_text(
            "The waterfall charts further below zoom in on specific individuals, breaking down "
            "their personal prediction into the contribution of each feature. Red bars push the "
            "prediction toward engagement; blue bars push it away from engagement."
        )

        # SHAP beeswarm
        try:
            fig = fig_shap_beeswarm(shap_values, X_sample, feature_names_shap,
                                    f"SHAP Feature Impact — {name}", color)
            pdf.add_figure(fig, f"SHAP beeswarm for {name}. Each dot is one person; x-axis = impact on prediction (right = more likely to engage). Color = feature value (red = high, blue = low). Features sorted top-to-bottom by overall importance.")
        except Exception as e:
            print(f"  Warning: SHAP beeswarm failed for {d['name']}: {e}", file=sys.stderr)

        # Individual examples
        pdf.section_title("Example Individuals", level=2)
        pdf.body_text(
            "To make this concrete, here are three real individuals from our dataset — a high-propensity "
            "user, a low-propensity user, and someone in the middle. The waterfall charts show "
            "which features are pushing each person's prediction up or down."
        )

        ev = float(expected_value[1]) if isinstance(expected_value, (list, np.ndarray)) and len(expected_value) > 1 else float(expected_value)
        personas = get_persona_examples(
            shap_values, X_sample, feature_names_shap, ev, d["name"],
            original_data=d.get("original_val_data"),
            original_names=d.get("original_val_names"),
        )

        # Summary comparison table of all 3 personas
        _render_persona_summary_table(pdf, personas, d.get("original_val_names", []), color)

        for persona in personas:
            pdf.section_title(persona["label"], level=3)
            pdf.body_text(persona["description"])

            # Feature value table for this individual
            if persona["raw_features"]:
                _render_persona_feature_table(pdf, persona["raw_features"], color)

            try:
                fig = fig_persona_waterfall(
                    persona["shap_row"], feature_names_shap, ev,
                    f"{persona['label']} — SHAP Decomposition"
                )
                direction_label = "toward" if "High" in persona["label"] else "away from" if "Low" in persona["label"] else "around average"
                pdf.add_figure(fig,
                    f"SHAP waterfall for a {persona['label'].lower()}. Each bar is one feature's contribution: red bars push {direction_label} engagement, blue bars push the opposite direction. Bar length = magnitude of effect.",
                    width=140)
            except Exception as e:
                print(f"  Warning: persona waterfall failed: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60, file=sys.stderr)
    print("  AutoML Stakeholder PDF Report Generator", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    print("\n[1/3] Collecting and analyzing data...", file=sys.stderr)
    all_data = collect_all_problem_data()

    if not all_data:
        print("ERROR: No problem data collected. Check results.tsv.", file=sys.stderr)
        sys.exit(1)

    print(f"\n[2/3] Building PDF ({len(all_data)} problems)...", file=sys.stderr)
    pdf = build_pdf(all_data)

    output_path = "propensity_model_report.pdf"
    print(f"\n[3/3] Writing {output_path}...", file=sys.stderr)
    pdf.output(output_path)

    print(f"\nDone! Report saved to {output_path}", file=sys.stderr)
    print(f"  File size: {os.path.getsize(output_path) / 1024:.0f} KB", file=sys.stderr)


if __name__ == "__main__":
    main()
