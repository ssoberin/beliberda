from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/user/hackaton/innopolis_2026")
OUT_DIR = ROOT / "output" / "presentation"
CHARTS_DIR = OUT_DIR / "charts"

# Apple / OpenAI clean light palette
CANVAS_BG = "#F5F5F7"
CARD_BG = "#FFFFFF"
INK = "#1D1D1F"
MUTED = "#6E6E73"
BLUE = "#0071E3"
AMBER = "#D97706"
SLATE_BAR = "#CBD5E1"


def _setup_plt() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans", "DejaVu Sans"],
        "axes.edgecolor": "#D2D2D7",
        "axes.linewidth": 0.9,
        "axes.facecolor": CARD_BG,
        "figure.facecolor": CARD_BG,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": INK,
        "ytick.color": MUTED,
        "svg.fonttype": "none",
    })


def _fig_to_svg(fig: plt.Figure, save_name: str) -> str:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    out_svg = CHARTS_DIR / save_name
    fig.savefig(out_svg, format="svg", bbox_inches="tight", facecolor=CARD_BG)
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", facecolor=CARD_BG)
    plt.close(fig)
    raw = buf.getvalue()
    raw = re.sub(r"<\?xml[^>]*\?>", "", raw).strip()
    raw = re.sub(r"<!DOCTYPE[^>]*>", "", raw).strip()
    raw = re.sub(
        r'width="[^"]*" height="[^"]*"',
        'style="width:100%;height:auto;display:block;"',
        raw,
        count=1,
    )
    return raw


def make_eda_chart() -> str:
    """Plot label count distribution directly from output/dataset-audit/statistics.json."""
    stats = json.loads((ROOT / "output/dataset-audit/statistics.json").read_text(encoding="utf-8"))
    dist = stats["train"]["label_count_distribution"]
    total = stats["train"]["rows"]
    keys = ["0", "1", "2", "3", "4", "5"]
    labels = ["0 услуг", "1", "2", "3", "4", "Все 5"]
    pcts = [dist[k] / total * 100.0 for k in keys]
    colors = [AMBER, SLATE_BAR, SLATE_BAR, SLATE_BAR, SLATE_BAR, AMBER]

    _setup_plt()
    fig, ax = plt.subplots(figsize=(6.1, 3.65), dpi=150)
    x = np.arange(len(labels))
    bars = ax.bar(x, pcts, width=0.56, color=colors, edgecolor="none", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 38)
    ax.set_ylabel("Доля объявлений, %", fontsize=10, labelpad=6)
    ax.yaxis.grid(True, linestyle="-", alpha=0.35, color="#D2D2D7", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    for b, p in zip(bars, pcts):
        is_hi = p > 25
        ax.text(
            b.get_x() + b.get_width() / 2,
            p + 0.9,
            f"{p:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold" if is_hi else "normal",
            color=AMBER if is_hi else INK,
        )

    return _fig_to_svg(fig, "chart_eda_distribution.svg")


def make_synth_curve_chart() -> str:
    """Plot learning curve directly from output/v3-fraction-study/summary.json."""
    summary = json.loads((ROOT / "output/v3-fraction-study/summary.json").read_text(encoding="utf-8"))
    runs = summary["runs"]
    fractions = ["0", "20", "60", "100"]
    x_vals = [runs[k]["synthetic_rows"] for k in fractions]
    val_729 = [runs[k]["evaluation"]["public_validation_729"]["micro_f1"] for k in fractions]
    org_208 = [runs[k]["evaluation"]["official_public_test_208"]["micro_f1"] for k in fractions]
    synth_600 = [runs[k]["evaluation"]["v3_test_600"]["micro_f1"] for k in fractions]

    _setup_plt()
    fig, ax = plt.subplots(figsize=(6.3, 3.65), dpi=150)

    ax.plot(x_vals, val_729, marker="o", markersize=6, linewidth=2.2, color="#8E8E93", label="Валидация (729 из train)", zorder=3)
    ax.plot(x_vals, org_208, marker="s", markersize=6, linewidth=2.5, color=AMBER, label="Тест организаторов (208)", zorder=4)
    ax.plot(x_vals, synth_600, marker="^", markersize=7, linewidth=2.8, color=BLUE, label="Тест на отказах и подрядчиках (600)", zorder=5)

    for xi, y1, y2, y3 in zip(x_vals, val_729, org_208, synth_600):
        ax.annotate(f"{y1:.3f}", xy=(xi, y1), xytext=(0, 8), textcoords="offset points", ha="center", va="bottom", fontsize=9.5, color="#6E6E73")
        ax.annotate(
            f"{y2:.3f}",
            xy=(xi, y2),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color=AMBER,
        )
        if xi == 0:
            ax.annotate(f"{y3:.3f}", xy=(xi, y3), xytext=(12, -2), textcoords="offset points", ha="left", va="top", fontsize=10, fontweight="bold", color=BLUE)
        else:
            ax.annotate(f"{y3:.3f}", xy=(xi, y3), xytext=(0, 8), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold", color=BLUE)

    ax.set_xticks(x_vals)
    ax.set_xticklabels([f"{v:,}".replace(",", " ") for v in x_vals], fontsize=10)
    ax.set_xlabel("Добавлено синтетических объявлений в обучение (абляция)", fontsize=9.8, labelpad=6)
    ax.set_ylabel("Micro-F1 (порог 0.5)", fontsize=10)
    ax.set_ylim(0.08, 1.09)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_xlim(-220, 3380)
    ax.yaxis.grid(True, linestyle="-", alpha=0.45, color="#E5E5EA", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, fontsize=9.2, loc="lower right")

    return _fig_to_svg(fig, "chart_synth_curve.svg")


def make_main_benchmark_chart() -> str:
    """Plot benchmark comparison across the 3 core splits from experiment JSON files."""
    sm = json.loads((ROOT / "output/syntax-aware-experiment/metrics.json").read_text(encoding="utf-8"))
    pm = json.loads((ROOT / "output/syntax-aware-experiment/public_tests_metrics.json").read_text(encoding="utf-8"))

    splits = [
        ("Валидация\n(729 из train)", sm["splits"]["pub_validation"]["thr_05"]),
        ("Тест организаторов\n(208)", pm["splits"]["official_208"]["thr_05"]),
        ("Сложные отказы\n(600)", sm["splits"]["v3_test"]["thr_05"]),
    ]

    tfidf_vals = [s[1]["tfidf_sgd"]["micro_f1"] for s in splits]
    qwen_sgd_vals = [s[1]["A_qwen_sgd"]["micro_f1"] for s in splits]
    csyn_vals = [s[1]["C_syntax"]["micro_f1"] for s in splits]
    cnoglobal_vals = [s[1]["C_no_global"]["micro_f1"] for s in splits]

    _setup_plt()
    fig, ax = plt.subplots(figsize=(7.2, 3.85), dpi=150)
    x = np.arange(len(splits)) * 1.18
    w = 0.21

    b1 = ax.bar(x - 1.5 * w, tfidf_vals, width=w, color=AMBER, label="TF-IDF + SGD", zorder=3)
    b2 = ax.bar(x - 0.5 * w, qwen_sgd_vals, width=w, color="#AEAEB2", label="Qwen3 + SGD", zorder=3)
    b3 = ax.bar(x + 0.5 * w, csyn_vals, width=w, color="#7FB5F2", label="Пайплайн баз.", zorder=3)
    b4 = ax.bar(x + 1.5 * w, cnoglobal_vals, width=w, color=BLUE, label="Пайплайн (без глоб. фич)", zorder=3)

    for idx_b, (bars, c) in enumerate(((b1, AMBER), (b2, MUTED), (b3, "#3B82F6"), (b4, BLUE))):
        for bar in bars:
            val = bar.get_height()
            dy = 0.025 if idx_b in (1, 3) else 0.008
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + dy,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.4,
                fontweight="bold",
                color=c,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in splits], fontsize=10, fontweight="bold")
    ax.set_ylim(0.58, 1.07)
    ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylabel("Micro-F1 (порог 0.5)", fontsize=10)
    ax.yaxis.grid(True, linestyle="-", alpha=0.45, color="#E5E5EA", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, fontsize=8.8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.08), columnspacing=1.0, handlelength=1.4)

    return _fig_to_svg(fig, "chart_main_benchmark.svg")


COMMON_CSS = r"""
  :root {
    --canvas: #F5F5F7;
    --card: #FFFFFF;
    --ink: #1D1D1F;
    --ink-80: rgba(29, 29, 31, 0.84);
    --muted: #6E6E73;
    --hairline: rgba(0, 0, 0, 0.08);
    --blue: #0071E3;
    --blue-soft: rgba(0, 113, 227, 0.09);
    --amber: #D97706;
    --amber-soft: rgba(217, 119, 6, 0.10);

    --font-sans: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Noto Sans", "Inter", system-ui, sans-serif;
    --font-mono: "SF Mono", "Noto Mono", "JetBrains Mono", monospace;
  }

  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  html, body {
    width: 100%;
    height: 100%;
    background: var(--canvas);
    color: var(--ink);
    font-family: var(--font-sans);
    font-size: 17px;
    line-height: 1.45;
    -webkit-font-smoothing: antialiased;
  }

  .deck {
    width: 100vw;
    height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }

  .slide {
    display: none;
    width: 1280px;
    height: 720px;
    background: var(--canvas);
    padding: 40px 68px 26px 68px;
    flex-direction: column;
    justify-content: space-between;
    position: relative;
  }

  .slide.active {
    display: flex;
  }

  .topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12.5px;
    color: var(--muted);
    letter-spacing: 0.02em;
    flex-shrink: 0;
  }

  .kicker {
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 11.5px;
    color: var(--muted);
  }

  .slide-num {
    font-family: var(--font-mono);
    font-size: 12.5px;
    color: var(--muted);
  }

  .main {
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }

  h1 {
    font-size: 33px;
    font-weight: 700;
    line-height: 1.12;
    letter-spacing: -0.03em;
    color: var(--ink);
  }

  h1.hero {
    font-size: 46px;
    line-height: 1.07;
    letter-spacing: -0.035em;
  }

  .subtitle {
    margin-top: 12px;
    font-size: 18px;
    line-height: 1.42;
    color: var(--muted);
    max-width: 58ch;
  }

  .card {
    background: var(--card);
    border: 1px solid var(--hairline);
    border-radius: 18px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.028);
  }

  .mono {
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
  }

  .blue { color: var(--blue); }
  .amber { color: var(--amber); }

  .footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-top: 10px;
    border-top: 1px solid var(--hairline);
    font-size: 12.5px;
    color: var(--muted);
    flex-shrink: 0;
  }

  .nav-btns {
    display: flex;
    gap: 8px;
  }

  .nav-btn {
    border: 1px solid var(--hairline);
    background: var(--card);
    color: var(--ink);
    font-family: var(--font-mono);
    font-size: 12.5px;
    padding: 3px 11px;
    border-radius: 999px;
    cursor: pointer;
  }

  .nav-btn:hover {
    background: rgba(0, 0, 0, 0.04);
  }

  /* Slide 1: Apple Keynote cover */
  .cover-split {
    display: grid;
    grid-template-columns: 1.08fr 0.92fr;
    gap: 44px;
    align-items: center;
  }

  .rule-pill {
    display: inline-block;
    margin-top: 18px;
    padding: 7px 14px;
    border-radius: 999px;
    background: var(--card);
    border: 1px solid var(--hairline);
    font-size: 14px;
    color: var(--ink-80);
  }

  .demo-card {
    padding: 26px 28px;
  }

  .demo-head {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--muted);
    margin-bottom: 14px;
  }

  .clause-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .clause {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 14px;
    border-radius: 11px;
    font-size: 15.5px;
    font-weight: 500;
  }

  .clause-keep {
    background: var(--blue-soft);
    color: var(--blue);
    font-weight: 600;
  }

  .clause-drop {
    background: var(--amber-soft);
    color: var(--ink-80);
  }

  .clause-tag {
    font-family: var(--font-mono);
    font-size: 12.5px;
    font-weight: 700;
    white-space: nowrap;
    flex-shrink: 0;
    margin-left: 14px;
  }

  .cats-row {
    margin-top: 28px;
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
  }

  .cat-chip {
    background: var(--card);
    border: 1px solid var(--hairline);
    border-radius: 999px;
    padding: 7px 14px;
    font-size: 14px;
    color: var(--ink-80);
  }

  .cat-chip strong {
    font-family: var(--font-mono);
    color: var(--blue);
    margin-right: 6px;
  }

  .cat-chip.src strong {
    color: var(--muted);
  }

  /* Slide 2: Bento 4 tiles */
  .bento-4 {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-top: 26px;
  }

  .bento-tile {
    padding: 24px 22px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 270px;
  }

  .bento-num {
    font-size: 36px;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.08;
    font-variant-numeric: tabular-nums;
  }

  .bento-title {
    margin-top: 20px;
    font-size: 17px;
    font-weight: 700;
    color: var(--ink);
    line-height: 1.3;
  }

  .bento-desc {
    margin-top: 8px;
    font-size: 14.5px;
    color: var(--muted);
    line-height: 1.45;
  }

  /* Split with White Chart Card */
  .split-left-stats {
    display: grid;
    grid-template-columns: 0.85fr 1.15fr;
    gap: 28px;
    align-items: center;
    margin-top: 20px;
  }

  .split-chart-left {
    display: grid;
    grid-template-columns: 1.24fr 0.76fr;
    gap: 28px;
    align-items: center;
    margin-top: 18px;
  }

  .chart-card {
    padding: 18px 20px 14px 20px;
  }

  .chart-title {
    font-size: 14.5px;
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 8px;
  }

  .stat-card {
    padding: 22px 24px;
  }

  .stat-card + .stat-card {
    margin-top: 14px;
  }

  .stat-hero {
    font-size: 42px;
    font-weight: 700;
    line-height: 1.05;
    letter-spacing: -0.035em;
    font-variant-numeric: tabular-nums;
  }

  .stat-desc {
    margin-top: 8px;
    font-size: 14.8px;
    color: var(--muted);
    line-height: 1.42;
  }

  .insight-card {
    padding: 18px 20px;
  }

  .insight-card + .insight-card {
    margin-top: 12px;
  }

  .insight-card p {
    font-size: 14.8px;
    color: var(--ink-80);
    line-height: 1.44;
  }

  /* Slide 5: Apple Table Card */
  .table-card {
    margin-top: 18px;
    padding: 8px 24px 12px 24px;
  }

  table.apple-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 15px;
  }

  table.apple-table th {
    text-align: left;
    font-size: 11.5px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    padding: 12px 8px;
    border-bottom: 1px solid var(--hairline);
  }

  table.apple-table td {
    padding: 13px 8px;
    border-bottom: 1px solid var(--hairline);
    vertical-align: top;
    color: var(--ink-80);
    line-height: 1.38;
  }

  table.apple-table tr:last-child td {
    border-bottom: none;
  }

  table.apple-table td.name {
    color: var(--ink);
    font-weight: 600;
    width: 310px;
  }

  table.apple-table td.res {
    font-family: var(--font-mono);
    font-weight: 700;
    color: var(--amber);
    width: 205px;
    white-space: nowrap;
  }

  /* Slide 6: 4-Stage Keynote Flow */
  .input-box {
    margin-top: 14px;
    padding: 14px 20px;
    display: grid;
    grid-template-columns: 110px 1fr;
    gap: 16px;
    align-items: center;
  }

  .input-lbl {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--blue);
  }

  .input-text {
    font-size: 15px;
    color: var(--ink);
    line-height: 1.42;
  }

  .pipe-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-top: 14px;
  }

  .pipe-card {
    padding: 18px 18px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 235px;
  }

  .pipe-step {
    font-family: var(--font-mono);
    font-size: 12px;
    font-weight: 700;
    color: var(--blue);
  }

  .pipe-name {
    margin-top: 6px;
    font-size: 17px;
    font-weight: 700;
    color: var(--ink);
    line-height: 1.25;
  }

  .pipe-body {
    margin-top: 10px;
    font-size: 14px;
    color: var(--muted);
    line-height: 1.42;
  }

  .pipe-out {
    margin-top: 14px;
    padding-top: 10px;
    border-top: 1px solid var(--hairline);
    font-family: var(--font-mono);
    font-size: 12.5px;
    font-weight: 600;
    color: var(--ink);
  }

  /* Slide 8: Audit + Precision/Recall tradeoff */
  .audit-bar-wrap {
    margin-top: 18px;
    padding: 18px 22px;
  }

  .audit-3 {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-top: 14px;
  }

  .audit-card {
    padding: 18px 20px;
  }

  .audit-num {
    font-size: 32px;
    font-weight: 700;
    letter-spacing: -0.025em;
  }

  .audit-txt {
    margin-top: 6px;
    font-size: 14.2px;
    color: var(--muted);
    line-height: 1.42;
  }

  .tradeoff-band {
    margin-top: 14px;
    padding: 15px 22px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 24px;
    font-size: 14.5px;
    color: var(--ink-80);
  }

  /* Slide 9: Honest Compute Comparison */
  .compute-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-top: 18px;
  }

  .compute-card {
    padding: 22px 24px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }

  .compute-tag {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
  }

  .compute-title {
    margin-top: 6px;
    font-size: 23px;
    font-weight: 700;
    letter-spacing: -0.02em;
  }

  .spec-list {
    margin-top: 16px;
    border-top: 1px solid var(--hairline);
  }

  .spec-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 14px;
    padding: 10px 0;
    border-bottom: 1px solid var(--hairline);
    font-size: 14.5px;
  }

  .spec-k { color: var(--muted); }
  .spec-v { font-family: var(--font-mono); font-weight: 600; color: var(--ink); text-align: right; }

  .compute-note {
    margin-top: 14px;
    font-size: 13.5px;
    color: var(--muted);
    line-height: 1.4;
  }

  /* Slide 10: Verdict */
  .verdict-grid {
    display: grid;
    grid-template-columns: 1fr 1.04fr;
    gap: 26px;
    align-items: start;
    margin-top: 20px;
  }

  .verdict-card {
    padding: 20px 24px;
  }

  .verdict-item {
    padding: 11px 0;
    border-bottom: 1px solid var(--hairline);
    font-size: 15.2px;
    color: var(--ink-80);
    line-height: 1.42;
  }

  .verdict-item:first-child { padding-top: 0; }
  .verdict-item:last-child { border-bottom: none; padding-bottom: 0; }

  .track-slot {
    margin-top: 14px;
    padding: 14px 18px;
    border-radius: 14px;
    border: 1.5px dashed rgba(0, 0, 0, 0.22);
    background: var(--card);
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 6px;
    font-size: 14.5px;
  }

  table.cmp-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 14.5px;
  }

  table.cmp-table th {
    text-align: left;
    font-size: 11.5px;
    font-family: var(--font-mono);
    color: var(--muted);
    padding: 11px 10px;
    border-bottom: 1px solid var(--hairline);
  }

  table.cmp-table th.num, table.cmp-table td.num {
    text-align: right;
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
  }

  table.cmp-table td {
    padding: 13px 10px;
    border-bottom: 1px solid var(--hairline);
    color: var(--ink-80);
  }

  table.cmp-table tr:last-child td { border-bottom: none; }

  table.cmp-table tr.hi-blue td {
    background: var(--blue-soft);
    color: var(--ink);
    font-weight: 700;
  }

  table.cmp-table tr.hi-amber td {
    background: var(--amber-soft);
    color: var(--ink);
    font-weight: 700;
  }

  /* Slide 11: Next Steps */
  .next-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 18px;
    margin-top: 28px;
  }

  .next-card {
    padding: 24px 24px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 235px;
  }

  .next-num {
    font-family: var(--font-mono);
    font-size: 13px;
    font-weight: 700;
    color: var(--blue);
  }

  .next-title {
    margin-top: 12px;
    font-size: 19px;
    font-weight: 700;
    line-height: 1.25;
  }

  .next-desc {
    margin-top: 10px;
    font-size: 14.8px;
    color: var(--muted);
    line-height: 1.45;
  }

  @media print {
    @page { size: 1280px 720px; margin: 0; }
    html, body { height: auto; background: var(--canvas); }
    .deck { display: block; width: auto; height: auto; overflow: visible; }
    .slide { display: flex !important; page-break-after: always; break-after: page; }
    .nav-btns { display: none !important; }
  }
"""


COMMON_JS = r"""
  const slides = Array.from(document.querySelectorAll('.slide'));
  let idx = 0;

  function show(n) {
    idx = (n + slides.length) % slides.length;
    slides.forEach((s, i) => s.classList.toggle('active', i === idx));
    const hash = '#s' + (idx + 1);
    if (window.location.hash !== hash) {
      history.replaceState(null, '', hash);
    }
  }

  document.querySelectorAll('[data-prev]').forEach(btn =>
    btn.addEventListener('click', () => show(idx - 1))
  );
  document.querySelectorAll('[data-next]').forEach(btn =>
    btn.addEventListener('click', () => show(idx + 1))
  );

  document.addEventListener('keydown', (e) => {
    if (['ArrowRight', ' ', 'PageDown', 'Enter'].includes(e.key)) {
      e.preventDefault();
      show(idx + 1);
    } else if (['ArrowLeft', 'PageUp', 'Backspace'].includes(e.key)) {
      e.preventDefault();
      show(idx - 1);
    }
  });

  const syncHash = () => {
    const m = window.location.hash.match(/^#s(\d+)$/);
    if (m) {
      const target = parseInt(m[1], 10) - 1;
      if (target >= 0 && target < slides.length) show(target);
      return true;
    }
    return false;
  };
  window.addEventListener('hashchange', syncHash);
  if (!syncHash()) show(0);
"""


def svg_fp_bar() -> str:
    w, h = 1096, 52
    w1 = w * (61 / 111)
    w2 = w * (33 / 111)
    w3 = w - w1 - w2
    return f"""
    <svg viewBox="0 0 {w} {h}" width="100%" height="52">
      <rect x="0" y="2" width="{w1 - 4:.1f}" height="48" rx="10" fill="#0071E3"/>
      <text x="{w1 / 2:.1f}" y="31" text-anchor="middle" font-size="15" font-weight="700" font-family="-apple-system, Noto Sans, sans-serif" fill="#FFFFFF">61 · 55% услуга есть в тексте</text>

      <rect x="{w1:.1f}" y="2" width="{w2 - 4:.1f}" height="48" rx="10" fill="#E5E5EA"/>
      <text x="{w1 + w2 / 2:.1f}" y="31" text-anchor="middle" font-size="14.5" font-weight="700" font-family="-apple-system, Noto Sans, sans-serif" fill="#1D1D1F">33 · 30% спорно</text>

      <rect x="{w1 + w2:.1f}" y="2" width="{w3:.1f}" height="48" rx="10" fill="#D97706"/>
      <text x="{w1 + w2 + w3 / 2:.1f}" y="31" text-anchor="middle" font-size="14.5" font-weight="700" font-family="-apple-system, Noto Sans, sans-serif" fill="#FFFFFF">17 · 15% промахи</text>
    </svg>
    """


def slide(num: int, total: int, kicker: str, body: str) -> str:
    return f"""
    <section class="slide">
      <div class="topbar">
        <span class="kicker">{kicker}</span>
        <span class="slide-num">{num:02d} / {total:02d}</span>
      </div>
      {body}
      <footer class="footer">
        <span>Авито · Автопредложение смежных услуг в ремонте</span>
        <div class="nav-btns">
          <button class="nav-btn" type="button" data-prev aria-label="Назад">←</button>
          <button class="nav-btn" type="button" data-next aria-label="Вперёд">→</button>
        </div>
      </footer>
    </section>
    """


def build_deck(mode: Literal["tfidf", "qwen"]) -> str:
    total = 11

    chart_eda = make_eda_chart()
    chart_synth = make_synth_curve_chart()
    chart_bench = make_main_benchmark_chart()

    # Slide 1: Keynote cover + unambiguous task framing (101 -> 102-106)
    s1 = slide(
        1,
        total,
        "Задача Авито",
        """
        <div class="main">
          <div class="cover-split">
            <div>
              <h1 class="hero">Допуслуги из объявлений без лишнего спама</h1>
              <p class="subtitle">
                Мастер публикует одно объявление в «Ремонте под ключ», а модель находит смежные работы
                и предлагает создать отдельные объявления только там, где исполнитель реально оказывает услугу.
              </p>
              <div class="rule-pill">
                Правило: вход <strong>101</strong> → предсказываем <strong>102–106</strong> без самой 101
              </div>
            </div>

            <div class="card demo-card">
              <div class="demo-head">Пример разбора одного объявления</div>
              <div class="clause-list">
                <div class="clause clause-keep">
                  <span>«Плитку и керамогранит кладём сами»</span>
                  <span class="clause-tag">103 · Плитка ✓</span>
                </div>
                <div class="clause clause-drop">
                  <span>«Электрику берёт внешний подрядчик»</span>
                  <span class="clause-tag">Отсев · чужое</span>
                </div>
                <div class="clause clause-drop">
                  <span>«Сантехнику и трубы не выполняем»</span>
                  <span class="clause-tag">Отсев · отказ</span>
                </div>
              </div>
            </div>
          </div>

          <div class="cats-row">
            <div class="cat-chip src"><strong>101</strong>Вход: Ремонт под ключ</div>
            <div class="cat-chip"><strong>102</strong>Штукатурка</div>
            <div class="cat-chip"><strong>103</strong>Плитка</div>
            <div class="cat-chip"><strong>104</strong>Сантехника</div>
            <div class="cat-chip"><strong>105</strong>Электрика</div>
            <div class="cat-chip"><strong>106</strong>Обои и малярка</div>
          </div>
        </div>
        """,
    )

    # Slide 2: Bento 4 tiles — clean numbers, honest split & scope
    s2 = slide(
        2,
        total,
        "Путь исследования",
        """
        <div class="main">
          <h1>От аудита 10&nbsp;000 объявлений до двух готовых решений</h1>
          <div class="bento-4">
            <div class="card bento-tile">
              <div class="bento-num">10&nbsp;000 → 4&nbsp;855</div>
              <div>
                <div class="bento-title">Очистка базы</div>
                <div class="bento-desc">Удалили нормализованные повторы и собрали сплит 3&nbsp;398 / 728 / 729.</div>
              </div>
            </div>
            <div class="card bento-tile">
              <div class="bento-num blue">+3&nbsp;000 / 600</div>
              <div>
                <div class="bento-title">Сложный стенд</div>
                <div class="bento-desc">Сгенерировали длинные тексты с отказами и подрядчиками (после чистки 2&nbsp;921 train + 600 тест).</div>
              </div>
            </div>
            <div class="card bento-tile">
              <div class="bento-num amber">7 серий</div>
              <div>
                <div class="bento-title">Проверка гипотез</div>
                <div class="bento-desc">Сравнили TF-IDF, n-граммы, rubert-tiny2, Qwen3-0.6B, Qwen 7B, Natasha UD и CatBoost.</div>
              </div>
            </div>
            <div class="card bento-tile">
              <div class="bento-num blue">111 разборов</div>
              <div>
                <div class="bento-title">Ручной аудит</div>
                <div class="bento-desc">Проверили все расхождения модели на 208 строках теста организаторов.</div>
              </div>
            </div>
          </div>
        </div>
        """,
    )

    # Slide 3: EDA — accurate numbers, no misleading "error" framing of 0-labels
    s3 = slide(
        3,
        total,
        "Аудит данных",
        f"""
        <div class="main">
          <h1>Половина текстов повторяется, а 56 из 208 строк теста противоречат себе</h1>
          <div class="split-left-stats">
            <div>
              <div class="card stat-card">
                <div class="stat-hero amber">60.8%</div>
                <div class="stat-desc">
                  строк обучения в крайностях: <strong>31.7%</strong> со всеми 5 услугами сразу и <strong>29.1%</strong> с пустым ответом (легальный исход вне 102–106). Уникальных сырых текстов — 4&nbsp;866 (после нормализации — 4&nbsp;855).
                </div>
              </div>

              <div class="card stat-card">
                <div class="stat-hero">56 / 208</div>
                <div class="stat-desc">
                  строк теста организаторов образуют 13 групп одинаковых текстов с разными ответами. Поэтому разница ±0.003&nbsp;F1 лежит внутри шума разметки.
                </div>
              </div>
            </div>

            <div class="card chart-card">
              <div class="chart-title">Число целевых услуг в исходном обучении (10&nbsp;000 строк)</div>
              {chart_eda}
            </div>
          </div>
        </div>
        """,
    )

    # Slide 4: Synthetic learning curve — honest val label, consistent counts
    s4 = slide(
        4,
        total,
        "Обучающая выборка",
        f"""
        <div class="main">
          <h1>В базе мало явных отказов — синтетика даёт +50&nbsp;п.п. без просадки теста</h1>
          <div class="split-chart-left">
            <div class="card chart-card">
              <div class="chart-title">Рост Micro-F1 при добавлении сложных примеров</div>
              {chart_synth}
            </div>

            <div>
              <div class="card insight-card">
                <p>В 208 строках теста явных маркеров вроде «не делаем» или чужого мастера — всего <strong>8–11 упоминаний</strong>, поэтому линейная модель просто запоминала слова.</p>
              </div>
              <div class="card insight-card">
                <p>Добавили в абляции до <strong class="blue">3&nbsp;100</strong> длинных объявлений (после отсева 79 шаблонов в финале — <strong class="blue">2&nbsp;921</strong>) и отдельный тест на <strong class="blue">600</strong> примеров.</p>
              </div>
              <div class="card insight-card">
                <p>Даже у линейного TF-IDF качество на отказах выросло с <strong>0.183 до 0.682</strong>, а на тесте организаторов чуть поднялось — с 0.815 до 0.822.</p>
              </div>
            </div>
          </div>
        </div>
        """,
    )

    # Slide 5: 5 rejected ideas — 100% verified from EXPERIMENTS.md
    s5 = slide(
        5,
        total,
        "Отрицательные результаты",
        """
        <div class="main">
          <h1>Пять проверенных гипотез, которые проиграли по замерам</h1>
          <div class="card table-card">
            <table class="apple-table">
              <thead>
                <tr>
                  <th>Что попробовали</th>
                  <th>Замер качества</th>
                  <th>Причина отказа</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td class="name">Word (1,6) и 339.5k n-грамм</td>
                  <td class="res">Вал. 0.935 → 0.927</td>
                  <td>Вес модели вырос в 10×, но 1&nbsp;152 сложных списка так и остались с F1 = 0.</td>
                </tr>
                <tr>
                  <td class="name">Короткая синтетика (&lt;1000 симв.)</td>
                  <td class="res">Тест 0.815 → 0.792</td>
                  <td>Шаблоны сбили распределение длин (медиана 143 симв. против 1&nbsp;467 в базе).</td>
                </tr>
                <tr>
                  <td class="name">rubert-tiny2 без чанкинга</td>
                  <td class="res">0.700 F1 на тесте</td>
                  <td>В скрипте обрезалась на 768 токенах и теряла услуги из середины длинных текстов.</td>
                </tr>
                <tr>
                  <td class="name">Qwen 2.5 7B (zero-shot)</td>
                  <td class="res">Recall 0.925 → 0.329</td>
                  <td>Жестко отсекала реальные услуги в перечнях и тратила ~0.66&nbsp;с на текст на GPU.</td>
                </tr>
                <tr>
                  <td class="name">Один вектор Qwen3 на весь текст</td>
                  <td class="res">0.667 F1 на тесте</td>
                  <td>На чистой базе без синтетики один вектор размывается между 5 разными услугами.</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
        """,
    )

    # Slide 6: Honest pipeline trace with genuine multi-line input (>240 chars) & clear architectural roles
    s6 = slide(
        6,
        total,
        "Архитектура пайплайна",
        """
        <div class="main">
          <h1>Четыре этапа без дообучения LLM: семантика плюс синтаксис</h1>

          <div class="card input-box">
            <div class="input-lbl">Многострочный вход</div>
            <div class="input-text">
              «Выполняем комплексный ремонт квартир в новостройках и вторичке по договору:<br>
              • Укладываем керамогранит и плитку любой сложности • Сантехнику и разводку труб не выполняем • Электрику берёт на себя проверенный подрядчик…»
            </div>
          </div>

          <div class="pipe-grid">
            <div class="card pipe-card">
              <div>
                <div class="pipe-step">ЭТАП 01</div>
                <div class="pipe-name">Разбивка списков</div>
                <div class="pipe-body">Режем объявления по строкам и буллетам (&gt;240 симв.), сохраняя слова из коротких пунктов и хештегов.</div>
              </div>
              <div class="pipe-out">Роль: изоляция каждого тезиса</div>
            </div>

            <div class="card pipe-card">
              <div>
                <div class="pipe-step">ЭТАП 02</div>
                <div class="pipe-name">Qwen3-0.6B + SGD</div>
                <div class="pipe-body">Замороженный энкодер отбирает топ-3 релевантных фрагмента отдельно под каждую категорию 102–106.</div>
              </div>
              <div class="pipe-out">Роль: семантический отбор (0.81 на отказах)</div>
            </div>

            <div class="card pipe-card">
              <div>
                <div class="pipe-step">ЭТАП 03</div>
                <div class="pipe-name">Natasha UD (CPU)</div>
                <div class="pipe-body">Парсер зависимостей на CPU помечает частицы «не» («не выполняем» → сантехника) и чужой подряд.</div>
              </div>
              <div class="pipe-out">Роль: фильтр полярности (17.2 мс CPU)</div>
            </div>

            <div class="card pipe-card">
              <div>
                <div class="pipe-step">ЭТАП 04</div>
                <div class="pipe-name">CatBoost-голова</div>
                <div class="pipe-body">Объединяет оценки по чанкам и признаки синтаксиса (в финале без жёстких флагов «только под ключ»).</div>
              </div>
              <div class="pipe-out blue">Итог: 103 · Плитка (0.924 на отказах)</div>
            </div>
          </div>
        </div>
        """,
    )

    # Slide 7: Benchmark across 3 splits (clean, transparent, no synthetic_300 noise)
    s7 = slide(
        7,
        total,
        "Результаты замеров",
        f"""
        <div class="main">
          <h1>На сложных отказах пайплайн выше на +23&nbsp;п.п., на тесте организаторов — паритет</h1>
          <div class="split-chart-left">
            <div class="card chart-card">
              <div class="chart-title">Сравнение моделей на едином обучении (3&nbsp;398 + 2&nbsp;921 объявление)</div>
              {chart_bench}
            </div>

            <div>
              <div class="card insight-card">
                <p><strong>Сложные отказы (600):</strong> пайплайн набирает <strong class="blue">0.924</strong> против <strong class="amber">0.693</strong> у TF-IDF (исправляет 170 из 245 ошибок линейной модели на срезе без старых флагов).</p>
              </div>
              <div class="card insight-card">
                <p><strong>Тест организаторов (208):</strong> исходный синтаксис дал <strong>0.820</strong>, а после снятия 3 флагов «под ключ» — <strong class="blue">0.827</strong> против <strong>0.824</strong> у TF-IDF (EM 0.712 vs 0.688).</p>
              </div>
              <div class="card insight-card">
                <p><strong>Реалистичные сложные тексты (48 шт.):</strong> при ручной проверке 48 самых естественных сложных текстов из теста отказ/подряд TF-IDF ошиблась <strong>25 раз</strong>, пайплайн — <strong>12</strong>.</p>
              </div>
            </div>
          </div>
        </div>
        """,
    )

    # Slide 8: Audit 208 + transparent precision/recall tradeoff
    s8 = slide(
        8,
        total,
        "Аудит 208 строк",
        f"""
        <div class="main">
          <h1>Разбор 111 расхождений и баланс ложных срабатываний</h1>

          <div class="card audit-bar-wrap">
            {svg_fp_bar()}
          </div>

          <div class="audit-3">
            <div class="card audit-card">
              <div class="audit-num blue">61 случай</div>
              <div class="audit-txt">услуга прямо описана в тексте (например, поклейка обоев), но не отмечена в эталоне организаторов.</div>
            </div>
            <div class="card audit-card">
              <div class="audit-num">33 случая</div>
              <div class="audit-txt">пограничные формулировки внутри общего «ремонта под ключ» (с учётом повторов — 75 уникальных ситуаций).</div>
            </div>
            <div class="card audit-card">
              <div class="audit-num amber">17 случаев</div>
              <div class="audit-txt">реальные промахи модели на смежных терминах («выравнивание дома» → штукатурка).</div>
            </div>
          </div>

          <div class="card tradeoff-band">
            <span><strong>Продуктовый баланс (102 с услугами vs 106 без услуг):</strong> там, где услуги размечены, пайплайн точнее (<strong class="blue mono">0.938 vs 0.918</strong>),</span>
            <span>но на 106 пустых строках реже ошибается TF-IDF (<strong class="amber mono">70 vs 81</strong> ложное предложение).</span>
          </div>
        </div>
        """,
    )

    # Slide 9: Honest energy & compute economics — strictly uniform units (МБ and мс)
    s9 = slide(
        9,
        total,
        "Ресурсы и масштаб",
        """
        <div class="main">
          <h1>Два профиля нагрузки в единых метриках: МБ и мс на объявление</h1>
          <div class="compute-grid">
            <div class="card compute-card">
              <div>
                <div class="compute-tag amber">Вариант 1 · 100% CPU</div>
                <div class="compute-title">TF-IDF + SGDClassifier</div>
                <div class="spec-list">
                  <div class="spec-row"><span class="spec-k">Вес всех моделей на диске</span><span class="spec-v amber">2.8 МБ</span></div>
                  <div class="spec-row"><span class="spec-k">Время на CPU (векторизация + скор)</span><span class="spec-v amber">1.2 мс</span></div>
                  <div class="spec-row"><span class="spec-k">Время нейросети на GPU</span><span class="spec-v">0 мс (не нужна)</span></div>
                  <div class="spec-row"><span class="spec-k">Полное обучение (6&nbsp;319 текстов)</span><span class="spec-v">8.2 с на CPU</span></div>
                </div>
              </div>
              <div class="compute-note">Замер 1.2 мс получен на CPU-батче из 728 объявлений. Линейно масштабируется без видеокарт на миллионы карточек.</div>
            </div>

            <div class="card compute-card">
              <div>
                <div class="compute-tag blue">Вариант 2 · Замороженная LLM + CPU</div>
                <div class="compute-title">Qwen3-0.6B + Natasha UD + CatBoost</div>
                <div class="spec-list">
                  <div class="spec-row"><span class="spec-k">Вес всех моделей на диске</span><span class="spec-v blue">~1 238 МБ</span></div>
                  <div class="spec-row"><span class="spec-k">Время на CPU (Natasha UD + CatBoost)</span><span class="spec-v">17.2 мс</span></div>
                  <div class="spec-row"><span class="spec-k">Время нейросети на GPU T4</span><span class="spec-v blue">~85 мс (целый текст)</span></div>
                  <div class="spec-row"><span class="spec-k">Отказ от тяжёлой Qwen 2.5 7B</span><span class="spec-v">~660 мс GPU (отброшена)</span></div>
                </div>
              </div>
              <div class="compute-note">Вес ~1 238 МБ = ~1 200 МБ Qwen3 (fp16) + 37.4 МБ Natasha + 0.4 МБ CatBoost. Отдельный сквозной E2E-профиль на 1 запрос не снимался.</div>
            </div>
          </div>
        </div>
        """,
    )

    # Slide 10: Variant 1 vs Variant 2 Verdict — strictly uniform units (МБ and мс)
    if mode == "tfidf":
        h1_txt = 'Рекомендуем для запуска <span class="amber">TF-IDF + SGD</span>'
        pts = """
          <div class="verdict-item">На тесте организаторов не уступает сложному пайплайну (<strong>0.824 против 0.827</strong>) и реже выдаёт лишние черновики на пустых объявлениях (<strong>70 против 81</strong>).</div>
          <div class="verdict-item">Весит <strong>2.8 МБ</strong> (в ~440× меньше пайплайна), считается за <strong>1.2 мс</strong> на CPU в батче и обходится без видеокарт в проде.</div>
          <div class="verdict-item">Переобучается на всех 6&nbsp;319 объявлениях за <strong>8.2 с</strong> на CPU и не привязан к правилам одной категории.</div>
        """
        r1_cls, r2_cls = "hi-amber", ""
        slot_html = '<span style="color:var(--muted);">Закрытый трек Авито (Micro-F1)</span><span class="mono">TF-IDF + SGD: <strong>[ 0.____ ]</strong> &nbsp; Пайплайн: [ 0.____ ]</span>'
    else:
        h1_txt = 'Выбираем пайплайн <span class="blue">Qwen3 + Natasha UD + CatBoost</span>'
        pts = """
          <div class="verdict-item">Максимальное качество: <strong>0.827 F1</strong> (EM 0.712) на тесте организаторов и <strong>0.924 F1</strong> на длинных текстах с отказами и подрядчиками.</div>
          <div class="verdict-item">Точнее находит смежные услуги там, где они есть (<strong>0.938 против 0.918</strong>), и исправляет 170 из 245 ошибок TF-IDF на сложных текстах.</div>
          <div class="verdict-item">Нейросеть <strong>Qwen3-0.6B</strong> не дообучается — на CPU обновляется только лёгкий <strong>CatBoost (0.4 МБ)</strong>, в 8× быстрее отброшенной Qwen 7B.</div>
        """
        r1_cls, r2_cls = "", "hi-blue"
        slot_html = '<span style="color:var(--muted);">Закрытый трек Авито (Micro-F1)</span><span class="mono">Пайплайн: <strong>[ 0.____ ]</strong> &nbsp; TF-IDF + SGD: [ 0.____ ]</span>'

    s10 = slide(
        10,
        total,
        "Итоговый выбор",
        f"""
        <div class="main">
          <h1>{h1_txt}</h1>
          <div class="verdict-grid">
            <div>
              <div class="card verdict-card">{pts}</div>
              <div class="track-slot">{slot_html}</div>
            </div>

            <div class="card table-card" style="margin-top:0; padding:10px 20px;">
              <table class="cmp-table">
                <thead>
                  <tr>
                    <th>Критерий</th>
                    <th class="num">TF-IDF + SGD</th>
                    <th class="num">Qwen3 + Natasha + CB</th>
                  </tr>
                </thead>
                <tbody>
                  <tr class="{r1_cls}">
                    <td>Тест организаторов 208 (F1)</td>
                    <td class="num">0.824</td>
                    <td class="num">0.827</td>
                  </tr>
                  <tr class="{r2_cls}">
                    <td>Сложные отказы 600 (F1)</td>
                    <td class="num">0.693</td>
                    <td class="num">0.924</td>
                  </tr>
                  <tr>
                    <td>Вес всех моделей</td>
                    <td class="num" style="white-space:nowrap;">2.8 МБ</td>
                    <td class="num" style="white-space:nowrap;">~1 238 МБ</td>
                  </tr>
                  <tr>
                    <td>Обработка на CPU</td>
                    <td class="num" style="white-space:nowrap;">1.2 мс</td>
                    <td class="num" style="white-space:nowrap;">17.2 мс</td>
                  </tr>
                  <tr>
                    <td>Векторизация на GPU T4</td>
                    <td class="num" style="white-space:nowrap;">0 мс</td>
                    <td class="num" style="white-space:nowrap;">~85 мс</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
        """,
    )

    # Slide 11: Concrete engineering next steps directly grounded in verified failure analysis & experiments
    s11 = slide(
        11,
        total,
        "Дальнейшие шаги",
        """
        <div class="main">
          <h1>Три конкретных доработки по результатам аудита ошибок</h1>
          <div class="next-grid">
            <div class="card next-card">
              <div>
                <div class="next-num">01 · ОШИБКИ РАЗБОРА</div>
                <div class="next-title">Закрыть 3 подтверждённых промаха</div>
              </div>
              <div class="next-desc">Смежные слова («выравнивание дома → штукатурка» из 17 промахов на тесте), частичный отказ внутри услуги («плитку на пол не кладём, только на стены») и услуги в блоке характеристик.</div>
            </div>

            <div class="card next-card">
              <div>
                <div class="next-num">02 · ЧИСТОТА ПОД КЛЮЧ</div>
                <div class="next-title">Покатегорийные пороги и CatBoost + TF-IDF</div>
              </div>
              <div class="next-desc">Снять лишние предложения на 106 пустых карточках (сейчас 70–81 FP). В наших 45 запусках добавление TF-IDF OOF и отдельных порогов уже подняло тест до 0.828 и валидацию до 0.962.</div>
            </div>

            <div class="card next-card">
              <div>
                <div class="next-num">03 · БОЕВОЙ ПРОД</div>
                <div class="next-title">Сквозной p95-замер и A/B на потоке 101</div>
              </div>
              <div class="next-desc">Замерить полную E2E-задержку 4 этапов на 1 запросе без предвычисленного кеша и проверить на живом трафике Авито, окупает ли GPU-пайплайн разницу против TF-IDF в конверсии.</div>
            </div>
          </div>
        </div>
        """,
    )

    slides_html = "\n".join([s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11])
    title = "Допуслуги Авито — TF-IDF + SGD" if mode == "tfidf" else "Допуслуги Авито — Qwen3 + Natasha UD + CatBoost"
    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>{COMMON_CSS}</style>
</head>
<body>
  <div class="deck">
    {slides_html}
  </div>
  <script>{COMMON_JS}</script>
</body>
</html>
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    p1 = OUT_DIR / "presentation_v1_tfidf_wins.html"
    p2 = OUT_DIR / "presentation_v2_qwen_natasha_wins.html"
    p1.write_text(build_deck("tfidf"), encoding="utf-8")
    p2.write_text(build_deck("qwen"), encoding="utf-8")
    print(f"Wrote {p1}, {p2}, and charts in {CHARTS_DIR}")


if __name__ == "__main__":
    main()
