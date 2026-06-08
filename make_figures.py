"""Create the five figures used in the Bondora analysis.

Inputs are read from ``data/closed_loans.csv`` and
``data/step5_predictions.csv``. Figures are written to
``figures/phase_a`` as PNG and PDF files.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# PATHS

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures" / "phase_a"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CLOSED_LOANS_PATH = DATA_DIR / "closed_loans.csv"
PREDICTIONS_PATH = DATA_DIR / "step5_predictions.csv"


# COLOR PALETTE

COUNTRY_COLORS: Dict[str, str] = {
    "EE": "#4E79A7",   # blue   — base / domestic market
    "FI": "#F28E2B",   # orange — first international expansion
    "ES": "#E15759",   # red    — highest-risk market
}

# Model colors — used in fig3, fig4, and in Deck 2(perhaps).
MODEL_COLORS: Dict[str, str] = {
    "PoD":      "#4E79A7",   # blue   — Bondora's deployed score
    "LR":       "#F28E2B",   # orange — linear baseline
    "LightGBM": "#59A14F",   # green  — boosted tree
    "XGBoost":  "#76B7B2",   # teal   — boosted tree
    "HistGBDT": "#EDC948",   # yellow — sklearn histogram GBDT
    "TabPFN-3": "#B07AA1",   # purple — frontier
}

# Brier component colors — semantic.
COMPONENT_COLORS: Dict[str, str] = {
    "REL": "#E15759",   # red   — miscalibration (smaller is better)
    "RES": "#59A14F",   # green — resolution / discrimination (larger is better)
    "UNC": "#999999",   # grey  — base-rate uncertainty (data-fixed)
    "BS":  "#4E79A7",   # blue  — total Brier (final answer)
}

# Default vs non-default fill 
DEFAULT_FILL = "#E15759"      # same red as ES — semantic: "risk"
NONDEFAULT_FILL = "#E8E8E8"   # very light grey

GRID_COLOR = "#D0D0D0"
TEXT_MUTED = "#555555"
REFERENCE_LINE = "#888888"


plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 12.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.6,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 10,
    "lines.linewidth": 2.2,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

#column_name

LABEL_ALIASES: Dict[str, List[str]] = {
    "L1": ["L1_strict_late", "default_l1"],
    "L2": ["L2_status_or_default", "default_l2"],
    "L3": ["L3_default_date", "default_l3", "default_date_only"],
    "L4": ["L4_ever_60d_late", "default_l4", "ever_60d_late"],
    "L5": ["L5_default_excl_cured", "default_l5", "cure_aware"],
}

LABEL_SHORT = {
    "L1": "L1\nstrict\nlate",
    "L2": "L2\nstatus or\ndefault",
    "L3": "L3\ndefault\ndate",
    "L4": "L4\never 60d+\nlate",
    "L5": "L5\ndefault\nexcl. cured",
}

LABEL_TAG = {"L3": "Bondora official", "L4": "Basel 60+dpd"}

PRED_ALIASES = {
    "pod":          ["pod_bondora", "PoD", "ProbabilityOfDefault"],
    "lr_1y":        ["pred_lr_1y"],
    "lr_lifetime":  ["pred_lr_lifetime"],
    "tabpfn_1y":       ["pred_tabpfn3_1y", "pred_tabpfn_1y"],
    "tabpfn_lifetime": ["pred_tabpfn3_lifetime", "pred_tabpfn_lifetime"],
    "lgb_1y":       ["pred_lgb_1y", "pred_lightgbm_1y"],
    "lgb_lifetime": ["pred_lgb_lifetime", "pred_lightgbm_lifetime"],
    "xgb_1y":       ["pred_xgboost_1y", "pred_xgb_1y"],
    "xgb_lifetime": ["pred_xgboost_lifetime", "pred_xgb_lifetime"],
    "hgbdt_1y":       ["pred_sklearn_hgbdt_1y"],
    "hgbdt_lifetime": ["pred_sklearn_hgbdt_lifetime"],
    "y_1y":         ["y_1y", "default_1y"],
    "y_lifetime":   ["y_lifetime", "default_lifetime"],
}

COUNTRY_ALIASES = ["Country", "country"]
DATE_ALIASES = ["LoanDate", "loan_date", "ListedOnUTC", "Origination_Date"]


def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# DATA LOAD

def load_closed_loans() -> pd.DataFrame:
    if not CLOSED_LOANS_PATH.exists():
        sys.exit(f"[error] not found: {CLOSED_LOANS_PATH}\n"
                 f"        Run bondora_reanalysis.py first.\n")
    df = pd.read_csv(CLOSED_LOANS_PATH, low_memory=False)
    print(f"[load] closed_loans: {len(df):,} rows, {len(df.columns)} cols")
    return df


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        sys.exit(f"[error] not found: {PREDICTIONS_PATH}\n"
                 f"        Run bondora_reanalysis.py first.\n")
    df = pd.read_csv(PREDICTIONS_PATH, low_memory=False)
    print(f"[load] step5_predictions: {len(df):,} rows, {len(df.columns)} cols")
    return df

# Calibration Brier

def reliability_curve(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_pred, bins) - 1, 0, n_bins - 1)
    centers, obs, sizes = [], [], []
    for k in range(n_bins):
        mask = idx == k
        if not mask.any():
            continue
        centers.append(y_pred[mask].mean())
        obs.append(y_true[mask].mean())
        sizes.append(mask.sum())
    return np.array(centers), np.array(obs), np.array(sizes)


def ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    c, o, s = reliability_curve(y_true, y_pred, n_bins)
    if len(s) == 0:
        return float("nan")
    return float(np.sum(s * np.abs(c - o)) / s.sum())


def brier_decomposition(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    overall = y_true.mean()
    c, o, s = reliability_curve(y_true, y_pred, n_bins)
    w = s / s.sum()
    rel = float(np.sum(w * (c - o) ** 2))
    res = float(np.sum(w * (o - overall) ** 2))
    unc = float(overall * (1 - overall))
    bs = float(np.mean((y_pred - y_true) ** 2))
    return {"REL": rel, "RES": res, "UNC": unc, "BS": bs, "base_rate": overall}


def comparison_frame(
    base: pd.DataFrame, model_cols: List[str], ycol: str
) -> pd.DataFrame:
    """Return rows with complete predictions for the requested models."""
    cols = [c for c in model_cols if c is not None] + [ycol]
    return base[cols].dropna()


def fmt_pct(x: float, decimals: int = 1) -> str:
    return f"{x * 100:.{decimals}f}%"


# Figure 1 : three-country default rate bars

def fig1_s3_country_bars(closed: pd.DataFrame) -> pd.DataFrame:
    print("\n[fig1] D1-S3 — three-country default rate")

    country_col = find_col(closed, COUNTRY_ALIASES)
    l3_col = find_col(closed, LABEL_ALIASES["L3"])
    if country_col is None or l3_col is None:
        print(f"       missing columns: country={country_col} l3={l3_col}")
        return pd.DataFrame()

    countries = ["EE", "FI", "ES"]
    sub = closed[closed[country_col].isin(countries)]
    summary = (sub.groupby(country_col)[l3_col]
                  .agg(["mean", "size"])
                  .rename(columns={"mean": "default_rate", "size": "n"})
                  .reindex(countries))
    print(summary)

    fig, ax = plt.subplots(figsize=(7.8, 5.2))

    x = np.arange(len(countries))
    rates = summary["default_rate"].values
    ns = summary["n"].values
    colors = [COUNTRY_COLORS[c] for c in countries]

    bars = ax.bar(x, rates, width=0.55, color=colors,
                  edgecolor="#222", linewidth=0.8)

    # In-bar percentage labels
    for bar, r in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, r + 0.018,
                fmt_pct(r, 1),
                ha="center", va="bottom",
                fontsize=15, fontweight="bold")

    # Sample size on x-axis
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\nn = {n:,}" for c, n in zip(countries, ns)],
                       fontsize=11)

    # pp dif from EE to ES 
    ee_rate = rates[0]
    es_rate = rates[-1]
    gap = es_rate - ee_rate
    bracket_y = max(rates) + 0.13
    ax.annotate("", xy=(len(countries) - 1, bracket_y), xytext=(0, bracket_y),
                arrowprops=dict(arrowstyle="<->", color="#222", lw=1.6))
    ax.text((len(countries) - 1) / 2, bracket_y + 0.022,
            f"+{gap * 100:.0f}pp",
            ha="center", va="bottom",
            fontsize=14, fontweight="bold", color="#222")

    # Reference dashed line at EE rate to make the gap visible
    ax.axhline(ee_rate, color=COUNTRY_COLORS["EE"], linestyle=":",
               linewidth=1.0, alpha=0.5)

    ax.set_ylabel("Default rate  (L3 — `DefaultDate` notna)")
    ax.set_ylim(0, bracket_y + 0.12)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_yticklabels([f"{int(p * 100)}%" for p in np.arange(0, 1.01, 0.1)])
    ax.grid(False, axis="x")
    ax.set_title("Default rate by country — same platform, same algorithm",
                 pad=14, loc="left")

    fig.text(0.5, -0.04,
             "Heterogeneity is structural, not noise.",
             ha="center", fontsize=10.5, style="italic", color=TEXT_MUTED)

    _save(fig, "fig1_s3_country_bars")
    plt.close(fig)
    return summary.reset_index().rename(columns={country_col: "country"})


# Figure 2 : 5-label bars + Jaccard card

def fig2_s9_label_spread(closed: pd.DataFrame) -> pd.DataFrame:
    print("\n[fig2] D1-S9 — five-label spread + Jaccard card")

    # Resolve columns
    resolved: Dict[str, str] = {}
    for key, candidates in LABEL_ALIASES.items():
        col = find_col(closed, candidates)
        if col is not None:
            resolved[key] = col
        else:
            print(f"       missing label {key}: tried {candidates}")
    if "L3" not in resolved or "L4" not in resolved:
        print("       L3 + L4 required (for Jaccard card); aborting.")
        return pd.DataFrame()

    rates = {k: closed[col].mean() for k, col in resolved.items()}
    n_total = len(closed)
    spread_pp = (max(rates.values()) - min(rates.values())) * 100

    # Jaccard L3 and L4
    l3 = closed[resolved["L3"]].astype(bool)
    l4 = closed[resolved["L4"]].astype(bool)
    l3_only = int((l3 & ~l4).sum())
    l4_only = int((~l3 & l4).sum())
    both = int((l3 & l4).sum())
    union = both + l3_only + l4_only
    jaccard = both / union if union > 0 else float("nan")

    print(f"       rates: {rates}")
    print(f"       spread = {spread_pp:.1f}pp  |  Jaccard(L3,L4) = {jaccard:.3f}")
    print(f"       L3-only={l3_only:,}  both={both:,}  L4-only={l4_only:,}")

    # Layout: bar panel (left, 3 cols) + Jaccard card (right, 1 col)
    fig = plt.figure(figsize=(13.0, 5.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[3, 3, 3, 1.8], wspace=0.45)
    ax = fig.add_subplot(gs[0, :3])
    card = fig.add_subplot(gs[0, 3])

    # Show L1..L5 top to bottom in canonical order
    order = ["L1", "L2", "L3", "L4", "L5"]
    order = [k for k in order if k in rates]
    y_pos = np.arange(len(order))[::-1]   # so L1 ends up on top

    defaults = [rates[k] for k in order]
    non_defaults = [1 - r for r in defaults]

    # Default segment -red
    ax.barh(y_pos, defaults,
            color=DEFAULT_FILL, edgecolor="#222", linewidth=0.6, height=0.65,
            label="Default")
    # Non-default segment -light grey
    ax.barh(y_pos, non_defaults, left=defaults,
            color=NONDEFAULT_FILL, edgecolor="#222", linewidth=0.6, height=0.65,
            label="Non-default")

    # In-segment rate label (white on red)
    for y, r in zip(y_pos, defaults):
        ax.text(r / 2, y, fmt_pct(r, 1),
                ha="center", va="center",
                color="white", fontsize=12, fontweight="bold")

    # Row labels (multiline, with optional tag for L3 / L4)
    for y, k in zip(y_pos, order):
        ax.text(-0.015, y, LABEL_SHORT[k].replace("\n", " "),
                ha="right", va="center", fontsize=10.5, color="#222",
                fontweight="bold" if k in LABEL_TAG else "normal")
        if k in LABEL_TAG:
            ax.text(1.02, y, f"  ★ {LABEL_TAG[k]}",
                    ha="left", va="center",
                    fontsize=9.5, color=TEXT_MUTED, style="italic")

    # Spread bracket on the left
    rmax_idx = defaults.index(max(defaults))
    rmin_idx = defaults.index(min(defaults))
    ymax = y_pos[rmax_idx]
    ymin = y_pos[rmin_idx]
    bx = -0.32
    ax.annotate("", xy=(bx, ymax), xytext=(bx, ymin),
                arrowprops=dict(arrowstyle="<->", color="#222", lw=1.6),
                annotation_clip=False)
    ax.text(bx - 0.05, (ymax + ymin) / 2,
            f"{spread_pp:.1f}pp\nspread",
            ha="right", va="center", fontsize=12, fontweight="bold")

    ax.set_yticks([])
    ax.set_xlim(-0.42, 1.30)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xticklabels([f"{int(p * 100)}%" for p in np.arange(0, 1.01, 0.2)])
    ax.set_xlabel("Share of closed loans flagged as default")
    ax.set_title("Five default definitions on the same 264K closed loans",
                 pad=12, loc="left")
    ax.grid(False, axis="y")
    ax.spines["left"].set_visible(False)

    # Jaccard card
    card.axis("off")
    card_lines = [
        "L3  ∩  L4",
        "─────────────────",
        "",
        f"Jaccard  =  {jaccard:.3f}",
        "",
        f"  L3 only  {l3_only:>7,}",
        f"  both     {both:>7,}",
        f"  L4 only  {l4_only:>7,}",
        "",
        "Bondora's official",
        "default label",
        " ≈  Basel 60+dpd",
    ]
    card.text(0.5, 0.5, "\n".join(card_lines),
              ha="center", va="center",
              family="monospace", fontsize=10.5,
              bbox=dict(boxstyle="round,pad=0.9",
                        facecolor="#F7F7F7",
                        edgecolor="#888", linewidth=0.9))

    fig.suptitle(
        f"Same loans · five labels · {spread_pp:.1f}pp spread   "
        f"— L3/L4 Jaccard = {jaccard:.3f}",
        fontsize=13.5, fontweight="bold", y=1.02,
    )

    _save(fig, "fig2_s9_label_spread")
    plt.close(fig)

    out = pd.DataFrame([
        {"label": k, "base_rate": rates[k], "n_total": n_total}
        for k in order
    ])
    out = pd.concat([out, pd.DataFrame([{
        "label": "L3∩L4_jaccard", "base_rate": jaccard, "n_total": both,
    }])], ignore_index=True)
    return out


# Figure 3 : calibration curves by model and horizon

_FIG3_MODEL_SPECS: List[tuple] = [
    ("PoD",      MODEL_COLORS["PoD"],      "-",  "o", 2.6, 9, 1.00, 6),
    ("LR",       MODEL_COLORS["LR"],       "--", "s", 1.5, 5, 0.85, 5),
    ("LightGBM", MODEL_COLORS["LightGBM"], ":",  "^", 1.5, 5, 0.85, 4),
    ("XGBoost",  MODEL_COLORS["XGBoost"],  ":",  "v", 1.5, 5, 0.85, 4),
    ("HistGBDT", MODEL_COLORS["HistGBDT"], "--", "P", 1.5, 5, 0.85, 4),
    ("TabPFN-3", MODEL_COLORS["TabPFN-3"], "-.", "D", 1.5, 5, 0.85, 4),
]


def fig3_s13_pod_mirror(preds: pd.DataFrame) -> pd.DataFrame:
    print("\n[fig3] D1-S13 — calibration by model and horizon")

    pod_col = find_col(preds, PRED_ALIASES["pod"])
    y1_col = find_col(preds, PRED_ALIASES["y_1y"])
    ylife_col = find_col(preds, PRED_ALIASES["y_lifetime"])
    if not (pod_col and y1_col and ylife_col):
        print(f"       missing columns. pod={pod_col} y_1y={y1_col} "
              f"y_lifetime={ylife_col}")
        return pd.DataFrame()

    # Resolve every non-PoD model's prediction column (per horizon).
    horizon_cols: Dict[str, Dict[str, Optional[str]]] = {
        "1y": {
            "PoD":      pod_col,
            "LR":       find_col(preds, PRED_ALIASES["lr_1y"]),
            "LightGBM": find_col(preds, PRED_ALIASES["lgb_1y"]),
            "XGBoost":  find_col(preds, PRED_ALIASES["xgb_1y"]),
            "HistGBDT": find_col(preds, PRED_ALIASES["hgbdt_1y"]),
            "TabPFN-3": find_col(preds, PRED_ALIASES["tabpfn_1y"]),
        },
        "lifetime": {
            "PoD":      pod_col,
            "LR":       find_col(preds, PRED_ALIASES["lr_lifetime"]),
            "LightGBM": find_col(preds, PRED_ALIASES["lgb_lifetime"]),
            "XGBoost":  find_col(preds, PRED_ALIASES["xgb_lifetime"]),
            "HistGBDT": find_col(preds, PRED_ALIASES["hgbdt_lifetime"]),
            "TabPFN-3": find_col(preds, PRED_ALIASES["tabpfn_lifetime"]),
        },
    }
    for h, cols in horizon_cols.items():
        present = [m for m, c in cols.items() if c is not None]
        missing = [m for m, c in cols.items() if c is None]
        print(f"       {h:<8s}: present={present}  missing={missing}")

    # Exclude structurally-missing PoD == 0 once at the top.
    base = preds[preds[pod_col] > 0].copy()

    # TabPFN-3 is trained on a subsample by design (context ≤ ~10K rows)
    FRONTIER = {"TabPFN-3"}

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.8), sharey=True)
    all_rows = []

    horizons = [
        (axes[0], "1-year horizon",        y1_col,    horizon_cols["1y"]),
        (axes[1], "Lifetime horizon (L3)", ylife_col, horizon_cols["lifetime"]),
    ]

    for ax, hname, ycol, mcols in horizons:
        # Underestimation wedge (above diagonal) — very faint orientation cue
        xs = np.linspace(0, 1, 50)
        ax.fill_between(xs, xs, 1.0, color="#E15759", alpha=0.04, linewidth=0,
                        zorder=1)

        # Perfect-calibration diagonal
        ax.plot([0, 1], [0, 1], "--", color=REFERENCE_LINE, linewidth=1.2,
                label="Perfect calibration", zorder=2)

        full_data_models = [m for m in mcols if m not in FRONTIER]
        fulldata_cols = [
            mcols[m] for m in full_data_models if mcols.get(m) is not None
        ]
        common = comparison_frame(base, fulldata_cols, ycol)
        n_common = len(common)
        base_rate = (float(common[ycol].astype(int).mean())
                     if n_common else float("nan"))
        print(f"       {hname:<22s}: full-data common n={n_common:,}  "
              f"base_rate={base_rate:.4f}")

        # Plot each available model — collect metrics for in-panel table
        metric_rows = []
        for name, color, ls, marker, lw, ms, alpha, z in _FIG3_MODEL_SPECS:
            mcol = mcols.get(name)
            if mcol is None or mcol not in base.columns:
                continue

            if name in FRONTIER:
                d = base[[mcol, ycol]].dropna()      # own (subsampled) rows
            elif n_common > 0:
                d = common[[mcol, ycol]]             # shared rows
            else:
                continue
            if len(d) == 0:
                continue

            y_pred = d[mcol].clip(0, 1).values
            y_true = d[ycol].astype(int).values

            centers, obs, _ = reliability_curve(y_true, y_pred)
            e = ece(y_true, y_pred)
            bs = float(np.mean((y_pred - y_true) ** 2))

            ax.plot(centers, obs, linestyle=ls, color=color, linewidth=lw,
                    marker=marker, markersize=ms,
                    markeredgecolor="white", markeredgewidth=0.8,
                    alpha=alpha, zorder=z,
                    label=name)

            metric_rows.append((name, e, bs, len(d)))
            all_rows.append({"horizon": hname, "model": name,
                             "ECE": e, "Brier": bs, "n": len(d),
                             "base_rate": float(y_true.mean())})

        # In-panel metric table (top-left). Sort by ECE so winner is on top.
        # Rows scored on a different n than the common set (i.e. TabPFN) get †.
        metric_rows.sort(key=lambda r: r[1])
        header = f"{'':10s}{'ECE':>7s}{'Brier':>8s}"
        sep = "─" * len(header)
        out_lines = [header, sep]
        for name, e, bs, nrow in metric_rows:
            tag = "" if nrow == n_common else " †"
            out_lines.append(f"{name:<10s}{e:7.3f}{bs:8.3f}{tag}")
        out_lines.append(sep)
        out_lines.append(f"n = {n_common:,}")
        for dn in sorted({nrow for *_, nrow in metric_rows if nrow != n_common}):
            out_lines.append(f"† n = {dn:,}")
        ax.text(0.03, 0.97, "\n".join(out_lines),
                transform=ax.transAxes, ha="left", va="top",
                family="monospace", fontsize=9.8,
                bbox=dict(boxstyle="round,pad=0.55",
                          facecolor="white", edgecolor="#777",
                          linewidth=0.9))

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Mean predicted PD")
        ax.set_title(hname, pad=10)
        ax.legend(loc="lower right", fontsize=9.5,
                  frameon=True, framealpha=0.92,
                  fancybox=False, edgecolor="#999")

    axes[0].set_ylabel("Observed default rate")
    fig.suptitle(
        "Calibration by model and target horizon",
        fontsize=13.5, fontweight="bold", y=1.02,
    )
    fig.text(
        0.5, -0.03,
        "Curves are evaluated on common rows for full-data models; "
        "TabPFN-3 uses its available prediction rows.",
        ha="center", fontsize=10.5, style="italic", color=TEXT_MUTED,
    )

    _save(fig, "fig3_s13_pod_mirror")
    plt.close(fig)
    return pd.DataFrame(all_rows)


# Figure 4  : Brier (PoD vs LR, lifetime)

def _draw_brier_components_bar(ax, decomp: Dict[str, float], title: str,
                               ymax: float) -> None:
    rel, res, unc, bs = decomp["REL"], decomp["RES"], decomp["UNC"], decomp["BS"]

    width = 0.65
    # 定义四个组件：REL(校准误差), RES(分辨率), UNC(不确定性), BS(总分)
    items = [
        (0, rel, COMPONENT_COLORS["REL"], f"+{rel:.3f}", "REL\n(Miscalibration)"),
        (1, res, COMPONENT_COLORS["RES"], f"−{res:.3f}", "RES\n(Resolution)"),
        (2, unc, COMPONENT_COLORS["UNC"], f"+{unc:.3f}", "UNC\n(Uncertainty)"),
        (3, bs,  COMPONENT_COLORS["BS"],  f"{bs:.3f}",  "BS\n(Total Brier)"),
    ]

    for x, val, color, txt, label in items:
        # 所有柱子从 0 开始画
        ax.bar(x, val, width=width,
               color=color, edgecolor="#222", linewidth=0.8)
        # 数值标在柱顶
        ax.text(x, val + ymax * 0.015, txt, ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([i[4] for i in items], fontsize=9.5)
    ax.set_title(title, pad=15)
    ax.set_ylim(0, ymax)
    ax.grid(False, axis="x")


def fig4_s14_brier_waterfall(preds: pd.DataFrame) -> pd.DataFrame:
    print("\n[fig4] D1-S14 — Brier component comparison (lifetime horizon)")

    pod_col = find_col(preds, PRED_ALIASES["pod"])
    lr_col  = find_col(preds, PRED_ALIASES["lr_lifetime"])
    ylife_col = find_col(preds, PRED_ALIASES["y_lifetime"])
    if not (pod_col and lr_col and ylife_col):
        print(f"       missing columns. pod={pod_col} "
              f"lr_lifetime={lr_col} y_lifetime={ylife_col}")
        return pd.DataFrame()

    base = preds[preds[pod_col] > 0].copy()
    common = comparison_frame(base, [pod_col, lr_col], ylife_col)
    y_true = common[ylife_col].astype(int).values
    print(f"       common n={len(common):,}  base_rate={y_true.mean():.4f}")

    pod_dec = brier_decomposition(y_true, common[pod_col].clip(0, 1).values)
    lr_dec  = brier_decomposition(y_true, common[lr_col].clip(0, 1).values)
    print(f"       PoD lifetime: {pod_dec}")
    print(f"       LR  lifetime: {lr_dec}")

    rel_ratio = pod_dec["REL"] / lr_dec["REL"] if lr_dec["REL"] > 0 else float("inf")
    res_ratio = pod_dec["RES"] / lr_dec["RES"] if lr_dec["RES"] > 0 else float("inf")
    print(f"       REL ratio PoD/LR = {rel_ratio:.1f}×    "
          f"RES ratio = {res_ratio:.2f}×")

    # Shared y-limit
    ymax = max(pod_dec["BS"], lr_dec["BS"], pod_dec["REL"], pod_dec["UNC"]) * 1.25

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6), sharey=True)
    _draw_brier_components_bar(axes[0], pod_dec,
                               "Bondora PoD  ·  lifetime", ymax)
    _draw_brier_components_bar(axes[1], lr_dec,
                               "Logistic Regression  ·  lifetime", ymax)
    axes[0].set_ylabel("Brier component value")

    fig.suptitle(
        f" PoD's miscalibration is "
        f"{rel_ratio:.0f}× larger than LR's",
        fontsize=13.5, fontweight="bold", y=1.02,
    )
    fig.text(
        0.5, -0.04,
        f"REL ratio  PoD : LR  ≈  {rel_ratio:.0f}×   (miscalibration)        "
        f"RES ratio  PoD : LR  ≈  {res_ratio:.1f}×   (resolution)        "
        "UNC same  (data-fixed)",
        ha="center", fontsize=10.5, style="italic", color=TEXT_MUTED,
    )

    _save(fig, "fig4_s14_brier_waterfall")
    plt.close(fig)
    return pd.DataFrame([
        {"model": "PoD", **pod_dec},
        {"model": "LR",  **lr_dec},
    ])


# Figure 5 : entry-year + 2017-peak country bars

def _entry_and_peak(closed: pd.DataFrame) -> Optional[pd.DataFrame]:
    country_col = find_col(closed, COUNTRY_ALIASES)
    date_col = find_col(closed, DATE_ALIASES)
    l3_col = find_col(closed, LABEL_ALIASES["L3"])
    if not (country_col and date_col and l3_col):
        print(f"       missing columns. country={country_col} "
              f"date={date_col} l3={l3_col}")
        return None
    df = closed[[country_col, date_col, l3_col]].copy()
    df["year"] = pd.to_datetime(df[date_col], errors="coerce").dt.year
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df = df[df[country_col].isin(["EE", "FI", "ES"])]

    rows = []
    for c in ["EE", "FI", "ES"]:
        sub = df[df[country_col] == c]
        if sub.empty:
            continue
        ey = int(sub["year"].min())
        er = sub[sub["year"] == ey][l3_col].mean()
        en = int((sub["year"] == ey).sum())
        py = 2017
        peak_mask = sub["year"] == py
        pr = sub[peak_mask][l3_col].mean() if peak_mask.any() else float("nan")
        pn = int(peak_mask.sum())
        rows.append({"country": c,
                     "entry_year": ey, "entry_rate": er, "entry_n": en,
                     "peak_year": py, "peak_rate": pr, "peak_n": pn})
    return pd.DataFrame(rows)


def fig5_s17_entry_year_bars(closed: pd.DataFrame) -> pd.DataFrame:
    print("\n[fig5] D1-S17 — country entry-year + 2017 peak bars")

    table = _entry_and_peak(closed)
    if table is None or table.empty:
        print("       could not compute; aborting.")
        return pd.DataFrame()
    print(table.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10.0, 5.6))

    countries = table["country"].tolist()
    x = np.arange(len(countries))
    width = 0.36

    # Entry-year bars (solid, country color)
    entry_bars = ax.bar(
        x - width / 2, table["entry_rate"], width,
        color=[COUNTRY_COLORS[c] for c in countries],
        edgecolor="#222", linewidth=0.8,
        label="Entry-year cumulative default rate",
    )
    # 2017-peak bars (same country color, hatched, lighter)
    peak_colors = [mpl.colors.to_rgba(COUNTRY_COLORS[c], alpha=0.55)
                   for c in countries]
    peak_bars = ax.bar(
        x + width / 2, table["peak_rate"], width,
        color=peak_colors,
        edgecolor="#222", linewidth=0.8, hatch="///",
        label="2017-vintage cumulative default rate",
    )

    # Value + year labels on each bar
    for bar, rate, year in zip(entry_bars, table["entry_rate"], table["entry_year"]):
        if pd.isna(rate):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.012,
                f"{rate * 100:.1f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.058,
                f"({year})",
                ha="center", va="bottom", fontsize=9, color=TEXT_MUTED)
    for bar, rate, year in zip(peak_bars, table["peak_rate"], table["peak_year"]):
        if pd.isna(rate):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.012,
                f"{rate * 100:.1f}%",
                ha="center", va="bottom", fontsize=11)
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.058,
                f"({year})",
                ha="center", va="bottom", fontsize=9, color=TEXT_MUTED)

    # Reference line at EE entry rate
    ee_entry = table.loc[table.country == "EE", "entry_rate"].iloc[0]
    ax.axhline(ee_entry, color=COUNTRY_COLORS["EE"], linestyle=":",
               linewidth=1.2, alpha=0.55,
               label=f"EE 2009 baseline = {ee_entry * 100:.1f}%")

    # Δpp annotations FI vs EE, ES vs EE
    for c, x_pos in zip(["FI", "ES"], [1, 2]):
        sub = table[table.country == c]
        if sub.empty:
            continue
        gap = sub["entry_rate"].iloc[0] - ee_entry
        ax.annotate(
            f"+{gap * 100:.0f}pp\nfrom day one",
            xy=(x_pos - width / 2, sub["entry_rate"].iloc[0]),
            xytext=(x_pos - width / 2,
                    sub["entry_rate"].iloc[0] + 0.18),
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            color="#B0413E",
            arrowprops=dict(arrowstyle="->", color="#B0413E", lw=1.3,
                            connectionstyle="arc3,rad=0.0"),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(countries, fontsize=12.5, fontweight="bold")
    ax.set_ylabel("Cumulative default rate  (L3 label)")
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_yticklabels([f"{int(p * 100)}%" for p in np.arange(0, 1.01, 0.2)])
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_title(
        "Entry-year and 2017 default rates by country",
        pad=12, loc="left",
    )
    ax.grid(False, axis="x")

    fig.text(0.5, -0.03,
             "Bars compare each country's first observed vintage with 2017.",
             ha="center", fontsize=10.5, style="italic", color=TEXT_MUTED)

    _save(fig, "fig5_s17_entry_year_bars")
    plt.close(fig)
    return table


# I/O HELPERS

def _save(fig, stem: str) -> None:
    for ext in ("png", "pdf"):
        out = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"       wrote {out}")


def write_summary(parts: Dict[str, pd.DataFrame]) -> None:
    rows = []
    for section, df in parts.items():
        if df is None or df.empty:
            continue
        df = df.copy()
        df.insert(0, "section", section)
        rows.append(df)
    if not rows:
        return
    out = pd.concat(rows, ignore_index=True, sort=False)
    path = FIG_DIR / "phase_a_summary.csv"
    out.to_csv(path, index=False)
    print(f"\n[summary] wrote {path}")


# MAIN

def main():
    parser = argparse.ArgumentParser(
        description="Create figures for the Bondora credit-risk analysis",
    )
    parser.add_argument("--only", choices=["s3", "s9", "s13", "s14", "s17"],
                        help="Run only one figure")
    args = parser.parse_args()

    print("=" * 72)
    print("Bondora analysis figures")
    print(f"Color palette: EE={COUNTRY_COLORS['EE']}  "
          f"FI={COUNTRY_COLORS['FI']}  ES={COUNTRY_COLORS['ES']}")
    print("=" * 72)

    closed = None
    preds = None
    if args.only in (None, "s3", "s9", "s17"):
        closed = load_closed_loans()
    if args.only in (None, "s13", "s14"):
        preds = load_predictions()

    parts: Dict[str, pd.DataFrame] = {}
    if args.only in (None, "s3"):
        parts["s3_country_bars"] = fig1_s3_country_bars(closed)
    if args.only in (None, "s9"):
        parts["s9_label_spread"] = fig2_s9_label_spread(closed)
    if args.only in (None, "s13"):
        parts["s13_pod_mirror"] = fig3_s13_pod_mirror(preds)
    if args.only in (None, "s14"):
        parts["s14_brier_waterfall"] = fig4_s14_brier_waterfall(preds)
    if args.only in (None, "s17"):
        parts["s17_entry_year_bars"] = fig5_s17_entry_year_bars(closed)

    write_summary(parts)

    print("\n" + "=" * 72)
    print(f"Done. Figures in:  {FIG_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
