"""Run conformal-inference experiments for the Bondora credit-risk analysis.

The conformal helper functions are included in this file to keep the
submission code compact.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def save_transparent_figure(fig, path, dpi=160, **kwargs):
    """Save a matplotlib figure with transparent figure and axes backgrounds."""
    fig.patch.set_alpha(0)
    for ax in fig.axes:
        ax.set_facecolor("none")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=True, **kwargs)


def save_current_transparent_figure(plt, path, dpi=160):
    save_transparent_figure(plt.gcf(), path, dpi=dpi)


# Split-conformal utility functions
def nonconformity_binary(p_hat: Iterable[float], y: Iterable[int]) -> np.ndarray:
    """Return binary nonconformity scores: ``1-p`` for positives, ``p`` for negatives."""
    p = np.asarray(p_hat, dtype=float)
    yy = np.asarray(y).astype(int)
    if p.shape[0] != yy.shape[0]:
        raise ValueError("p_hat and y must have the same length")
    return np.where(yy == 1, 1.0 - p, p)


def conformal_quantile(scores: Iterable[float], alpha: float) -> float:
    """Finite-sample split-conformal quantile.

    Uses the kth order statistic with ``k = ceil((n + 1) * (1 - alpha))``.
    If ``k > n``, the finite-sample-valid threshold is ``inf``.
    """
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    n = s.size
    if n == 0:
        return math.inf
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return math.inf
    return float(np.sort(s)[k - 1])


def predict_set_binary(p_hat: Iterable[float], qhat: float) -> pd.DataFrame:
    """Construct binary conformal prediction sets for a fixed threshold."""
    p = np.asarray(p_hat, dtype=float)
    include_0 = p <= qhat
    include_1 = (1.0 - p) <= qhat
    size = include_0.astype(int) + include_1.astype(int)
    return pd.DataFrame(
        {
            "include_0": include_0,
            "include_1": include_1,
            "set_size": size,
        }
    )


def set_metrics(p_hat: Iterable[float], y: Iterable[int], qhat: float) -> dict[str, float]:
    """Coverage and set-size metrics for binary prediction sets."""
    yy = np.asarray(y).astype(int)
    sets = predict_set_binary(p_hat, qhat)
    covered = np.where(yy == 1, sets["include_1"].to_numpy(), sets["include_0"].to_numpy())
    size = sets["set_size"].to_numpy()
    n = int(len(yy))
    if n == 0:
        return {
            "n": 0,
            "coverage": np.nan,
            "avg_size": np.nan,
            "ambiguity": np.nan,
            "singleton": np.nan,
            "empty": np.nan,
            "qhat": qhat,
        }
    return {
        "n": n,
        "coverage": float(np.mean(covered)),
        "avg_size": float(np.mean(size)),
        "ambiguity": float(np.mean(size == 2)),
        "singleton": float(np.mean(size == 1)),
        "empty": float(np.mean(size == 0)),
        "qhat": float(qhat),
    }


def split_conformal_metrics(
    p_cal: Iterable[float],
    y_cal: Iterable[int],
    p_eval: Iterable[float],
    y_eval: Iterable[int],
    alpha: float,
) -> dict[str, float]:
    """Calibrate a split-conformal threshold and evaluate binary set metrics."""
    scores = nonconformity_binary(p_cal, y_cal)
    qhat = conformal_quantile(scores, alpha)
    return set_metrics(p_eval, y_eval, qhat)


def random_cal_eval_indices(n: int, n_cal: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Randomly split ``n`` rows into calibration and evaluation indices."""
    if n <= 0:
        raise ValueError("n must be positive")
    if n_cal <= 0 or n_cal >= n:
        raise ValueError("n_cal must be between 1 and n - 1")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return perm[:n_cal], perm[n_cal:]


def aggregate_records(records: list[dict], group_cols: list[str]) -> pd.DataFrame:
    """Average numeric fields across repeated runs."""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    numeric = [c for c in df.columns if c not in group_cols and pd.api.types.is_numeric_dtype(df[c])]
    out = df.groupby(group_cols, dropna=False)[numeric].agg(["mean", "std"]).reset_index()
    out.columns = [
        "_".join([x for x in col if x]) if isinstance(col, tuple) else col
        for col in out.columns
    ]
    return out


def bh_select(p_values: Iterable[float], q: float) -> np.ndarray:
    """Benjamini-Hochberg selections for a vector of p-values."""
    p = np.asarray(p_values, dtype=float)
    valid = np.isfinite(p)
    selected = np.zeros(p.shape[0], dtype=bool)
    if valid.sum() == 0:
        return selected
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = ranked.size
    thresholds = q * (np.arange(1, m + 1) / m)
    ok = ranked <= thresholds
    if not ok.any():
        return selected
    k = np.max(np.where(ok)[0])
    valid_idx = np.where(valid)[0]
    selected[valid_idx[order[: k + 1]]] = True
    return selected


def conformal_pvalues_for_good_loans(
    p_cal: Iterable[float],
    y_cal: Iterable[int],
    p_eval: Iterable[float],
) -> np.ndarray:
    """One-sided conformal p-values for approving low-risk loans.

    The null class is ``Y=1``. A small score ``p_hat`` is evidence against
    default, so smaller p-values correspond to more attractive loans.
    """
    p_cal = np.asarray(p_cal, dtype=float)
    y_cal = np.asarray(y_cal).astype(int)
    p_eval = np.asarray(p_eval, dtype=float)
    null_scores = np.sort(p_cal[y_cal == 1])
    n_null = null_scores.size
    if n_null == 0:
        return np.ones(p_eval.shape[0], dtype=float)
    counts = np.searchsorted(null_scores, p_eval, side="right")
    return (1.0 + counts) / (1.0 + n_null)

SCORE_ALIASES = {
    "pod": {
        "1y": ["pod_bondora", "ProbabilityOfDefault", "pred_pod", "pred_pod_1y"],
        "lifetime": ["pod_bondora", "ProbabilityOfDefault", "pred_pod", "pred_pod_lifetime"],
    },
    "lr": {
        "1y": ["pred_lr_1y"],
        "lifetime": ["pred_lr_lifetime"],
    },
    "gbdt": {
        "1y": ["pred_lightgbm_1y", "pred_lgb_1y", "pred_xgboost_1y", "pred_sklearn_hgbdt_1y"],
        "lifetime": [
            "pred_lightgbm_lifetime",
            "pred_lgb_lifetime",
            "pred_xgboost_lifetime",
            "pred_sklearn_hgbdt_lifetime",
        ],
    },
    "tabpfn": {
        "1y": ["pred_tabpfn3_1y", "pred_tabpfn_1y"],
        "lifetime": ["pred_tabpfn3_lifetime", "pred_tabpfn_lifetime"],
    },
}

SCORE_LABELS = {
    "pod": "Bondora PoD",
    "lr": "Logistic regression",
    "gbdt": "GBDT",
    "tabpfn": "TabPFN",
}

LABEL_ALIASES = {
    "L1": ["label_L1", "L1_strict_late", "L1"],
    "L2": ["label_L2", "L2_status_or_default", "L2"],
    "L3": ["label_L3", "L3_default_date", "L3", "y_lifetime"],
    "L4": ["label_L4", "L4_ever_60d_late", "L4"],
    "L5": ["label_L5", "L5_default_excl_cured", "L5"],
}


def get_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def first_present(df: pd.DataFrame, names: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def require_col(df: pd.DataFrame, names: list[str], label: str) -> str:
    col = first_present(df, names)
    if col is None:
        raise SystemExit(f"Could not find {label}. Tried: {names}")
    return col


def load_master(args: argparse.Namespace) -> pd.DataFrame:
    path = Path(args.master)
    if not path.exists():
        candidate = Path(args.data_dir) / "conformal_master.csv"
        if candidate.exists():
            path = candidate
    return read_table(path)


def ensure_outdir(args: argparse.Namespace) -> Path:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def target_col(horizon: str) -> str:
    return "y_1y" if horizon == "1y" else "y_lifetime"


def eligible_mask(df: pd.DataFrame, horizon: str) -> pd.Series:
    flag = f"conformal_eligible_{horizon}"
    if flag in df.columns:
        return df[flag].astype(bool)
    split = df["split"].astype(str).str.lower() if "split" in df.columns else pd.Series("", index=df.index)
    return split.isin(["cal", "test", "calibration", "evaluation", "eval"]) & df[target_col(horizon)].notna()


def heldout_frame(df: pd.DataFrame, horizon: str, score_col: str, extra_cols: list[str] | None = None) -> pd.DataFrame:
    cols = [target_col(horizon), score_col]
    if "split" in df.columns:
        cols.append("split")
    if extra_cols:
        cols.extend(extra_cols)
    cols = list(dict.fromkeys([c for c in cols if c in df.columns]))
    sub = df.loc[eligible_mask(df, horizon), cols].copy()
    return sub.dropna(subset=[target_col(horizon), score_col])


def original_cal_size(sub: pd.DataFrame) -> int:
    if "split" not in sub.columns:
        return max(1, len(sub) // 2)
    split = sub["split"].astype(str).str.lower()
    n_cal = int(split.isin(["cal", "calibration"]).sum())
    if n_cal <= 0 or n_cal >= len(sub):
        n_cal = max(1, len(sub) // 2)
    return n_cal


def metric_from_set_columns(include_0: np.ndarray, include_1: np.ndarray, y: np.ndarray) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    size = include_0.astype(int) + include_1.astype(int)
    covered = np.where(y == 1, include_1, include_0)
    return {
        "n": int(len(y)),
        "coverage": float(np.mean(covered)) if len(y) else np.nan,
        "avg_size": float(np.mean(size)) if len(y) else np.nan,
        "ambiguity": float(np.mean(size == 2)) if len(y) else np.nan,
        "singleton": float(np.mean(size == 1)) if len(y) else np.nan,
        "empty": float(np.mean(size == 0)) if len(y) else np.nan,
    }


def save_bar_plot(df: pd.DataFrame, x_col: str, y_col: str, hue_col: str, title: str, path: Path) -> None:
    if df.empty:
        return
    plt = get_plt()
    pivot = df.pivot_table(index=x_col, columns=hue_col, values=y_col, aggfunc="mean")
    ax = pivot.plot(kind="bar", figsize=(9, 4), rot=0)
    ax.axhline(0.9, color="black", ls="--", lw=1, alpha=0.7)
    ax.set_title(title)
    ax.set_ylabel(y_col.replace("_", " "))
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    save_current_transparent_figure(plt, path, dpi=160)
    plt.close()


def run_four_score(args: argparse.Namespace) -> pd.DataFrame:
    df = load_master(args)
    outdir = ensure_outdir(args)
    rows = []

    for horizon in ("1y", "lifetime"):
        y_col = target_col(horizon)
        if y_col not in df.columns:
            continue
        for score_key, by_horizon in SCORE_ALIASES.items():
            score_col = first_present(df, by_horizon[horizon])
            if score_col is None:
                continue
            sub = heldout_frame(df, horizon, score_col)
            if len(sub) < 100:
                continue
            n_cal = original_cal_size(sub)
            p = sub[score_col].to_numpy(float)
            y = sub[y_col].to_numpy(int)
            for s in range(args.seeds):
                cal_idx, eval_idx = random_cal_eval_indices(len(sub), n_cal, args.base_seed + s)
                rec = split_conformal_metrics(p[cal_idx], y[cal_idx], p[eval_idx], y[eval_idx], args.alpha)
                rec.update(
                    {
                        "experiment": "four_score",
                        "horizon": horizon,
                        "score": SCORE_LABELS[score_key],
                        "score_col": score_col,
                        "seed": s,
                    }
                )
                rows.append(rec)

    raw = pd.DataFrame(rows)
    raw.to_csv(outdir / "four_score_runs.csv", index=False)
    summary = aggregate_records(rows, ["experiment", "horizon", "score", "score_col"])
    summary.to_csv(outdir / "four_score_summary.csv", index=False)

    if not raw.empty:
        plot_df = raw.groupby(["horizon", "score"], as_index=False)["coverage"].mean()
        save_bar_plot(plot_df, "score", "coverage", "horizon", "Marginal coverage by score", outdir / "score_comparison_coverage.png")
        width_df = raw.groupby(["horizon", "score"], as_index=False)["avg_size"].mean()
        save_bar_plot(width_df, "score", "avg_size", "horizon", "Average set size by score", outdir / "score_comparison_set_size.png")
    print(f"Wrote four-score outputs to {outdir}")
    return raw


def run_mondrian(args: argparse.Namespace) -> pd.DataFrame:
    df = load_master(args)
    outdir = ensure_outdir(args)
    score_col = require_col(df, SCORE_ALIASES["gbdt"]["1y"], "1-year GBDT score")
    country_col = require_col(df, ["Country", "country", "loancountry"], "country")
    sub = heldout_frame(df, "1y", score_col, [country_col])
    n_cal = original_cal_size(sub)
    p = sub[score_col].to_numpy(float)
    y = sub["y_1y"].to_numpy(int)
    groups = sub[country_col].astype(str).to_numpy()

    rows = []
    for s in range(args.seeds):
        cal_idx, eval_idx = random_cal_eval_indices(len(sub), n_cal, args.base_seed + s)
        q_pool = conformal_quantile(nonconformity_binary(p[cal_idx], y[cal_idx]), args.alpha)
        for country in sorted(pd.unique(groups[eval_idx])):
            ev = eval_idx[groups[eval_idx] == country]
            rec = set_metrics(p[ev], y[ev], q_pool)
            rec.update({"experiment": "mondrian", "method": "pooled", "country": country, "seed": s})
            rows.append(rec)

        q_by_country = {}
        for country in sorted(pd.unique(groups[cal_idx])):
            cal = cal_idx[groups[cal_idx] == country]
            q_by_country[country] = conformal_quantile(nonconformity_binary(p[cal], y[cal]), args.alpha)
        for country in sorted(pd.unique(groups[eval_idx])):
            ev = eval_idx[groups[eval_idx] == country]
            qhat = q_by_country.get(country, q_pool)
            rec = set_metrics(p[ev], y[ev], qhat)
            rec.update({"experiment": "mondrian", "method": "mondrian", "country": country, "seed": s})
            rows.append(rec)

    raw = pd.DataFrame(rows)
    raw.to_csv(outdir / "mondrian_runs.csv", index=False)
    summary = aggregate_records(rows, ["experiment", "method", "country"])
    summary.to_csv(outdir / "mondrian_summary.csv", index=False)
    plot_df = raw.groupby(["method", "country"], as_index=False)["coverage"].mean()
    save_bar_plot(plot_df, "country", "coverage", "method", "Per-country coverage", outdir / "group_conditional_coverage.png")
    print(f"Wrote Mondrian outputs to {outdir}")
    return raw


def run_shift_break(args: argparse.Namespace) -> pd.DataFrame:
    df = load_master(args)
    outdir = ensure_outdir(args)
    score_col = require_col(df, SCORE_ALIASES["gbdt"]["1y"], "1-year GBDT score")
    country_col = require_col(df, ["Country", "country", "loancountry"], "country")
    sub = heldout_frame(df, "1y", score_col, [country_col])
    if "split" not in sub.columns:
        raise SystemExit("The shift experiment requires a split column.")

    split = sub["split"].astype(str).str.lower()
    p = sub[score_col].to_numpy(float)
    y = sub["y_1y"].to_numpy(int)
    countries = sub[country_col].astype(str).to_numpy()
    cal_mask = split.isin(["cal", "calibration"]).to_numpy()
    test_mask = split.isin(["test", "evaluation", "eval"]).to_numpy()

    rows = []
    if cal_mask.any() and test_mask.any():
        q_pool = conformal_quantile(nonconformity_binary(p[cal_mask], y[cal_mask]), args.alpha)
        rec = set_metrics(p[test_mask], y[test_mask], q_pool)
        rec.update({"experiment": "shift_break", "axis": "temporal", "case": "pooled_cal_to_test"})
        rows.append(rec)

        ee_cal = cal_mask & (countries == "EE")
        ee_test = test_mask & (countries == "EE")
        if ee_cal.any() and ee_test.any():
            q_ee = conformal_quantile(nonconformity_binary(p[ee_cal], y[ee_cal]), args.alpha)
            rec = set_metrics(p[ee_test], y[ee_test], q_ee)
            rec.update({"experiment": "shift_break", "axis": "temporal", "case": "ee_cal_to_ee_test"})
            rows.append(rec)

    n_cal = original_cal_size(sub)
    random_cov = []
    for s in range(args.seeds):
        cal_idx, eval_idx = random_cal_eval_indices(len(sub), n_cal, args.base_seed + s)
        rec = split_conformal_metrics(p[cal_idx], y[cal_idx], p[eval_idx], y[eval_idx], args.alpha)
        random_cov.append(rec["coverage"])
    rows.append(
        {
            "experiment": "shift_break",
            "axis": "temporal",
            "case": "random_heldout_control",
            "n": int(len(sub) - n_cal),
            "coverage": float(np.mean(random_cov)),
            "coverage_std": float(np.std(random_cov, ddof=1)) if len(random_cov) > 1 else 0.0,
            "avg_size": np.nan,
            "ambiguity": np.nan,
            "singleton": np.nan,
            "empty": np.nan,
            "qhat": np.nan,
        }
    )

    ee = np.where(countries == "EE")[0]
    n_ee_cal = int(np.sum(cal_mask & (countries == "EE")))
    if len(ee) > 100 and 0 < n_ee_cal < len(ee):
        for s in range(args.seeds):
            ee_cal_rel, _ = random_cal_eval_indices(len(ee), n_ee_cal, args.base_seed + s)
            cal_idx = ee[ee_cal_rel]
            q_ee = conformal_quantile(nonconformity_binary(p[cal_idx], y[cal_idx]), args.alpha)
            eval_idx = np.setdiff1d(np.arange(len(sub)), cal_idx, assume_unique=False)
            for country in sorted(pd.unique(countries[eval_idx])):
                ev = eval_idx[countries[eval_idx] == country]
                rec = set_metrics(p[ev], y[ev], q_ee)
                rec.update(
                    {
                        "experiment": "shift_break",
                        "axis": "geography",
                        "case": f"ee_cal_to_{country}",
                        "seed": s,
                    }
                )
                rows.append(rec)

    raw = pd.DataFrame(rows)
    raw.to_csv(outdir / "shift_break_summary.csv", index=False)
    plot_df = raw[raw["axis"] == "temporal"].copy()
    if not plot_df.empty:
        plt = get_plt()
        ax = plot_df.plot(kind="bar", x="case", y="coverage", legend=False, figsize=(9, 4), rot=20)
        ax.axhline(0.9, color="black", ls="--", lw=1)
        ax.set_title("Coverage under temporal and random held-out evaluations")
        ax.set_ylabel("coverage")
        plt.tight_layout()
        save_current_transparent_figure(plt, outdir / "temporal_shift_coverage.png", dpi=160)
        plt.close()
    print(f"Wrote shift-diagnostic outputs to {outdir}")
    return raw


def run_online_adapt(args: argparse.Namespace) -> pd.DataFrame:
    df = load_master(args)
    outdir = ensure_outdir(args)
    score_col = require_col(df, SCORE_ALIASES["gbdt"]["1y"], "1-year GBDT score")
    date_col = require_col(df, ["LoanDate", "loandate", "loan_date", "origination_date"], "loan date")
    sub = heldout_frame(df, "1y", score_col, [date_col]).copy()
    sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")
    sub = sub.dropna(subset=[date_col]).sort_values(date_col)
    if len(sub) <= args.warmup + 100:
        raise SystemExit("Not enough rows after warmup for the online adaptation experiment.")

    p = sub[score_col].to_numpy(float)
    y = sub["y_1y"].to_numpy(int)
    scores = nonconformity_binary(p, y)
    q0 = conformal_quantile(scores[: args.warmup], args.alpha)

    records = []
    q_pid = q0
    alpha_aci = args.alpha
    integral = 0.0
    rolling_scores = list(scores[: args.warmup])

    for t in range(args.warmup, len(sub)):
        p_t = np.array([p[t]])
        y_t = np.array([y[t]])

        for method, qhat in [("fixed", q0), ("pid", q_pid)]:
            metrics = set_metrics(p_t, y_t, qhat)
            records.append({"t": t, "method": method, "covered": metrics["coverage"], "set_size": metrics["avg_size"]})

        q_aci = conformal_quantile(rolling_scores[-args.window :], alpha_aci)
        metrics_aci = set_metrics(p_t, y_t, q_aci)
        records.append({"t": t, "method": "aci", "covered": metrics_aci["coverage"], "set_size": metrics_aci["avg_size"]})

        err_aci = 1.0 - metrics_aci["coverage"]
        alpha_aci = float(np.clip(alpha_aci + args.eta_aci * (args.alpha - err_aci), 0.001, 0.999))

        pid_metrics = set_metrics(p_t, y_t, q_pid)
        err_pid = 1.0 - pid_metrics["coverage"]
        integral += err_pid - args.alpha
        q_pid = float(np.clip(q_pid + args.eta_p * (err_pid - args.alpha) + args.eta_i * integral, 0.0, 1.0))

        rolling_scores.append(scores[t])

    stream = pd.DataFrame(records)
    summary = (
        stream.groupby("method")
        .agg(n=("covered", "size"), coverage=("covered", "mean"), avg_size=("set_size", "mean"))
        .reset_index()
    )
    stream.to_csv(outdir / "online_adapt_stream.csv", index=False)
    summary.to_csv(outdir / "online_adapt_summary.csv", index=False)

    roll = (
        stream.assign(block=stream["t"] // args.roll)
        .groupby(["method", "block"], as_index=False)
        .agg(coverage=("covered", "mean"))
    )
    plt = get_plt()
    fig, ax = plt.subplots(figsize=(9, 4))
    for method, part in roll.groupby("method"):
        ax.plot(part["block"], part["coverage"], marker="o", label=method)
    ax.axhline(0.9, color="black", ls="--", lw=1)
    ax.set_title("Rolling coverage under online threshold updates")
    ax.set_xlabel("time block")
    ax.set_ylabel("coverage")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    save_current_transparent_figure(plt, outdir / "online_adaptation_coverage.png", dpi=160)
    plt.close()
    print(f"Wrote online-adaptation outputs to {outdir}")
    return summary


def run_label_robust(args: argparse.Namespace) -> pd.DataFrame:
    df = load_master(args)
    outdir = ensure_outdir(args)
    score_col = require_col(df, SCORE_ALIASES["gbdt"]["lifetime"], "lifetime GBDT score")
    label_cols = {label: require_col(df, aliases, label) for label, aliases in LABEL_ALIASES.items()}
    sub = heldout_frame(df, "lifetime", score_col, list(label_cols.values()))
    sub = sub.dropna(subset=list(label_cols.values()))
    n_cal = original_cal_size(sub)
    p = sub[score_col].to_numpy(float)

    rows = []
    for s in range(args.seeds):
        cal_idx, eval_idx = random_cal_eval_indices(len(sub), n_cal, args.base_seed + s)
        q_by_label = {}
        for label, col in label_cols.items():
            y_label = sub[col].to_numpy(int)
            q_by_label[label] = conformal_quantile(nonconformity_binary(p[cal_idx], y_label[cal_idx]), args.alpha)
        q_l3 = q_by_label["L3"]
        q_worst = max(q_by_label.values())
        for label, col in label_cols.items():
            y_eval = sub[col].to_numpy(int)[eval_idx]
            for method, qhat in [("single_label_L3", q_l3), ("label_robust", q_worst)]:
                rec = set_metrics(p[eval_idx], y_eval, qhat)
                rec.update({"experiment": "label_robust", "method": method, "label": label, "seed": s})
                rows.append(rec)

    raw = pd.DataFrame(rows)
    raw.to_csv(outdir / "label_robust_runs.csv", index=False)
    summary = aggregate_records(rows, ["experiment", "method", "label"])
    summary.to_csv(outdir / "label_robust_summary.csv", index=False)
    plot_df = raw.groupby(["method", "label"], as_index=False)["coverage"].mean()
    save_bar_plot(plot_df, "label", "coverage", "method", "Label-robust coverage", outdir / "label_robustness_coverage.png")
    print(f"Wrote label-robust outputs to {outdir}")
    return raw


def parse_q_grid(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def run_selection(args: argparse.Namespace) -> pd.DataFrame:
    df = load_master(args)
    outdir = ensure_outdir(args)
    q_grid = parse_q_grid(args.q_grid)
    rows = []

    for horizon in ("1y", "lifetime"):
        y_col = target_col(horizon)
        if y_col not in df.columns:
            continue
        score_col = first_present(df, SCORE_ALIASES["gbdt"][horizon])
        if score_col is None:
            continue
        sub = heldout_frame(df, horizon, score_col)
        if len(sub) < 100:
            continue
        n_cal = original_cal_size(sub)
        p = sub[score_col].to_numpy(float)
        y = sub[y_col].to_numpy(int)
        for s in range(args.seeds):
            cal_idx, eval_idx = random_cal_eval_indices(len(sub), n_cal, args.base_seed + s)
            pvals = conformal_pvalues_for_good_loans(p[cal_idx], y[cal_idx], p[eval_idx])
            y_eval = y[eval_idx]
            n_good = int(np.sum(y_eval == 0))
            for q in q_grid:
                selected = bh_select(pvals, q)
                n_sel = int(selected.sum())
                if n_sel:
                    fdr = float(np.mean(y_eval[selected] == 1))
                    power = float(np.sum((y_eval[selected] == 0)) / max(n_good, 1))
                else:
                    fdr = 0.0
                    power = 0.0
                rows.append(
                    {
                        "experiment": "selection",
                        "horizon": horizon,
                        "seed": s,
                        "q": q,
                        "n_eval": int(len(eval_idx)),
                        "n_selected": n_sel,
                        "approval_rate": n_sel / len(eval_idx),
                        "realized_fdr": fdr,
                        "power": power,
                    }
                )

    raw = pd.DataFrame(rows)
    raw.to_csv(outdir / "selection_runs.csv", index=False)
    summary = aggregate_records(rows, ["experiment", "horizon", "q"])
    summary.to_csv(outdir / "selection_summary.csv", index=False)
    if not raw.empty:
        plt = get_plt()
        fig, ax = plt.subplots(figsize=(8, 4))
        for horizon, part in raw.groupby("horizon"):
            s = part.groupby("q", as_index=False)["realized_fdr"].mean()
            ax.plot(s["q"], s["realized_fdr"], marker="o", label=horizon)
        ax.plot(q_grid, q_grid, color="black", ls="--", label="FDR = q")
        ax.set_title("Conformal selection: realized FDR")
        ax.set_xlabel("target FDR q")
        ax.set_ylabel("realized FDR")
        ax.legend()
        ax.grid(alpha=0.25)
        plt.tight_layout()
        save_current_transparent_figure(plt, outdir / "conformal_selection_fdr.png", dpi=160)
        plt.close()
    print(f"Wrote conformal-selection outputs to {outdir}")
    return raw


def run_tabpfn_native(args: argparse.Namespace) -> pd.DataFrame:
    df = load_master(args)
    outdir = ensure_outdir(args)
    rows = []

    for horizon in ("1y", "lifetime"):
        y_col = target_col(horizon)
        score_col = first_present(df, SCORE_ALIASES["tabpfn"][horizon])
        if score_col is None or y_col not in df.columns:
            continue
        sub = heldout_frame(df, horizon, score_col)
        if len(sub) < 100:
            continue
        n_cal = original_cal_size(sub)
        p = sub[score_col].to_numpy(float)
        y = sub[y_col].to_numpy(int)
        for s in range(args.seeds):
            cal_idx, eval_idx = random_cal_eval_indices(len(sub), n_cal, args.base_seed + s)
            p_eval = p[eval_idx]
            y_eval = y[eval_idx]

            in1 = p_eval >= 0.5
            in0 = ~in1
            rec = metric_from_set_columns(in0, in1, y_eval)
            rec.update({"experiment": "tabpfn_native", "horizon": horizon, "method": "native_map", "seed": s})
            rows.append(rec)

            in1 = p_eval >= args.alpha
            in0 = (1.0 - p_eval) >= args.alpha
            rec = metric_from_set_columns(in0, in1, y_eval)
            rec.update({"experiment": "tabpfn_native", "horizon": horizon, "method": "native_threshold", "seed": s})
            rows.append(rec)

            rec = split_conformal_metrics(p[cal_idx], y[cal_idx], p_eval, y_eval, args.alpha)
            rec.update({"experiment": "tabpfn_native", "horizon": horizon, "method": "split_conformal", "seed": s})
            rows.append(rec)

    raw = pd.DataFrame(rows)
    raw.to_csv(outdir / "tabpfn_native_runs.csv", index=False)
    summary = aggregate_records(rows, ["experiment", "horizon", "method"])
    summary.to_csv(outdir / "tabpfn_native_summary.csv", index=False)
    if not raw.empty:
        plot_df = raw.groupby(["method", "horizon"], as_index=False)["coverage"].mean()
        save_bar_plot(plot_df, "method", "coverage", "horizon", "TabPFN native and conformalized sets", outdir / "tabpfn_conformal_coverage.png")
    print(f"Wrote TabPFN outputs to {outdir}")
    return raw


# ======================================================================
# Slide-used Deck 2 figures
# ======================================================================

def _ensure_deck_plotting():
    global matplotlib, plt
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


# Shared slide style
SCORE   = {"PoD": "#2E6FB0", "LR": "#E2861E", "LightGBM": "#4A9D5B", "TabPFN-3": "#7B5EA7"}
STATUS  = {"good": "#4A9D5B", "broken": "#C8503A", "neutral": "#9AA0A6"}
COUNTRY = {"EE": "#2E6FB0", "FI": "#E2861E", "ES": "#C8503A"}
TARGET  = "#B22222"

def apply_style():
    _ensure_deck_plotting()
    matplotlib.rcParams.update({
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 15, "axes.titlesize": 17, "axes.titleweight": "bold",
        "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 13, "legend.fontsize": 13,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": "#E8E8E8", "grid.linewidth": 0.8, "axes.edgecolor": "#5A5A5A",
    })

def target_line(ax, y=0.90, label="target 0.90"):
    ax.axhline(y, ls="--", lw=1.4, color=TARGET, zorder=1)
    ax.text(0.015, 0.965, label, color=TARGET, fontsize=11.5, va="top",
            ha="left", transform=ax.transAxes)

def caption(fig, text, y=0.012):
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=12.5,
             style="italic", color="#555555", wrap=True)

def _save(fig, outdir, name):
    p = Path(outdir) / name
    save_transparent_figure(fig, p, dpi=200)
    plt.close(fig)
    print(f"  wrote {p}")

# ======================================================================
# LOCKED-NUMBER FIGURES (S6, S8, S9, S14, S17)
# ======================================================================
def fig_s6(outdir):
    apply_style()
    sc = ["PoD", "LR", "LightGBM", "TabPFN-3"]; col = [SCORE[s] for s in sc]
    cov1 = [0.899, 0.899, 0.897, 0.898]; covL = [0.900, 0.898, 0.899, 0.899]
    ac1 = [1.253, 1.212, 1.209, 1.194];  am1 = [0.253, 0.212, 0.209, 0.194]
    acL = [1.526, 1.386, 1.393, 1.486];  amL = [0.526, 0.386, 0.393, 0.486]
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.8)); x = np.arange(4)
    for a, cov, t in [(ax[0, 0], cov1, "1-year   (n = 76,369)"),
                      (ax[0, 1], covL, "lifetime   (n = 69,978)")]:
        b = a.bar(x, cov, color=col, width=0.62, edgecolor="white"); target_line(a)
        a.set_ylim(0.84, 0.96); a.set_title(t); a.set_xticks(x); a.set_xticklabels(sc)
        for bb, c in zip(b, cov):
            a.text(bb.get_x()+bb.get_width()/2, c+0.0015, f"{c:.3f}", ha="center",
                   va="bottom", fontsize=13, fontweight="bold")
    ax[0, 0].set_ylabel("coverage")
    for a, acv, am in [(ax[1, 0], ac1, am1), (ax[1, 1], acL, amL)]:
        b = a.bar(x, acv, color=col, width=0.62, edgecolor="white"); a.set_ylim(1.0, 1.85)
        a.set_xticks(x); a.set_xticklabels(sc)
        for bb, v, m in zip(b, acv, am):
            a.text(bb.get_x()+bb.get_width()/2, v+0.012, f"{v:.2f}\namb {m:.2f}",
                   ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax[1, 0].set_ylabel("avg |C|  (lower = sharper)")
    fig.suptitle("Split conformal: coverage is equalized, width is not", fontsize=20, fontweight="bold", y=0.985)
    caption(fig, "Conformal equalizes coverage at ~0.90 across all four scores; set width tracks each score's sharpness — PoD pays the widest sets, most at lifetime.")
    fig.tight_layout(rect=[0, 0.04, 1, 0.95]); _save(fig, outdir, "split_conformal_coverage_width.png")

def fig_s8(outdir):
    apply_style()
    ctry = ["EE", "FI", "ES"]; col = [COUNTRY[c] for c in ctry]
    plain = [0.9144, 0.9080, 0.8439]
    mond = [0.9022, 0.9012, 0.9007]; mC = [1.099, 1.102, 1.731]; mA = [0.099, 0.102, 0.731]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.8)); x = np.arange(3)
    b = ax[0].bar(x, plain, color=col, width=0.6, edgecolor="white"); target_line(ax[0])
    ax[0].set_ylim(0.70, 1.00); ax[0].set_title("Plain conformal (one pooled q\u0302)")
    ax[0].set_xticks(x); ax[0].set_xticklabels(ctry); ax[0].set_ylabel("per-country coverage")
    for i, (bb, v) in enumerate(zip(b, plain)):
        bad = ctry[i] == "ES"
        ax[0].text(bb.get_x()+bb.get_width()/2, v+0.006, f"{v:.3f}", ha="center", va="bottom",
                   fontsize=13, fontweight="bold", color=(STATUS["broken"] if bad else "#333333"))
    b = ax[1].bar(x, mond, color=col, width=0.6, edgecolor="white"); target_line(ax[1])
    ax[1].set_ylim(0.70, 1.00); ax[1].set_title("Mondrian (per-country q\u0302) — coverage restored")
    ax[1].set_xticks(x); ax[1].set_xticklabels(ctry)
    for i, (bb, v, c, m) in enumerate(zip(b, mond, mC, mA)):
        bad = ctry[i] == "ES"
        ax[1].text(bb.get_x()+bb.get_width()/2, v+0.006, f"|C|={c:.2f}\namb {m:.2f}", ha="center",
                   va="bottom", fontsize=12, fontweight="bold", color=(STATUS["broken"] if bad else "#444444"))
    fig.suptitle("Pooled 90% hides Spain at 84%; Mondrian buys it back — at ES amb 0.73", fontsize=19, fontweight="bold", y=0.99)
    caption(fig, "The hardest market pays the widest sets not because it is smallest (8K cal is ample), but because the score is least able to commit there.   base score = LightGBM, \u03b1 = 0.1")
    fig.tight_layout(rect=[0, 0.05, 1, 0.94]); _save(fig, outdir, "group_conditional_coverage.png")

def fig_s9(outdir):
    apply_style()
    G = ["EE\u2192EE", "EE\u2192FI", "EE\u2192ES"]; gv = [0.902, 0.892, 0.819]
    gc = [STATUS["good"], STATUS["good"], STATUS["broken"]]
    M = ["ALL\nrandom", "ALL\nearly\u2192late", "EE\nearly\u2192late"]; mv = [0.902, 0.847, 0.910]
    mc = [STATUS["neutral"], STATUS["broken"], STATUS["good"]]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.8)); x = np.arange(3)
    b = ax[0].bar(x, gv, color=gc, width=0.6, edgecolor="white"); target_line(ax[0])
    ax[0].set_ylim(0.70, 1.00); ax[0].set_title("Geography — EE-calibrated, deployed per market", fontsize=15)
    ax[0].set_xticks(x); ax[0].set_xticklabels(G); ax[0].set_ylabel("coverage")
    for bb, v in zip(b, gv):
        ax[0].text(bb.get_x()+bb.get_width()/2, v+0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax[0].text(2, 0.862, "\u22128 pp", ha="center", va="bottom", fontsize=15, fontweight="bold", color=STATUS["broken"])
    b = ax[1].bar(x, mv, color=mc, width=0.6, edgecolor="white"); target_line(ax[1])
    ax[1].set_ylim(0.70, 1.00); ax[1].set_title("Calendar? Isolate EE — the break vanishes", fontsize=15)
    ax[1].set_xticks(x); ax[1].set_xticklabels(M)
    for bb, v in zip(b, mv):
        ax[1].text(bb.get_x()+bb.get_width()/2, v+0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax[1].text(1, 0.872, "\u22125 pp", ha="center", va="bottom", fontsize=15, fontweight="bold", color=STATUS["broken"])
    ax[1].text(2, 0.93, "calendar\nruled out", ha="center", va="bottom", fontsize=12, fontweight="bold", color=STATUS["good"])
    fig.suptitle("Two observable breaks, one mechanism: the market moved, not the clock", fontsize=19, fontweight="bold", y=0.99)
    caption(fig, "EE\u2192ES collapses (ES is the high-risk market); EE\u2192FI barely moves. Deploying early\u2192late drops pooled coverage — but isolate Estonia and it returns to 0.91.   base score = LightGBM, \u03b1 = 0.1")
    fig.subplots_adjust(wspace=0.16); fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    _save(fig, outdir, "distribution_shift_diagnostic.png")

def fig_s14(outdir):
    apply_style()
    L = ["L1", "L2", "L3", "L4", "L5"]
    l3 = [0.808, 0.829, 0.899, 0.890, 0.878]; worst = [0.900, 0.911, 0.958, 0.952, 0.947]
    lc = [STATUS["broken"], STATUS["broken"], STATUS["good"], STATUS["broken"], STATUS["broken"]]
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 5.2), gridspec_kw={"width_ratios": [1, 1, 0.55]}); x = np.arange(5)
    b = ax[0].bar(x, l3, color=lc, width=0.62, edgecolor="white"); target_line(ax[0])
    ax[0].set_ylim(0.78, 1.00); ax[0].set_title("Calibrate to L3 only")
    ax[0].set_xticks(x); ax[0].set_xticklabels(L); ax[0].set_ylabel("coverage under each label")
    for bb, v in zip(b, l3):
        bad = v < 0.895
        ax[0].text(bb.get_x()+bb.get_width()/2, v+0.004, f"{v:.3f}", ha="center", va="bottom",
                   fontsize=12.5, fontweight="bold", color=(STATUS["broken"] if bad else "#333333"))
    b = ax[1].bar(x, worst, color=SCORE["PoD"], width=0.62, edgecolor="white"); target_line(ax[1])
    ax[1].set_ylim(0.78, 1.00); ax[1].set_title("Worst-case set (q=max, driven by L1)\nvalid under EVERY label")
    ax[1].set_xticks(x); ax[1].set_xticklabels(L)
    for bb, v in zip(b, worst):
        ax[1].text(bb.get_x()+bb.get_width()/2, v+0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=12.5, fontweight="bold")
    wb = ax[2].bar([0, 1], [1.39, 1.63], color=[STATUS["neutral"], STATUS["broken"]], width=0.6, edgecolor="white")
    ax[2].set_ylim(1.0, 1.8); ax[2].set_title("width price"); ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["L3-only", "worst"])
    ax[2].set_ylabel("avg |C|")
    for bb, v, m in zip(wb, [1.39, 1.63], [0.39, 0.63]):
        ax[2].text(bb.get_x()+bb.get_width()/2, v+0.012, f"{v:.2f}\namb {m:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    fig.suptitle("Label-robust conformal: the price of being valid under all five definitions", fontsize=19, fontweight="bold", y=0.99)
    caption(fig, "Calibrating to L3 alone silently under-covers the labels the score under-predicts (L1: 0.81); q=max over L1–L5 restores \u22650.90 under every definition, at +0.24 avg|C|.")
    fig.tight_layout(rect=[0, 0.05, 1, 0.93]); _save(fig, outdir, "label_robustness_coverage.png")

def fig_s17(outdir):
    apply_style()
    lab = ["native\n(MAP)", "native\n(APS)", "wrapped\n(conformal)"]
    col = [STATUS["neutral"], STATUS["broken"], SCORE["TabPFN-3"]]
    cov1 = [0.818, 0.979, 0.898]; ac1 = [1.00, 1.65, 1.19]; am1 = [0.00, 0.65, 0.19]
    covL = [0.687, 0.998, 0.899]; acL = [1.00, 1.95, 1.49]; amL = [0.00, 0.95, 0.49]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.2)); x = np.arange(3)
    for a, cov, t in [(ax[0, 0], cov1, "1-year   (n = 76,369, base 0.187)"),
                      (ax[0, 1], covL, "lifetime   (n = 69,978, base 0.381)")]:
        b = a.bar(x, cov, color=col, width=0.6, edgecolor="white"); target_line(a)
        a.set_ylim(0.62, 1.02); a.set_title(t); a.set_xticks(x); a.set_xticklabels(lab)
        for bb, c in zip(b, cov):
            a.text(bb.get_x()+bb.get_width()/2, c+0.006, f"{c:.3f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax[0, 0].set_ylabel("coverage")
    for a, acv, am in [(ax[1, 0], ac1, am1), (ax[1, 1], acL, amL)]:
        b = a.bar(x, acv, color=col, width=0.6, edgecolor="white"); a.set_ylim(1.0, 2.15)
        a.axhline(1.0, ls=":", lw=1.0, color="#999999"); a.set_xticks(x); a.set_xticklabels(lab)
        for bb, v, m in zip(b, acv, am):
            a.text(bb.get_x()+bb.get_width()/2, v+0.02, f"{v:.2f}\namb {m:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax[1, 0].set_ylabel("avg |C|  (1 = singleton)")
    fig.suptitle("TabPFN's own uncertainty is not a valid set — wrapping it is", fontsize=19, fontweight="bold", y=0.985)
    caption(fig, "Trusting the point prediction under-covers (0.82 / 0.69); reading the probability at the nominal level over-covers by abstaining; the same scores, conformal-wrapped, hit 0.90 at honest width.")
    fig.tight_layout(rect=[0, 0.04, 1, 0.95]); _save(fig, outdir, "tabpfn_conformal_coverage.png")

# ======================================================================
# DATA-DEPENDENT FIGURES (S11, S16) -- algorithms ported from the run scripts
# ======================================================================
# --- inlined conformal_core helpers ---
def slide_nonconformity_binary(p, y):
    p = np.asarray(p, float); y = np.asarray(y, int)
    return np.where(y == 1, 1.0 - p, p)

def slide_conformal_quantile(s, alpha):
    s = np.asarray(s, float); n = s.size
    lvl = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(s, lvl, method="higher"))

def _resolve(df, aliases):
    lo = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a in lo: return lo[a]
    for a in aliases:
        for lc, real in lo.items():
            if a in lc: return real
    return None

# ---------- S11 ----------
_S11_AL = {
    "score": ["pred_lightgbm_1y", "pred_lgb_1y", "lightgbm_1y"],
    "y": ["y_1y", "default_1y", "label_1y"],
    "country": ["country", "loancountry", "ctry"],
    "loandate": ["loandate", "loan_date", "origination_date", "origdate", "listeddate", "issuedate"],
    "eligible": ["conformal_eligible_1y", "observed_1y_universe", "eligible_1y"],
}
def _covsize(p_t, y_t, q_t):
    c0 = p_t <= q_t; c1 = (1.0 - p_t) <= q_t
    return int(c1 if y_t == 1 else c0), int(c0) + int(c1)

def _run_plain(p, y, q0):
    cov = np.empty(len(p)); sz = np.empty(len(p))
    for t in range(len(p)): cov[t], sz[t] = _covsize(p[t], y[t], q0)
    return cov, sz

def _run_aci(p, y, s_warm, alpha, eta, window):
    buf = list(s_warm[-window:]); a_t = alpha
    cov = np.empty(len(p)); sz = np.empty(len(p))
    for t in range(len(p)):
        lvl = min(max(1.0 - a_t, 0.0), 1.0)
        q_t = float(np.quantile(buf, lvl, method="higher")) if buf else 1.0
        c, s = _covsize(p[t], y[t], q_t); cov[t], sz[t] = c, s
        a_t = min(max(a_t + eta * (alpha - (1 - c)), 1e-4), 1 - 1e-4)
        buf.append(float(slide_nonconformity_binary([p[t]], [y[t]])[0]))
        if len(buf) > window: buf.pop(0)
    return cov, sz

def _run_pid(p, y, q0, alpha, eta_p, eta_i, i_window=500):
    q_t = q0; hist = []
    cov = np.empty(len(p)); sz = np.empty(len(p))
    for t in range(len(p)):
        c, s = _covsize(p[t], y[t], q_t); cov[t], sz[t] = c, s
        err = 1 - c; hist.append(err - alpha)
        if len(hist) > i_window: hist.pop(0)
        q_t = min(max(q_t + eta_p * (err - alpha) + eta_i * float(np.mean(hist)), 0.0), 1.0)
    return cov, sz

def _rolling(a, w):
    a = np.asarray(a, float)
    return a if w <= 1 else np.convolve(a, np.ones(w) / w, mode="valid")

def _fies_bands(ctry, roll, thresh=0.15):
    share = _rolling(np.isin(ctry, ["FI", "ES"]).astype(float), roll)
    hot = share > thresh; spans = []; i = 0; n = len(hot)
    while i < n:
        if hot[i]:
            j = i
            while j < n and hot[j]: j += 1
            spans.append((i, j)); i = j
        else: i += 1
    return spans

def fig_s11(master, outdir, alpha=0.1, warmup=8000, window=2000, roll=2000,
            eta_aci=0.02, eta_p=0.05, eta_i=0.002):
    import pandas as pd
    apply_style()
    mp = Path(master)
    df = pd.read_parquet(mp) if mp.suffix == ".parquet" else pd.read_csv(mp)
    sc = _resolve(df, _S11_AL["score"]); yc = _resolve(df, _S11_AL["y"])
    cc = _resolve(df, _S11_AL["country"]); dc = _resolve(df, _S11_AL["loandate"])
    ec = _resolve(df, _S11_AL["eligible"])
    if None in (sc, yc, cc, dc, ec):
        sys.exit(f"[S11] missing column: score={sc} y={yc} country={cc} date={dc} elig={ec}")
    e = df[df[ec].astype(bool)].copy()
    e[dc] = pd.to_datetime(e[dc], errors="coerce")
    e = e.dropna(subset=[dc]).sort_values(dc).reset_index(drop=True)
    p = e[sc].to_numpy(float); y = e[yc].to_numpy(int); ctry = e[cc].astype(str).to_numpy()
    W = warmup
    if len(e) <= W + 1000: sys.exit("[S11] not enough rows; lower --warmup.")
    s_warm = slide_nonconformity_binary(p[:W], y[:W]); q0 = slide_conformal_quantile(s_warm, alpha)
    p_s, y_s, c_s = p[W:], y[W:], ctry[W:]; idx = np.arange(len(p_s))
    cov_p, sz_p = _run_plain(p_s, y_s, q0)
    cov_a, sz_a = _run_aci(p_s, y_s, s_warm, alpha, eta_aci, window)
    cov_i, sz_i = _run_pid(p_s, y_s, q0, alpha, eta_p, eta_i)
    covc = {"plain": _rolling(cov_p, roll), "ACI": _rolling(cov_a, roll), "PID": _rolling(cov_i, roll)}
    szc = {"plain": _rolling(sz_p, roll), "ACI": _rolling(sz_a, roll), "PID": _rolling(sz_i, roll)}
    bands = _fies_bands(c_s, roll)
    style = {"plain": (STATUS["broken"], "-"), "ACI": (STATUS["good"], "--"), "PID": (SCORE["PoD"], "-")}
    fig, ax = plt.subplots(2, 1, figsize=(13, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.1, 0.9]})
    for sp in bands:
        for a in ax: a.axvspan(idx[sp[0]], idx[min(sp[1], len(idx)-1)], color="#EDEDED", zorder=0)
    for name, c in covc.items():
        col, ls = style[name]; ax[0].plot(idx[:len(c)], c, lw=2.0, color=col, ls=ls, label=name)
    target_line(ax[0]); ax[0].set_ylim(0.70, 1.00); ax[0].set_ylabel("rolling coverage")
    ax[0].legend(loc="lower left", ncol=3, frameon=False)
    ax[0].set_title("Coverage as a control problem: plain drifts, online tracks back to 0.90", fontsize=15)
    for name, s in szc.items():
        col, ls = style[name]; ax[1].plot(idx[:len(s)], s, lw=2.0, color=col, ls=ls)
    ax[1].set_ylim(1.0, 2.0); ax[1].set_ylabel("rolling avg |C|")
    ax[1].set_xlabel("loan stream, ordered by origination date   (grey = high FI/ES-share epoch)")
    fig.suptitle("Online conformal: coverage repaired, the width curve is the price", fontsize=19, fontweight="bold", y=0.98)
    caption(fig, "plain drifts below 0.90 where the market moves; ACI / PID track back to 0.90 (curves coincide). Narrow-and-wrong vs wide-and-honest.")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]); _save(fig, outdir, "online_adaptation_coverage.png")

# ---------- S16 ----------
_S16_H = {
    "lifetime": {"p1": ["pred_lightgbm_lifetime", "pred_lgb_lifetime", "pred_lr_lifetime"],
                 "y": ["y_lifetime", "L3", "y_l3"], "elig": ["conformal_eligible_lifetime"]},
    "1y": {"p1": ["pred_lightgbm_1y", "pred_lgb_1y", "pred_lr_1y"],
           "y": ["y_1y", "label_1y"], "elig": ["conformal_eligible_1y"]},
}
_Q_GRID = [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30]
_Q_MARK = [0.10, 0.20]; _N_SEEDS = 5; _SEED0 = 20240617

def _conf_pvalues(p_cal, y_cal, p_test):
    ds = np.sort(p_cal[y_cal == 1]); n = ds.size
    if n == 0: return np.ones_like(p_test)
    return (1.0 + np.searchsorted(ds, p_test, side="right")) / (1.0 + n)

def _bh(pvals, q):
    m = pvals.size; order = np.argsort(pvals); sp = pvals[order]
    below = sp <= (np.arange(1, m + 1) / m) * q
    if not below.any(): return np.zeros(m, dtype=bool)
    k = np.max(np.nonzero(below)[0]) + 1
    mask = np.zeros(m, dtype=bool); mask[order[:k]] = True; return mask

def _s16_horizon(df, split_col, horizon, cols):
    p1c = _resolve(df, cols["p1"]); yc = _resolve(df, cols["y"]); ec = _resolve(df, cols["elig"])
    if None in (p1c, yc, ec):
        print(f"  [S16/{horizon}] skipped (p1={p1c} y={yc} elig={ec})"); return []
    pool = df[df[ec].astype(bool) & df[split_col].isin(["cal", "test"])].copy()
    p1 = pool[p1c].to_numpy(float); y = pool[yc].to_numpy(int)
    n = len(pool); cal_size = int((pool[split_col] == "cal").sum()); base = float(y.mean())
    acc = {q: {"fdr": [], "power": []} for q in _Q_GRID}
    rng = np.random.default_rng(_SEED0)
    for _ in range(_N_SEEDS):
        idx = rng.permutation(n); ci, ti = idx[:cal_size], idx[cal_size:]
        pv = _conf_pvalues(p1[ci], y[ci], p1[ti]); yt = y[ti]; ng = int((yt == 0).sum())
        for q in _Q_GRID:
            sel = _bh(pv, q); ns = int(sel.sum())
            acc[q]["fdr"].append((((yt == 1) & sel).sum() / ns) if ns else 0.0)
            acc[q]["power"].append((((yt == 0) & sel).sum() / ng) if ng else 0.0)
    return [{"horizon": horizon, "base_rate": base, "q": q,
             "fdr": float(np.mean(acc[q]["fdr"])), "power": float(np.mean(acc[q]["power"]))}
            for q in _Q_GRID]

def fig_s16(master, outdir, preds=None):
    import pandas as pd
    apply_style()
    mp = Path(master)
    df = pd.read_parquet(mp) if mp.suffix in (".parquet", ".pq") else pd.read_csv(mp)
    if preds and Path(preds).exists():
        pr = pd.read_csv(preds)
        idc = next((c for c in ("LoanId", "loan_id", "id") if c in df.columns and c in pr.columns), None)
        if idc:
            add = [c for c in pr.columns if c not in df.columns]
            if add: df = df.merge(pr[[idc] + add], on=idc, how="left")
    split_col = _resolve(df, ["split", "fold", "split_role"])
    if split_col is None: sys.exit("[S16] no split column.")
    rows = []
    for h, cols in _S16_H.items(): rows += _s16_horizon(df, split_col, h, cols)
    if not rows: sys.exit("[S16] no horizon produced results.")
    C = {"1y": ("#C0504D", "s", "--"), "lifetime": (SCORE["PoD"], "o", "-")}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
    qmin, qmax = min(_Q_GRID), max(_Q_GRID)
    ax[0].plot([qmin, qmax], [qmin, qmax], color="#555555", lw=1.2)
    ax[0].text(qmax, qmax, " FDR = q", fontsize=11, color="#555555", va="center")
    for h in sorted({r["horizon"] for r in rows}):
        rs = sorted([r for r in rows if r["horizon"] == h], key=lambda x: x["q"])
        q = np.array([r["q"] for r in rs]); fdr = np.array([r["fdr"] for r in rs])
        pw = np.array([r["power"] for r in rs]); base = rs[0]["base_rate"]
        col, mk, ls = C.get(h, ("gray", "o", "-")); lbl = f"{h} (base {base:.2f})"
        ax[0].plot(q, fdr, ls, marker=mk, color=col, lw=2, ms=6, label=lbl)
        ax[1].plot(q, pw, ls, marker=mk, color=col, lw=2, ms=6, label=lbl)
        ax[0].axhline(base, ls=":", lw=1, color=col, alpha=0.7)
        for qm in _Q_MARK:
            r = next(x for x in rs if abs(x["q"] - qm) < 1e-9)
            ax[0].scatter([qm], [r["fdr"]], s=170, facecolors="none", edgecolors=col, lw=2, zorder=5)
            ax[1].scatter([qm], [r["power"]], s=170, facecolors="none", edgecolors=col, lw=2, zorder=5)
    ax[0].set_xlabel("target FDR level  q"); ax[0].set_ylabel("realized FDR")
    ax[0].set_title("FDR is controlled (stays under the q line)", fontsize=15)
    ax[0].legend(loc="upper left", frameon=False)
    ax[1].set_xlabel("target FDR level  q"); ax[1].set_ylabel("power (good approved / all good)")
    ax[1].set_title("Power is the price: higher q \u2192 more approvals", fontsize=15)
    ax[1].legend(loc="upper left", frameon=False)
    fig.suptitle("Conformal selection: FDR-controlled approval, q is an auditable dial", fontsize=19, fontweight="bold", y=0.99)
    caption(fig, "q is a risk-appetite dial, not a data truth. Dotted = base default rate; circles mark q = 0.10 / 0.20. Realized FDR runs ~q/3–q/2.")
    fig.tight_layout(rect=[0, 0.04, 1, 0.93]); _save(fig, outdir, "conformal_selection_fdr.png")

def run_slide_figures(args: argparse.Namespace) -> None:
    """Write only the presentation figures used in the final slides."""
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig_s6(outdir)
    fig_s8(outdir)
    fig_s9(outdir)
    fig_s14(outdir)
    fig_s17(outdir)

    if args.master:
        fig_s11(
            args.master,
            outdir,
            args.alpha,
            args.warmup,
            args.window,
            args.roll,
            args.eta_aci,
            args.eta_p,
            args.eta_i,
        )
        fig_s16(args.master, outdir, getattr(args, "preds", None))
    else:
        print("No --master provided; skipped online_adaptation_coverage.png and conformal_selection_fdr.png.")


def run_all(args: argparse.Namespace) -> None:
    for func in [
        run_four_score,
        run_mondrian,
        run_shift_break,
        run_online_adapt,
        run_label_robust,
        run_selection,
        run_tabpfn_native,
    ]:
        try:
            func(args)
        except SystemExit as err:
            print(f"Skipped {func.__name__}: {err}")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--master", default="data/conformal_master.csv")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--outdir", default="figures/conformal")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=8000)
    parser.add_argument("--window", type=int, default=2000)
    parser.add_argument("--roll", type=int, default=2000)
    parser.add_argument("--eta-aci", type=float, default=0.02)
    parser.add_argument("--eta-p", type=float, default=0.05)
    parser.add_argument("--eta-i", type=float, default=0.002)
    parser.add_argument("--q-grid", default="0.05,0.075,0.10,0.125,0.15,0.175,0.20,0.25,0.30")
    parser.add_argument("--preds", default=None, help="Optional step5_predictions.csv used by slide selection diagnostics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Bondora conformal-inference experiments.")
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {
        "four-score": run_four_score,
        "mondrian": run_mondrian,
        "shift-break": run_shift_break,
        "online-adapt": run_online_adapt,
        "label-robust": run_label_robust,
        "selection": run_selection,
        "tabpfn-native": run_tabpfn_native,
        "slides": run_slide_figures,
        "all": run_all,
    }
    for name, func in commands.items():
        p = sub.add_parser(name)
        add_common_args(p)
        p.set_defaults(func=func)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
