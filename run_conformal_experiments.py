"""Run conformal-inference experiments for the Bondora credit-risk analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from conformal_core import (
    aggregate_records,
    bh_select,
    conformal_pvalues_for_good_loans,
    conformal_quantile,
    nonconformity_binary,
    predict_set_binary,
    random_cal_eval_indices,
    set_metrics,
    split_conformal_metrics,
)


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
    plt.savefig(path, dpi=160)
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
        save_bar_plot(plot_df, "score", "coverage", "horizon", "Marginal coverage by score", outdir / "four_score_coverage.png")
        width_df = raw.groupby(["horizon", "score"], as_index=False)["avg_size"].mean()
        save_bar_plot(width_df, "score", "avg_size", "horizon", "Average set size by score", outdir / "four_score_set_size.png")
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
    save_bar_plot(plot_df, "country", "coverage", "method", "Per-country coverage", outdir / "mondrian_coverage.png")
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
        plt.savefig(outdir / "shift_break_temporal.png", dpi=160)
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
    plt.savefig(outdir / "online_adapt_coverage.png", dpi=160)
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
    save_bar_plot(plot_df, "label", "coverage", "method", "Label-robust coverage", outdir / "label_robust_coverage.png")
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
        plt.savefig(outdir / "selection_fdr.png", dpi=160)
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
        save_bar_plot(plot_df, "method", "coverage", "horizon", "TabPFN native and conformalized sets", outdir / "tabpfn_native_coverage.png")
    print(f"Wrote TabPFN outputs to {outdir}")
    return raw


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
