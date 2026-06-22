"""Bondora credit-risk analysis.

This file contains the main loan-level pipeline, final figure generation,
and the small data-preparation utilities used by the conformal experiments.
Use the subcommands at the bottom to run each stage.
"""


import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore", category=FutureWarning)


def save_transparent_figure(fig, path, dpi=150, **kwargs):
    """Save a matplotlib figure with a transparent canvas and axes."""
    fig.patch.set_alpha(0)
    for ax in fig.axes:
        ax.set_facecolor("none")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=True, **kwargs)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# Config

DEFAULT_CSV_PATH = "./data/LoanData.csv"  # Put Data here
COUNTRIES = ["EE", "FI", "ES"]

# Fields excluded because they are unavailable at origination or directly encode
# repayment/default outcomes.
OUTCOME_OR_POST_ORIGINATION_EXCLUDE = [
    "PrincipalRecovery", "InterestRecovery",
    "PrincipalWriteOffs", "InterestAndPenaltyWriteOffs",
    "PrincipalDebtServicingCost", "InterestAndPenaltyDebtServicingCost",
    "PlannedPrincipalPostDefault", "PlannedInterestPostDefault", "EAD1", "EAD2",
    "ActiveLateCategory", "ActiveLateLastPaymentCategory",
    "ReScheduledOn", "StageActiveSince", "PreviousRepaymentsBeforeLoan",
    "PlannedPrincipalTillDate", "PlannedInterestTillDate",
    "RecoveryStage", "NextPaymentDate", "NextPaymentSum",
    "NrOfScheduledPayments", "ReportAsOfEOD",
    "AmountOfPreviousLoansBeforeLoan",
    "DateOfBirth",

    # post-originationdates
    "DebtOccuredOn",                # 进入催收日期
    "DebtOccuredOnForSecondary",    # 二次催收日期
    "ContractEndDate",              # 合同结束
    "LastPaymentOn",                # 最近一次还款
    "LoanStatusActiveFrom",         # 当前状态生效
    # post-origination 状态字段
    "Restructured",                 # 是否被 restructure
    "WorkoutProcessingType",        # 催收类型
    # 还款进度Type
    "InterestAndPenaltyBalance",
    "PrincipalBalance",
    "InterestAndPenaltyPaymentsMade",
    "PrincipalPaymentsMade",
]

# Borderline fields removed after the leakage audit.
AUDIT_EXCLUDE = [
    "PrincipalOverdueBySchedule",   # 直接衡量逾期本金
    "MaturityDate_Last",            # differs from original maturity after restructuring
    "FirstPaymentDate",             
    "NextPaymentNr",                # post-origination payment counter
    "ActiveScheduleFirstPaymentReached",    # post-origination boolean
]

DROP_LEAKAGE = OUTCOME_OR_POST_ORIGINATION_EXCLUDE + AUDIT_EXCLUDE

DROP_HIGH_MISSING = [
    "LoanCancelled",
    "CreditScoreEsEquifaxRisk",
    "PreviousEarlyRepaymentsBeforeLoan",
    "GracePeriodStart", "GracePeriodEnd",
    "ContractEndDate",
]

# Bondora 自己的 scoring outputs 
BONDORA_SIGNALS = [
    "ProbabilityOfDefault", "ExpectedLoss", "LossGivenDefault", "ExpectedReturn",
]

# 不能进training
LABEL_ONLY = ["Status", "DefaultDate", "WorseLateCategory"]

LABELS = [
    "L1_strict_late",
    "L2_status_or_default",
    "L3_default_date",
    "L4_ever_60d_late",
    "L5_default_excl_cured",
]

# A feature is flagged in either direction: AUC >= t or AUC <= 1 - t.
AUC_LEAKAGE_THRESHOLD = 0.85
AUC_SUSPICIOUS_THRESHOLD = 0.70

CLOSED_STATUSES = ["Late", "Repaid"]

# Helper functions

def parse_date_columns(df):
    """Standardize date columns to datetime objects."""
    date_cols = ["LoanDate", "DefaultDate", "MaturityDate_Original",
                 "MaturityDate_Last", "ContractEndDate", "ListedOnUTC",
                 "DebtOccuredOn", "DebtOccuredOnForSecondary", "LastPaymentOn",
                 "LoanStatusActiveFrom"]
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def add_one_year_label(df):
    """Construct the 1-year default horizon and observation flags."""
    data_cutoff = df["LoanDate"].max()
    df["days_to_default"] = (df["DefaultDate"] - df["LoanDate"]).dt.days
    df["default_1y"] = (
        (df["days_to_default"] >= 0) &
        (df["days_to_default"] <= 365)
    ).astype(int)
    df["fully_observed_1y"] = df["LoanDate"] <= (data_cutoff - pd.Timedelta(days=365))
    return df

def split_column_groups(df_clean):
    """Classify columns into meta, label, bondora signals, and feature candidates."""
    all_cols = set(df_clean.columns)
    bondora_cols = [c for c in BONDORA_SIGNALS if c in all_cols]
    label_cols = [c for c in LABEL_ONLY if c in all_cols]
    meta_cols = [c for c in ["LoanId", "LoanDate", "Country"] if c in all_cols]
    feature_cols = [c for c in df_clean.columns
                    if c not in set(bondora_cols + label_cols + meta_cols)]
    return meta_cols, label_cols, bondora_cols, feature_cols

def add_default_labels(closed):
    """Construct the L1 through L5 default labels on closed loans."""
    closed["L1_strict_late"] = (closed["Status"] == "Late").astype(int)
    closed["L2_status_or_default"] = (
        (closed["Status"] == "Late") | closed["DefaultDate"].notna()
    ).astype(int)
    closed["L3_default_date"] = closed["DefaultDate"].notna().astype(int)
    if "WorseLateCategory" in closed.columns:
        closed["L4_ever_60d_late"] = closed["WorseLateCategory"].apply(is_60plus)
    else:
        closed["L4_ever_60d_late"] = np.nan
    closed["L5_default_excl_cured"] = (
        closed["DefaultDate"].notna() & (closed["Status"] != "Repaid")
    ).astype(int)
    return closed

def write_feature_catalog(path, meta_cols, label_cols, bondora_cols, feature_cols):
    """Export a text file detailing the initial column classification."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Bondora field classification\n")
        f.write("# meta\n")
        for c in meta_cols: f.write(f"meta\t{c}\n")
        f.write("\n# label\n")
        for c in label_cols: f.write(f"label\t{c}\n")
        f.write("\n# bondora\n")
        for c in bondora_cols: f.write(f"bondora\t{c}\n")
        f.write("\n# feature\n")
        for c in sorted(feature_cols): f.write(f"feature\t{c}\n")

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def print_step(step_no: int, title: str):
    print(f"\nStep {step_no}: {title}")

def safe_auc(y_true, y_score):
    y = pd.Series(y_true).dropna()
    s = pd.Series(y_score).loc[y.index]
    mask = y.notna() & s.notna()
    y = y[mask].astype(int)
    s = s[mask].astype(float)
    if y.nunique() < 2 or len(y) < 30:
        return np.nan
    return roc_auc_score(y, s)


def normalize_probability(series: pd.Series) -> pd.Series: #兼容POD格式
    s = pd.to_numeric(series, errors="coerce")
    q99 = s.quantile(0.99)
    if pd.notna(q99) and q99 > 1.5:
        return s / 100.0
    return s


def is_60plus(x): #WorseLateCategory逾期判定
    if pd.isna(x):
        return 0
    return int(any(b in str(x) for b in ["61-90", "91-120", "121-150", "151-180", "180+"]))


def onehot_encoder(): #兼容scikit-learn版本
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


# Step 1: Load + clean + 5 labels + 1y label + column classification

def step1_load_and_prepare(csv_path: str, data_dir: str, fig_dir: str):
    #load Bondora data and do preparing
    print_step(1, "Load + clean + 5 labels + 1y label + column classification")

    if not os.path.exists(csv_path):
        abs_path = os.path.abspath(csv_path)
        raise FileNotFoundError(
            f"\n[ERROR] 无法打开数据文件: {abs_path}\n"
            f"请确保文件路径正确。如果文件名包含隐藏的扩展名（如 LoanData.csv.csv），请更正或指定完整名称。"
        )

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Raw data: {df.shape[0]:,} rows, {df.shape[1]:,} columns")

    df = df[df["Country"].isin(COUNTRIES)].copy()
    df = parse_date_columns(df)

    if "ProbabilityOfDefault" in df.columns:
        df["ProbabilityOfDefault"] = normalize_probability(df["ProbabilityOfDefault"])

    df = df.sort_values("LoanDate").reset_index(drop=True)
    df = add_one_year_label(df)

    drop_cols = [c for c in DROP_LEAKAGE + DROP_HIGH_MISSING if c in df.columns]
    df_clean = df.drop(columns=drop_cols)

    meta_cols, label_cols, bondora_cols, feature_cols = split_column_groups(df_clean)

    print(f"After country filter: {df_clean.shape[0]:,} rows")
    print(f"Excluded columns: {len(drop_cols)}")
    print(
        "Column groups: "
        f"{len(meta_cols)} meta, "
        f"{len(label_cols)} label-only, "
        f"{len(bondora_cols)} Bondora scores, "
        f"{len(feature_cols)} candidate features"
    )

    closed = df_clean[df_clean["Status"].isin(CLOSED_STATUSES)].copy()
    closed = add_default_labels(closed)

    if "LoanId" in closed.columns:
        one_year_cols = ["LoanId", "default_1y", "fully_observed_1y", "days_to_default"]
        closed = closed.merge(df_clean[one_year_cols], on="LoanId", how="left")

    base_rates = {label: closed[label].mean() for label in LABELS if label in closed.columns}
    spread = (
        max(base_rates.values()) - min(base_rates.values())
        if base_rates else np.nan
    )

    print(f"Closed loans: {len(closed):,}")
    print(f"Label spread: {spread * 100:.2f} pp")

    closed_path = f"{data_dir}/closed_loans.csv"
    clean_path = f"{data_dir}/loans_clean.csv"
    catalog_path = f"{data_dir}/feature_columns.txt"

    closed.to_csv(closed_path, index=False)
    df_clean.to_csv(clean_path, index=False)
    write_feature_catalog(
        catalog_path,
        meta_cols=meta_cols,
        label_cols=label_cols,
        bondora_cols=bondora_cols,
        feature_cols=feature_cols,
    )

    print(f"Saved {closed_path}")
    print(f"Saved {clean_path}")
    print(f"Saved {catalog_path}")

    return (
        df_clean, closed, feature_cols, meta_cols, bondora_cols, label_cols, base_rates, spread,
    )

# Step 2: Label diagnostics

def compute_country_label_rates(closed_df: pd.DataFrame):
    """Compute country-level default rates under each label definition."""
    labels = [col for col in LABELS if col in closed_df.columns]

    rates = closed_df.groupby("Country")[labels].mean().reindex(COUNTRIES)
    counts = closed_df.groupby("Country").size().reindex(COUNTRIES).fillna(0).astype(int)

    return rates, counts


def compute_l3_l4_overlap(closed_df: pd.DataFrame):
    """Compare DefaultDate-based defaults with the 60+ days late rule."""
    if not {"L3_default_date", "L4_ever_60d_late"}.issubset(closed_df.columns):
        return None, {}

    l3 = closed_df["L3_default_date"].astype(int)
    l4 = closed_df["L4_ever_60d_late"].astype(int)

    both = ((l3 == 1) & (l4 == 1)).sum()
    either = ((l3 == 1) | (l4 == 1)).sum()

    confusion = pd.crosstab(
        l3,
        l4,
        rownames=["L3_default_date"],
        colnames=["L4_ever_60d_late"],
    )

    metrics = {
        "L3_L4_jaccard": both / max(either, 1),
        "P_L4_given_L3": both / max((l3 == 1).sum(), 1),
        "P_L3_given_L4": both / max((l4 == 1).sum(), 1),
    }

    return confusion, metrics


def compute_cured_defaults(closed_df: pd.DataFrame):
    """Count loans that defaulted at some point but eventually ended as repaid."""
    cured = closed_df[
        (closed_df["Status"] == "Repaid") &
        closed_df["DefaultDate"].notna()
    ]

    n_cured = len(cured)
    n_default_date = closed_df["DefaultDate"].notna().sum()

    return {
        "n_cured_defaults": n_cured,
        "cured_share_all_closed": n_cured / max(len(closed_df), 1),
        "cured_share_default_date": n_cured / max(n_default_date, 1),
    }

def step2_label_diagnostics(_df_clean, closed_df, base_rates, spread, fig_dir, data_dir):
    """Compute label diagnostics and save the Step 2 outputs."""
    print_step(2, "Label diagnostics")

    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    metrics_rows = [{"metric": "spread_pp", "value": spread * 100}]
    metrics_rows.extend(
        {"metric": f"base_rate_{label}", "value": value}
        for label, value in base_rates.items()
    )

    country_rates, country_counts = compute_country_label_rates(closed_df)
    country_rates_path = os.path.join(data_dir, "country_label_rates.csv")
    country_rates.to_csv(country_rates_path)

    confusion, overlap_metrics = compute_l3_l4_overlap(closed_df)
    if confusion is not None:
        confusion.to_csv(os.path.join(data_dir, "l3_l4_confusion_matrix.csv"))
        metrics_rows.extend(
            {"metric": name, "value": value}
            for name, value in overlap_metrics.items()
        )

    cured_metrics = compute_cured_defaults(closed_df)
    metrics_rows.extend(
        {"metric": name, "value": value}
        for name, value in cured_metrics.items()
    )
    out_df = pd.DataFrame(metrics_rows)
    out_path = os.path.join(data_dir, "step2_label_diagnostics.csv")
    out_df.to_csv(out_path, index=False)

    print(f"Country-label rates saved to: {country_rates_path}")
    print(f"Label diagnostics saved to:   {out_path}")

    return out_df

# STEP 3 — Bondora PoD diagnostics

def expected_calibration_error(y_true, y_prob, n_bins=10):
    df_cal = pd.DataFrame({"y": y_true, "p": y_prob}).dropna()
    df_cal = df_cal[(df_cal["p"] >= 0) & (df_cal["p"] <= 1)]
    if df_cal.empty or df_cal["y"].nunique() < 2:
        return np.nan, pd.DataFrame()
    try:
        df_cal["bin"] = pd.qcut(df_cal["p"], q=n_bins, duplicates="drop")
    except ValueError:
        df_cal["bin"] = pd.cut(df_cal["p"], bins=n_bins, include_lowest=True)
    g = df_cal.groupby("bin", observed=False).agg(
        n=("y", "size"),
        pred=("p", "mean"),
        actual=("y", "mean"),
    ).reset_index()
    g["abs_gap"] = (g["actual"] - g["pred"]).abs()
    ece = (g["n"] / g["n"].sum() * g["abs_gap"]).sum()
    return ece, g


def calibration_slope_intercept(y_true, y_prob):
    df_cal = pd.DataFrame({"y": y_true, "p": y_prob}).dropna()
    df_cal = df_cal[(df_cal["p"] > 0) & (df_cal["p"] < 1)]
    if df_cal.empty or df_cal["y"].nunique() < 2:
        return np.nan, np.nan
    eps = 1e-6
    p = np.clip(df_cal["p"].astype(float).values, eps, 1 - eps)
    y = df_cal["y"].astype(int).values
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(fit_intercept=True, solver="lbfgs")
    lr.fit(logit_p, y)
    return lr.intercept_[0], lr.coef_[0][0]


def eval_probability_score(y_true, y_prob, label_name, n_bins=10):
    df_eval = pd.DataFrame({"y": y_true, "p": y_prob}).dropna()
    df_eval = df_eval[(df_eval["p"] >= 0) & (df_eval["p"] <= 1)]
    y = df_eval["y"].astype(int)
    p = df_eval["p"].astype(float)
    if len(df_eval) == 0 or y.nunique() < 2:
        return {"label": label_name, "n": len(df_eval), "base_rate": np.nan,
                "pred_mean": np.nan, "auc": np.nan, "brier": np.nan,
                "ece": np.nan, "slope": np.nan, "intercept": np.nan}, pd.DataFrame()
    ece, bins = expected_calibration_error(y, p, n_bins=n_bins)
    intercept, slope = calibration_slope_intercept(y, p)
    metrics = {
        "label": label_name,
        "n": len(df_eval),
        "base_rate": y.mean(),
        "pred_mean": p.mean(),
        "auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
        "ece": ece,
        "slope": slope,
        "intercept": intercept,
    }
    bins.insert(0, "label", label_name)
    return metrics, bins


def step3_pod_diagnostics(df_clean, closed, data_dir, fig_dir):
    """Evaluate Bondora ProbabilityOfDefault across 1-year and lifetime labels."""
    print_step(3, "Bondora PoD diagnostics")

    if "ProbabilityOfDefault" not in df_clean.columns:
        print("Warning: No ProbabilityOfDefault column — skip Step 3.")
        return pd.DataFrame(), pd.DataFrame()

    # 3.1 Sanity check: PoD 是 origination-time 还是 post-hoc 
    df = df_clean[
        df_clean["ProbabilityOfDefault"].notna() &
        (df_clean["ProbabilityOfDefault"] > 0)
    ].copy()

    def group_label(row):
        s, dd = row["Status"], pd.notna(row["DefaultDate"])
        if s == "Current": return "1_Current"
        if s == "Late" and not dd: return "2_Late_noDD"
        if s == "Late" and dd: return "3_Late_hasDD"
        if s == "Repaid" and not dd: return "4_Repaid_clean"
        if s == "Repaid" and dd: return "5_Repaid_cured"
        return "other"

    df["group"] = df.apply(group_label, axis=1)
    g1 = df.groupby("group")["ProbabilityOfDefault"].agg(["count", "mean", "median", "std"]).round(4)
    print("\n[3a] PoD mean by status × DefaultDate group:")
    print(g1)

    mean_current = df[df["group"] == "1_Current"]["ProbabilityOfDefault"].mean()
    mean_late_dd = df[df["group"] == "3_Late_hasDD"]["ProbabilityOfDefault"].mean()
    ratio = mean_late_dd / mean_current if mean_current and mean_current > 0 else np.inf
    print(f"  ratio Late_hasDD / Current = {ratio:.2f}  (post-hoc 嫌疑 if >> 1)")

    # 3.2 PoD-as-classifier AUC for each label
    pod_closed = closed[
        closed["ProbabilityOfDefault"].notna() &
        (closed["ProbabilityOfDefault"] > 0)
    ].copy()
    auc_rows = []
    for lab in LABELS:
        if lab in pod_closed.columns and pod_closed[lab].nunique() == 2:
            auc_rows.append({
                "label": lab,
                "auc": safe_auc(pod_closed[lab], pod_closed["ProbabilityOfDefault"]),
            })
    auc_df = pd.DataFrame(auc_rows)
    print("\n[3b] PoD-as-classifier AUC by label:")
    print(auc_df.round(4))

    # 3.3 Main: PoD horizon calibration — 1y vs lifetime
    obs_1y = df_clean[
        df_clean["fully_observed_1y"] &
        df_clean["ProbabilityOfDefault"].notna() &
        (df_clean["ProbabilityOfDefault"] > 0)
    ].copy()
    print(f"\n[3c] PoD horizon calibration — 1y subset n={len(obs_1y):,}, lifetime subset n={len(pod_closed):,}")

    m_1y, bins_1y = eval_probability_score(
        obs_1y["default_1y"], obs_1y["ProbabilityOfDefault"], "Bondora_PoD_vs_default_1y"
    )
    m_lt, bins_lt = eval_probability_score(
        pod_closed["L3_default_date"], pod_closed["ProbabilityOfDefault"], "Bondora_PoD_vs_L3_lifetime"
    )

    metrics = pd.DataFrame([m_1y, m_lt])
    bins = pd.concat([bins_1y, bins_lt], ignore_index=True)
    print(metrics.round(4)[["label", "n", "base_rate", "pred_mean", "auc", "brier", "ece"]])

    metrics.to_csv(f"{data_dir}/step3_pod_horizon_metrics.csv", index=False)
    bins.to_csv(f"{data_dir}/step3_pod_calibration_bins.csv", index=False)

    # 3.4 PoD vs Actual by vintage 
    pod_closed["vintage"] = pod_closed["LoanDate"].dt.year
    vintage = pod_closed.groupby("vintage").agg(
        n=("L3_default_date", "size"),
        pod_mean=("ProbabilityOfDefault", "mean"),
        actual_dr=("L3_default_date", "mean"),
    ).round(4)
    vintage["gap"] = (vintage["pod_mean"] - vintage["actual_dr"]).round(4)
    vintage["abs_gap"] = vintage["gap"].abs()
    old_vintage_gap = vintage[vintage.index <= 2019]["abs_gap"].mean()

    # 3 summary
    sanity_rows = [
        {"metric": "late_hasDD_over_current_ratio", "value": ratio},
        {"metric": "old_vintage_abs_gap_mean_le_2019", "value": old_vintage_gap},
        {"metric": "ece_1y", "value": m_1y["ece"]},
        {"metric": "ece_lifetime", "value": m_lt["ece"]},
        {"metric": "auc_1y", "value": m_1y["auc"]},
        {"metric": "auc_lifetime", "value": m_lt["auc"]},
    ]
    for _, row in auc_df.iterrows():
        sanity_rows.append({"metric": f"auc_PoD_vs_{row['label']}", "value": row["auc"]})

    sanity = pd.DataFrame(sanity_rows)
    sanity.to_csv(f"{data_dir}/step3_pod_sanity.csv", index=False)
    vintage.to_csv(f"{data_dir}/step3_pod_vintage.csv")

    flags = (
        int(ratio > 3) +
        int(bool((auc_df["auc"] > 0.85).any())) +
        int(pd.notna(old_vintage_gap) and old_vintage_gap < 0.02)
    )
    print(f"\nPoD sanity flags: {flags}")

    return metrics, sanity


# Step 4: Feature leakage audit

def feature_auc_against_target(feature: pd.Series, target: pd.Series):
    """Screen one feature using univariate AUC against the audit target."""
    df = pd.DataFrame({"x": feature, "y": target}).dropna()

    if len(df) < 100 or df["y"].nunique() < 2:
        return np.nan

    x = df["x"]
    y = df["y"].astype(int)

    if pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x):
        score = pd.to_numeric(x, errors="coerce")
    else:
        # For categorical fields, use a simple target-rate encoding.
        # This is only for leakage screening, not for model training.
        x_str = x.astype(str)
        rates = df.groupby(x_str)["y"].mean()
        score = x_str.map(rates)

    return safe_auc(y, score)


def audit_flag_from_auc(auc: float) -> str:
    """Assign an audit flag using the stronger direction of the AUC."""
    if pd.isna(auc):
        return "uncomputable"

    auc_strength = max(auc, 1 - auc)

    if auc_strength >= AUC_LEAKAGE_THRESHOLD:
        return "leakage_suspect"
    if auc_strength >= AUC_SUSPICIOUS_THRESHOLD:
        return "review"

    return "safe"


def run_feature_audit(closed: pd.DataFrame, feature_cols, target_col: str):
    """Run the univariate feature screen and return the audit table."""
    rows = []

    for col in feature_cols:
        if col not in closed.columns:
            continue

        feature = closed[col]
        n_nonnull = int(feature.notna().sum())

        if n_nonnull < 100:
            rows.append({
                "feature": col,
                "auc": np.nan,
                "auc_strength": np.nan,
                "n_nonnull": n_nonnull,
                "dtype": str(feature.dtype),
                "audit_flag": "insufficient_data",
            })
            continue

        auc = feature_auc_against_target(feature, closed[target_col])
        auc_strength = max(auc, 1 - auc) if pd.notna(auc) else np.nan

        rows.append({
            "feature": col,
            "auc": auc,
            "auc_strength": auc_strength,
            "n_nonnull": n_nonnull,
            "dtype": str(feature.dtype),
            "audit_flag": audit_flag_from_auc(auc),
        })

    audit = pd.DataFrame(rows)
    if audit.empty:
        return audit

    return audit.sort_values(
        ["auc_strength", "n_nonnull"],
        ascending=[False, False],
        na_position="last",
    )


def plot_feature_audit(audit: pd.DataFrame, output_path: str):
    """Plot the distribution of univariate AUC strengths."""
    auc_strength = audit["auc_strength"].dropna()

    if auc_strength.empty:
        print("No valid AUC values to plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(auc_strength, bins=40, color="#3b82f6", edgecolor="white")
    ax.axvline(
        AUC_SUSPICIOUS_THRESHOLD,
        ls="--",
        color="orange",
        label=f"Review threshold: {AUC_SUSPICIOUS_THRESHOLD}",
    )
    ax.axvline(
        AUC_LEAKAGE_THRESHOLD,
        ls="--",
        color="red",
        label=f"High-risk threshold: {AUC_LEAKAGE_THRESHOLD}",
    )

    ax.set_xlabel("Univariate AUC strength: max(AUC, 1 - AUC)")
    ax.set_ylabel("Number of features")
    ax.set_title("Feature audit based on univariate AUC", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_transparent_figure(fig, output_path, dpi=150)
    plt.close(fig)


def step4_feature_audit(df_clean, closed, feature_cols, data_dir, fig_dir):
    """Audit candidate features using univariate AUC against the lifetime label."""
    print_step(4, "Feature leakage audit")

    target_col = "L3_default_date"
    if target_col not in closed.columns:
        print(f"{target_col} not found; skipping feature audit.")
        return [], pd.DataFrame()

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    candidate_features = [col for col in feature_cols if col in closed.columns]
    print(f"Auditing {len(candidate_features)} candidate features.")

    audit = run_feature_audit(
        closed=closed,
        feature_cols=candidate_features,
        target_col=target_col,
    )

    audit_path = os.path.join(data_dir, "step4_feature_audit.csv")
    audit.to_csv(audit_path, index=False)

    if not audit.empty:
        print("Audit flag counts:")
        print(audit["audit_flag"].value_counts())

        high_risk = audit[audit["audit_flag"] == "leakage_suspect"]
        review = audit[audit["audit_flag"] == "review"]

        if not high_risk.empty:
            print("\nHigh-risk features:")
            print(high_risk[["feature", "auc", "auc_strength", "n_nonnull"]].head(15).to_string(index=False))

        if not review.empty:
            print("\nFeatures to review:")
            print(review[["feature", "auc", "auc_strength", "n_nonnull"]].head(15).to_string(index=False))

    safe_features = audit.loc[audit["audit_flag"] == "safe", "feature"].tolist()

    plot_feature_audit(
        audit,
        os.path.join(fig_dir, "feature_leakage_audit_auc.png"),
    )

    print(f"Feature audit saved to: {audit_path}")
    print(f"Safe features for modeling: {len(safe_features)}")

    return safe_features, audit


# Step 5: Model comparison

def add_datetime_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    for col in list(X.columns):
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            dt = pd.to_datetime(X[col], errors="coerce")
            X[f"{col}_year"] = dt.dt.year
            X[f"{col}_month"] = dt.dt.month
            X[f"{col}_days"] = (dt - pd.Timestamp("1970-01-01")).dt.days
            X = X.drop(columns=[col])

    return X


def cap_categorical_levels(X_train: pd.DataFrame, X_all: pd.DataFrame, cat_cols, top_k=50):
    """
    Keep the most frequent levels from the training split and group the rest
    as __OTHER__.
    """
    X_all = X_all.copy()
    level_maps = {}

    for col in cat_cols:
        if col not in X_train.columns:
            continue
        levels = X_train[col].astype(str).value_counts(dropna=False).head(top_k).index
        level_maps[col] = set(levels)

    for col, keep in level_maps.items():
        s = X_all[col].astype(str).fillna("__MISSING__")
        X_all[col] = np.where(s.isin(keep), s, "__OTHER__")

    return X_all


def make_country_stratified_time_split(df: pd.DataFrame, train_q=0.60, cal_q=0.80):
    """Split each country by LoanDate so train/cal/test are time ordered."""
    out = df.sort_values(["Country", "LoanDate"]).copy()
    out["split"] = "test"

    for country, g in out.groupby("Country", sort=False):
        idx = g.index.to_numpy()
        n = len(idx)

        train_end = int(n * train_q)
        cal_end = int(n * cal_q)

        out.loc[idx[:train_end], "split"] = "train"
        out.loc[idx[train_end:cal_end], "split"] = "cal"
        out.loc[idx[cal_end:], "split"] = "test"

    return out


def build_model_frame(df_clean: pd.DataFrame, closed: pd.DataFrame, max_rows_model: int):
    """Merge lifetime labels, create splits, and optionally downsample training rows."""
    lifetime = closed[["LoanId", "L3_default_date"]].drop_duplicates("LoanId")

    model_df = (
        df_clean
        .merge(lifetime, on="LoanId", how="left")
        .loc[lambda d: d["LoanDate"].notna()]
        .copy()
    )
    model_df = make_country_stratified_time_split(model_df)

    if len(model_df) <= max_rows_model:
        return model_df

    train = model_df[model_df["split"] == "train"]
    rest = model_df[model_df["split"] != "train"]

    keep_train_n = max(max_rows_model - len(rest), int(max_rows_model * 0.4))
    train = train.sample(n=min(len(train), keep_train_n), random_state=42)

    return (
        pd.concat([train, rest], axis=0)
        .sort_values(["Country", "LoanDate"])
        .reset_index(drop=True)
    )


def select_model_features(model_df: pd.DataFrame, safe_features):
    """Keep audited features and remove labels or baseline scores."""
    excluded = set(
        BONDORA_SIGNALS
        + LABEL_ONLY
        + LABELS
        + ["default_1y", "fully_observed_1y", "days_to_default"]
    )

    return [
        col for col in safe_features
        if col in model_df.columns and col not in excluded
    ]


def prepare_feature_matrix(model_df: pd.DataFrame, feature_cols, top_k_cat=50):
    """Build the modeling matrix and identify numeric/categorical columns."""
    X = add_datetime_features(model_df[feature_cols])

    cat_cols = [
        col for col in X.columns
        if not pd.api.types.is_numeric_dtype(X[col])
        or pd.api.types.is_bool_dtype(X[col])
    ]

    train_mask = model_df["split"] == "train"
    X = cap_categorical_levels(
        X_train=X.loc[train_mask],
        X_all=X,
        cat_cols=cat_cols,
        top_k=top_k_cat,
    )

    cat_cols = [
        col for col in X.columns
        if not pd.api.types.is_numeric_dtype(X[col])
        or pd.api.types.is_bool_dtype(X[col])
    ]
    num_cols = [col for col in X.columns if col not in cat_cols]

    return X, num_cols, cat_cols


def build_lr_pipeline(num_cols, cat_cols):
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", onehot_encoder()),
            ]), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    model = LogisticRegression(max_iter=1000, solver="saga", n_jobs=-1)
    return Pipeline([("pre", pre), ("model", model)])


def build_gbdt_pipeline(num_cols, cat_cols, backend="auto"):
    """Build the requested tree model and its preprocessing pipeline."""
    valid_backends = {"auto", "lightgbm", "xgboost", "sklearn"}
    if backend not in valid_backends:
        raise ValueError(
            f"Unknown GBDT backend '{backend}'. Choose from {sorted(valid_backends)}."
        )

    model_name = None
    model = None
    import_errors = []

    if backend in {"auto", "lightgbm"}:
        try:
            from lightgbm import LGBMClassifier

            model_name = "lightgbm"
            model = LGBMClassifier(
                n_estimators=400,
                learning_rate=0.03,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
        except ImportError as err:
            import_errors.append(f"lightgbm: {err}")
            if backend == "lightgbm":
                raise RuntimeError(
                    "LightGBM was requested but is not installed. "
                    "Install it with `pip install lightgbm`."
                ) from err

    if model is None and backend in {"auto", "xgboost"}:
        try:
            from xgboost import XGBClassifier

            model_name = "xgboost"
            model = XGBClassifier(
                n_estimators=400,
                learning_rate=0.03,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
        except ImportError as err:
            import_errors.append(f"xgboost: {err}")
            if backend == "xgboost":
                raise RuntimeError(
                    "XGBoost was requested but is not installed. "
                    "Install it with `pip install xgboost`."
                ) from err

    if model is None:
        if backend == "auto" and import_errors:
            print(
                "Optional boosting libraries unavailable; "
                "using sklearn HistGradientBoostingClassifier."
            )
        model_name = "sklearn_hgbdt"
        model = HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=42,
        )

    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )),
            ]), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    return model_name, Pipeline([("pre", pre), ("model", model)])


def initial_prediction_frame(model_df: pd.DataFrame) -> pd.DataFrame:
    """Create the prediction output frame shared by all models."""
    pred = model_df[["LoanId", "LoanDate", "Country", "split"]].copy()

    if "ProbabilityOfDefault" in model_df.columns:
        pred["pod_bondora"] = model_df["ProbabilityOfDefault"].values

    pred["y_1y"] = model_df["default_1y"].values
    pred["y_lifetime"] = model_df["L3_default_date"].values
    pred["fully_observed_1y"] = model_df["fully_observed_1y"].values

    return pred


def horizon_specs(model_df: pd.DataFrame):
    return [
        ("1y", "default_1y", model_df["fully_observed_1y"].fillna(False)),
        ("lifetime", "L3_default_date", model_df["L3_default_date"].notna()),
    ]


def fit_and_score_model(
    pipe,
    model_name: str,
    horizon_name: str,
    target_col: str,
    model_df: pd.DataFrame,
    X_all: pd.DataFrame,
    idx_train,
    idx_cal,
    idx_test,
    pred_out: pd.DataFrame,
):
    """Fit one model for one horizon and append predictions/metrics."""
    y_train = model_df.loc[idx_train, target_col].astype(int)
    y_test = model_df.loc[idx_test, target_col].astype(int)

    pipe.fit(X_all.loc[idx_train], y_train)

    pred_col = f"pred_{model_name}_{horizon_name}"
    pred_out[pred_col] = np.nan

    if idx_cal.sum() > 0:
        pred_out.loc[idx_cal, pred_col] = pipe.predict_proba(X_all.loc[idx_cal])[:, 1]

    p_test = pipe.predict_proba(X_all.loc[idx_test])[:, 1]
    pred_out.loc[idx_test, pred_col] = p_test

    metrics, _ = eval_probability_score(
        y_test,
        p_test,
        f"{model_name}_{horizon_name}",
    )
    metrics["model"] = model_name
    metrics["horizon"] = horizon_name
    metrics["n_cal_with_pred"] = int(idx_cal.sum())

    return metrics


def plot_model_auc(metrics: pd.DataFrame, output_path: str):
    """Plot test AUC for each model and horizon."""
    if metrics.empty:
        return

    model_names = list(metrics["model"].unique())
    horizons = ["1y", "lifetime"]
    colors = ["#3b82f6", "#ef4444"]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, horizon in enumerate(horizons):
        vals = []
        for model_name in model_names:
            sub = metrics[
                (metrics["model"] == model_name)
                & (metrics["horizon"] == horizon)
            ]
            vals.append(sub["auc"].iloc[0] if not sub.empty else np.nan)

        bars = ax.bar(
            x + (i - 0.5) * width,
            vals,
            width,
            label=horizon,
            color=colors[i],
            edgecolor="white",
        )

        for bar, value in zip(bars, vals):
            if pd.notna(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.005,
                    f"{value:.3f}",
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Test AUC")
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.95, ls="--", color="gray", alpha=0.5, label="review threshold")
    ax.set_title("Model AUC by horizon", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_transparent_figure(fig, output_path, dpi=150)
    plt.close(fig)


def step5_clean_model_bakeoff(
    df_clean,
    closed,
    safe_features,
    audit_df,
    data_dir,
    fig_dir,
    max_rows_model=200000,
    top_k_cat=50,
    gbdt_backend="auto",
):
    """Train LR and GBDT baselines after the feature audit."""
    print_step(5, "Model comparison")

    if "LoanId" not in df_clean.columns:
        print("LoanId not found; skipping model comparison.")
        return pd.DataFrame(), pd.DataFrame()

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    model_df = build_model_frame(df_clean, closed, max_rows_model=max_rows_model)

    print("Split sizes by country:")
    print(pd.crosstab(model_df["Country"], model_df["split"]))

    feature_cols = select_model_features(model_df, safe_features)
    X_all, num_cols, cat_cols = prepare_feature_matrix(
        model_df,
        feature_cols,
        top_k_cat=top_k_cat,
    )

    print(f"Using {len(feature_cols)} model features.")
    print(f"Feature matrix: {len(num_cols)} numeric, {len(cat_cols)} categorical.")

    pred_out = initial_prediction_frame(model_df)
    metric_rows = []

    gbdt_name, _ = build_gbdt_pipeline(
        num_cols,
        cat_cols,
        backend=gbdt_backend,
    )
    print(f"Tree-model backend: {gbdt_name}")

    for horizon_name, target_col, eligible in horizon_specs(model_df):
        idx_train = (model_df["split"] == "train") & eligible & model_df[target_col].notna()
        idx_cal = (model_df["split"] == "cal") & eligible & model_df[target_col].notna()
        idx_test = (model_df["split"] == "test") & eligible & model_df[target_col].notna()

        print(
            f"{horizon_name}: "
            f"train={idx_train.sum():,}, "
            f"cal={idx_cal.sum():,}, "
            f"test={idx_test.sum():,}"
        )

        if idx_train.sum() < 1000 or idx_test.sum() < 300:
            print(f"{horizon_name}: skipped because there is not enough data.")
            continue

        if model_df.loc[idx_train, target_col].nunique() < 2:
            print(f"{horizon_name}: skipped because the training target has one class.")
            continue

        models = [
            ("lr", build_lr_pipeline(num_cols, cat_cols)),
            build_gbdt_pipeline(num_cols, cat_cols, backend=gbdt_backend),
        ]

        for model_name, pipe in models:
            try:
                metrics = fit_and_score_model(
                    pipe=pipe,
                    model_name=model_name,
                    horizon_name=horizon_name,
                    target_col=target_col,
                    model_df=model_df,
                    X_all=X_all,
                    idx_train=idx_train,
                    idx_cal=idx_cal,
                    idx_test=idx_test,
                    pred_out=pred_out,
                )
                metric_rows.append(metrics)

                print(
                    f"{model_name}/{horizon_name}: "
                    f"AUC={metrics['auc']:.4f}, "
                    f"ECE={metrics['ece']:.3f}, "
                    f"Brier={metrics['brier']:.3f}"
                )

                if metrics["auc"] > 0.95:
                    print(f"{model_name}/{horizon_name}: AUC above review threshold.")

            except Exception as err:
                print(f"{model_name}/{horizon_name}: failed ({err})")

    metrics = pd.DataFrame(metric_rows)

    pred_path = os.path.join(data_dir, "step5_predictions.csv")
    metrics_path = os.path.join(data_dir, "step5_model_horizon_metrics.csv")

    pred_out.to_csv(pred_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    plot_model_auc(
        metrics,
        os.path.join(fig_dir, "model_auc_by_prediction_horizon.png"),
    )

    print(f"Predictions saved to: {pred_path}")
    print(f"Model metrics saved to: {metrics_path}")

    return pred_out, metrics


# Step 6: Country-vintage analysis

def compute_vintage_country_gaps(df_clean: pd.DataFrame, min_n=50) -> pd.DataFrame:
    """Compare Bondora PoD with observed 1-year defaults by vintage and country."""
    obs = df_clean[
        df_clean["fully_observed_1y"]
        & df_clean["ProbabilityOfDefault"].notna()
        & (df_clean["ProbabilityOfDefault"] > 0)
    ].copy()

    obs["vintage"] = obs["LoanDate"].dt.year

    vintage = (
        obs.groupby(["vintage", "Country"])
        .agg(
            n=("default_1y", "size"),
            pod_mean=("ProbabilityOfDefault", "mean"),
            actual_1y=("default_1y", "mean"),
        )
        .round(4)
    )

    vintage["gap_pp"] = (
        (vintage["actual_1y"] - vintage["pod_mean"]) * 100
    ).round(2)

    return vintage[vintage["n"] >= min_n]


def plot_single_vintage(vintage_df: pd.DataFrame, year: int, output_path: str):
    """Plot PoD vs observed default rate for one vintage year."""
    if year not in vintage_df.index.get_level_values("vintage"):
        return

    df_year = vintage_df.loc[year].reset_index()
    countries = df_year["Country"].tolist()

    pod = df_year["pod_mean"].values * 100
    actual = df_year["actual_1y"].values * 100

    x = np.arange(len(countries))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_pod = ax.bar(
        x - width / 2,
        pod,
        width,
        label="Bondora PoD",
        color="#3b82f6",
        edgecolor="white",
    )
    bars_actual = ax.bar(
        x + width / 2,
        actual,
        width,
        label="Actual 1-year default",
        color="#ef4444",
        edgecolor="white",
    )

    for bar, value in zip(bars_pod, pod):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.5,
            f"{value:.1f}",
            ha="center",
            fontsize=10,
        )

    for bar, value in zip(bars_actual, actual):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.5,
            f"{value:.1f}",
            ha="center",
            fontsize=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{country}\nn={int(df_year.loc[i, 'n']):,}" for i, country in enumerate(countries)]
    )
    ax.set_ylabel("Rate (%)")
    ax.set_title(
        f"{year} vintage: PoD vs actual 1-year default",
        fontweight="bold",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    save_transparent_figure(fig, output_path, dpi=150)
    plt.close(fig)


def plot_vintage_gap_lines(vintage_df: pd.DataFrame, output_path: str):
    """Plot the PoD calibration gap across vintage years."""
    fig, ax = plt.subplots(figsize=(11, 5))

    countries_in_data = vintage_df.index.get_level_values("Country")

    for country in COUNTRIES:
        if country not in countries_in_data:
            continue

        country_df = vintage_df.xs(country, level="Country").reset_index()
        if country_df.empty:
            continue

        ax.plot(
            country_df["vintage"],
            country_df["gap_pp"],
            "o-",
            label=country,
            linewidth=2,
        )

    ax.axhline(0, ls="--", color="black", alpha=0.5)
    ax.set_xlabel("Vintage year")
    ax.set_ylabel("Gap: actual 1y default - PoD, pp")
    ax.set_title("PoD calibration gap by vintage and country", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_transparent_figure(fig, output_path, dpi=150)
    plt.close(fig)


def step6_country_vintage(df_clean, closed, data_dir, fig_dir, min_n=50):
    """Compare predicted and observed 1-year default rates by vintage and country."""
    print_step(6, "Country-vintage analysis")

    if "ProbabilityOfDefault" not in df_clean.columns:
        print("ProbabilityOfDefault not found; skipping vintage analysis.")
        return pd.DataFrame()

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    vintage = compute_vintage_country_gaps(df_clean, min_n=min_n)

    out_path = os.path.join(data_dir, "step6_vintage_country.csv")
    vintage.to_csv(out_path)

    plot_single_vintage(
        vintage,
        year=2015,
        output_path=os.path.join(fig_dir, "vintage_2015_calibration_gap.png"),
    )
    plot_vintage_gap_lines(
        vintage,
        output_path=os.path.join(fig_dir, "vintage_calibration_gap_by_country.png"),
    )

    print(f"Vintage-country table saved to: {out_path}")
    print(f"Rows retained after min_n={min_n}: {len(vintage):,}")

    return vintage


# Step 7: Conformal transfer

def conformal_binary_sets(p_cal, y_cal, p_test, y_test, alpha=0.10):
    """Split-conformal prediction sets for binary probability scores."""
    p_cal = np.asarray(p_cal, dtype=float)
    y_cal = np.asarray(y_cal, dtype=int)
    p_test = np.asarray(p_test, dtype=float)
    y_test = np.asarray(y_test, dtype=int)

    cal_scores = np.where(y_cal == 1, 1 - p_cal, p_cal)

    if len(cal_scores) == 0:
        return np.nan, np.nan, np.nan

    q_level = np.ceil((len(cal_scores) + 1) * (1 - alpha)) / len(cal_scores)
    qhat = np.quantile(cal_scores, min(q_level, 1.0), method="higher")

    include_0 = p_test <= qhat
    include_1 = (1 - p_test) <= qhat

    covered = np.where(y_test == 1, include_1, include_0)
    width = include_0.astype(int) + include_1.astype(int)

    return float(covered.mean()), float(width.mean()), float(qhat)


def lifetime_score_columns(pred: pd.DataFrame):
    """Return available lifetime score columns."""
    score_cols = ["pod_bondora"]
    score_cols.extend(
        col for col in pred.columns
        if col.startswith("pred_") and col.endswith("_lifetime")
    )
    return [col for col in score_cols if col in pred.columns]


def conformal_subset(
    pred: pd.DataFrame,
    score_col: str,
    country: str,
    split: str,
):
    """Select rows with valid score and lifetime target."""
    return pred[
        (pred["Country"] == country)
        & (pred["split"] == split)
        & pred[score_col].notna()
        & pred["y_lifetime"].notna()
    ].copy()


def evaluate_conformal_score(
    pred: pd.DataFrame,
    score_col: str,
    alpha: float,
    calibration_country: str = "EE",
    min_cal: int = 200,
    min_test: int = 100,
):
    """Evaluate one score column using one-country calibration and country-level tests."""
    target_coverage = 1 - alpha
    rows = []

    cal = conformal_subset(
        pred,
        score_col=score_col,
        country=calibration_country,
        split="cal",
    )

    if len(cal) < min_cal:
        return [{
            "score_col": score_col,
            "test_country": "ALL",
            "alpha": alpha,
            "target_coverage": target_coverage,
            "empirical_coverage": np.nan,
            "mean_width": np.nan,
            "qhat": np.nan,
            "n_cal": len(cal),
            "n_test": 0,
            "status": "insufficient_calibration_data",
        }]

    if cal["y_lifetime"].nunique() < 2:
        return [{
            "score_col": score_col,
            "test_country": "ALL",
            "alpha": alpha,
            "target_coverage": target_coverage,
            "empirical_coverage": np.nan,
            "mean_width": np.nan,
            "qhat": np.nan,
            "n_cal": len(cal),
            "n_test": 0,
            "status": "single_class_calibration",
        }]

    for country in COUNTRIES:
        test = conformal_subset(
            pred,
            score_col=score_col,
            country=country,
            split="test",
        )

        if len(test) < min_test:
            rows.append({
                "score_col": score_col,
                "test_country": country,
                "alpha": alpha,
                "target_coverage": target_coverage,
                "empirical_coverage": np.nan,
                "mean_width": np.nan,
                "qhat": np.nan,
                "n_cal": len(cal),
                "n_test": len(test),
                "status": "insufficient_test_data",
            })
            continue

        coverage, width, qhat = conformal_binary_sets(
            cal[score_col].values,
            cal["y_lifetime"].astype(int).values,
            test[score_col].values,
            test["y_lifetime"].astype(int).values,
            alpha=alpha,
        )

        rows.append({
            "score_col": score_col,
            "test_country": country,
            "alpha": alpha,
            "target_coverage": target_coverage,
            "empirical_coverage": coverage,
            "mean_width": width,
            "qhat": qhat,
            "n_cal": len(cal),
            "n_test": len(test),
            "status": "ok",
        })

    return rows


def plot_conformal_coverage(res: pd.DataFrame, output_path: str):
    """Plot empirical coverage by score and test country."""
    ok = res[res["status"] == "ok"].copy()
    if ok.empty:
        return

    target_coverage = ok["target_coverage"].iloc[0]

    pivot = ok.pivot(
        index="score_col",
        columns="test_country",
        values="empirical_coverage",
    )

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(8, max(3, 0.6 * len(pivot) + 2)))

    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap="RdYlGn",
        vmin=max(0, target_coverage - 0.15),
        vmax=min(1, target_coverage + 0.15),
    )

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.values[i, j]
            if pd.notna(value):
                text_color = (
                    "black"
                    if abs(value - target_coverage) < 0.1
                    else "white"
                )
                ax.text(
                    j,
                    i,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontweight="bold",
                    color=text_color,
                )

    ax.set_title(
        f"Conformal coverage by test country; target = {target_coverage:.2f}",
        fontweight="bold",
    )
    plt.colorbar(im, ax=ax, label="Empirical coverage")

    plt.tight_layout()
    save_transparent_figure(fig, output_path, dpi=150)
    plt.close(fig)


def step7_conformal_transfer(data_dir, fig_dir, alpha=0.10):
    """Evaluate split-conformal transfer from EE calibration to each test country."""
    print_step(7, "Conformal transfer")

    pred_path = os.path.join(data_dir, "step5_predictions.csv")
    if not os.path.exists(pred_path):
        print("step5_predictions.csv not found; skipping conformal transfer.")
        return pd.DataFrame()

    os.makedirs(fig_dir, exist_ok=True)

    pred = pd.read_csv(pred_path, parse_dates=["LoanDate"])
    score_cols = lifetime_score_columns(pred)

    print(f"Conformal base scores: {score_cols}")

    rows = []
    for score_col in score_cols:
        rows.extend(
            evaluate_conformal_score(
                pred,
                score_col=score_col,
                alpha=alpha,
                calibration_country="EE",
            )
        )

    res = pd.DataFrame(rows)

    out_path = os.path.join(data_dir, "step7_conformal_transfer.csv")
    res.to_csv(out_path, index=False)

    plot_conformal_coverage(
        res,
        os.path.join(fig_dir, "cross_country_conformal_coverage.png"),
    )

    ok = res[res["status"] == "ok"]
    if not ok.empty:
        print("Conformal coverage summary:")
        print(
            ok.pivot(
                index="score_col",
                columns="test_country",
                values="empirical_coverage",
            ).round(3)
        )

    print(f"Conformal transfer results saved to: {out_path}")
    return res


# Step 8: Output

def write_output_index(data_dir):
    """Write a short index of generated output files."""
    print_step(8, "Output index")

    files = {
        "closed_loans.csv": "closed-loan subset with constructed default labels",
        "loans_clean.csv": "cleaned loan-level frame used by later steps",
        "feature_columns.txt": "column groups before audit filtering",
        "step2_label_diagnostics.csv": "label base rates, L3/L4 overlap, cured defaults",
        "step3_pod_horizon_metrics.csv": "Bondora PoD metrics by target horizon",
        "step3_pod_sanity.csv": "sanity-check metrics for Bondora PoD",
        "step4_feature_audit.csv": "univariate leakage audit",
        "step5_predictions.csv": "model predictions for train/cal/test splits",
        "step5_model_horizon_metrics.csv": "test-set metrics for LR and GBDT",
        "step6_vintage_country.csv": "vintage-country PoD calibration gaps",
        "step7_conformal_transfer.csv": "conformal coverage by score and country",
    }

    lines = ["# Output files", ""]

    for filename, note in files.items():
        path = os.path.join(data_dir, filename)
        marker = "x" if os.path.exists(path) else " "
        lines.append(f"- [{marker}] `{filename}` — {note}")

    out_path = os.path.join(data_dir, "output_index.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Output index saved to: {out_path}")


# Main

def parse_pipeline_args():
    parser = argparse.ArgumentParser(
        description="Bondora credit-risk reanalysis pipeline"
    )
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="Path to LoanData.csv")
    parser.add_argument("--fig-dir", default="figures/pipeline", help="Figure output directory")
    parser.add_argument("--data-dir", default="data", help="Data output directory")
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip model fitting and conformal evaluation",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run only the data preparation, diagnostics, and feature audit steps",
    )
    parser.add_argument(
        "--max-rows-model",
        type=int,
        default=200000,
        help="Maximum number of rows used for model fitting",
    )
    parser.add_argument(
        "--top-k-cat",
        type=int,
        default=50,
        help="Top-K levels kept for each categorical feature",
    )
    parser.add_argument(
        "--gbdt-backend",
        choices=["auto", "lightgbm", "xgboost", "sklearn"],
        default="lightgbm",
        help=(
            "Tree-model implementation. The default requires LightGBM; "
            "use 'auto' only when fallback behavior is acceptable."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="Conformal miscoverage level",
    )
    return parser.parse_args()


def pipeline_main():
    args = parse_pipeline_args()
    ensure_dirs(args.fig_dir, args.data_dir)

    (
        df_clean,
        closed,
        feature_cols,
        _meta_cols,
        _bondora_cols,
        _label_cols,
        base_rates,
        spread,
    ) = step1_load_and_prepare(args.csv, args.data_dir, args.fig_dir)

    step2_label_diagnostics(
        df_clean,
        closed,
        base_rates,
        spread,
        args.fig_dir,
        args.data_dir,
    )

    step3_pod_diagnostics(
        df_clean,
        closed,
        args.data_dir,
        args.fig_dir,
    )

    safe_features, audit_df = step4_feature_audit(
        df_clean,
        closed,
        feature_cols,
        args.data_dir,
        args.fig_dir,
    )

    if args.audit_only:
        print("Audit-only mode: stopping after feature audit.")
        write_output_index(args.data_dir)
        return

    step6_country_vintage(
        df_clean,
        closed,
        args.data_dir,
        args.fig_dir,
    )

    if not args.skip_models:
        step5_clean_model_bakeoff(
            df_clean,
            closed,
            safe_features,
            audit_df,
            args.data_dir,
            args.fig_dir,
            max_rows_model=args.max_rows_model,
            top_k_cat=args.top_k_cat,
            gbdt_backend=args.gbdt_backend,
        )

        step7_conformal_transfer(
            args.data_dir,
            args.fig_dir,
            alpha=args.alpha,
        )
    else:
        print("Model-dependent steps skipped.")

    write_output_index(args.data_dir)


# Conformal master-file preparation

PREP_HORIZONS = ("1y", "lifetime")
PREP_SCORE_ALIASES = {
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

PREP_LABEL_ALIASES = {
    "L1": ["L1_strict_late", "L1", "label_L1"],
    "L2": ["L2_status_or_default", "L2", "label_L2"],
    "L3": ["L3_default_date", "L3", "label_L3", "y_lifetime"],
    "L4": ["L4_ever_60d_late", "L4", "label_L4"],
    "L5": ["L5_default_excl_cured", "L5", "label_L5"],
}


def read_conformal_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def first_present_conformal_prep(df: pd.DataFrame, names: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def merge_without_duplicate_columns(left: pd.DataFrame, right: pd.DataFrame, on: str) -> pd.DataFrame:
    keep = [on] + [c for c in right.columns if c != on and c not in left.columns]
    return left.merge(right[keep], on=on, how="left")


def add_public_label_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for short, names in PREP_LABEL_ALIASES.items():
        col = first_present_conformal_prep(out, names)
        if col is not None:
            out[f"label_{short}"] = out[col]

    if "label_L5" not in out.columns:
        dd_col = first_present_conformal_prep(out, ["DefaultDate", "default_date"])
        status_col = first_present_conformal_prep(out, ["Status", "status", "LoanStatus"])
        if dd_col and status_col:
            defaulted = pd.to_datetime(out[dd_col], errors="coerce").notna()
            status = out[status_col].astype(str).str.lower()
            out["label_L5"] = (defaulted & (status != "repaid")).astype(int)

    return out


def add_eligibility_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    split = out["split"].astype(str).str.lower() if "split" in out.columns else pd.Series("", index=out.index)
    heldout = split.isin(["cal", "test", "calibration", "eval", "evaluation"])

    if "y_1y" in out.columns:
        out["conformal_eligible_1y"] = heldout & out["y_1y"].notna()
    if "y_lifetime" in out.columns:
        out["conformal_eligible_lifetime"] = heldout & out["y_lifetime"].notna()

    for horizon in PREP_HORIZONS:
        flag = f"conformal_eligible_{horizon}"
        if flag not in out.columns:
            continue
        for score_name, by_horizon in PREP_SCORE_ALIASES.items():
            col = first_present_conformal_prep(out, by_horizon[horizon])
            if col is not None:
                out[f"{flag}_{score_name}"] = out[flag] & out[col].notna()
    return out


def build_master(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    predictions = read_conformal_table(Path(args.predictions or data_dir / "step5_predictions.csv"))
    closed = read_conformal_table(Path(args.closed or data_dir / "closed_loans.csv"))

    if "LoanId" not in predictions.columns or "LoanId" not in closed.columns:
        raise SystemExit("Both predictions and closed-loan files must contain LoanId.")

    master = merge_without_duplicate_columns(predictions, closed, "LoanId")
    master = add_public_label_aliases(master)
    master = add_eligibility_flags(master)

    out = Path(args.out or data_dir / "conformal_master.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() in {".parquet", ".pq"}:
        master.to_parquet(out, index=False)
    else:
        master.to_csv(out, index=False)

    print(f"Wrote {out} with {len(master):,} rows and {master.shape[1]} columns.")
    for horizon in PREP_HORIZONS:
        flag = f"conformal_eligible_{horizon}"
        if flag in master.columns:
            print(f"  {flag}: {int(master[flag].sum()):,} rows")


def load_master_or_predictions(args: argparse.Namespace) -> pd.DataFrame:
    path = Path(args.master or Path(args.data_dir) / "conformal_master.csv")
    if path.exists():
        return read_conformal_table(path)
    return read_conformal_table(Path(args.predictions or Path(args.data_dir) / "step5_predictions.csv"))


def check_tabpfn(args: argparse.Namespace) -> None:
    df = load_master_or_predictions(args)
    split = df["split"].astype(str).str.lower() if "split" in df.columns else pd.Series("", index=df.index)
    heldout = split.isin(["cal", "test", "calibration", "eval", "evaluation"])

    print("TabPFN prediction availability")
    for horizon in PREP_HORIZONS:
        col = first_present_conformal_prep(df, PREP_SCORE_ALIASES["tabpfn"][horizon])
        y_col = "y_1y" if horizon == "1y" else "y_lifetime"
        if col is None:
            print(f"  {horizon}: no TabPFN prediction column found")
            continue
        eligible = heldout & df[y_col].notna() if y_col in df.columns else heldout
        available = eligible & df[col].notna()
        denom = int(eligible.sum())
        numer = int(available.sum())
        rate = numer / denom if denom else np.nan
        print(f"  {horizon}: {col}, {numer:,}/{denom:,} held-out rows available ({rate:.1%})")


def diagnose_window(args: argparse.Namespace) -> None:
    df = load_master_or_predictions(args)
    date_col = first_present_conformal_prep(df, ["LoanDate", "loandate", "loan_date", "origination_date"])
    country_col = first_present_conformal_prep(df, ["Country", "country", "loancountry"])
    if date_col is None:
        raise SystemExit("No loan-date column found.")

    print("One-year conformal-universe diagnostics")
    masks = []
    if "y_1y" in df.columns:
        masks.append(("1y label observed", df["y_1y"].notna()))
    for name, aliases in [
        ("LR score present", PREP_SCORE_ALIASES["lr"]["1y"]),
        ("GBDT score present", PREP_SCORE_ALIASES["gbdt"]["1y"]),
        ("TabPFN score present", PREP_SCORE_ALIASES["tabpfn"]["1y"]),
        ("Bondora PoD present", PREP_SCORE_ALIASES["pod"]["1y"]),
    ]:
        col = first_present_conformal_prep(df, aliases)
        if col is not None:
            masks.append((name, df[col].notna()))
    if "conformal_eligible_1y" in df.columns:
        masks.append(("existing conformal_eligible_1y flag", df["conformal_eligible_1y"].astype(bool)))

    running = pd.Series(True, index=df.index)
    for name, mask in masks:
        running = running & mask
        sub = df.loc[running].copy()
        dates = pd.to_datetime(sub[date_col], errors="coerce").dropna()
        if dates.empty:
            span = "no valid dates"
        else:
            span = f"{dates.min().date()} to {dates.max().date()} (median {dates.median().date()})"
        print(f"\n{name}")
        print(f"  rows: {len(sub):,}")
        print(f"  date span: {span}")
        if country_col is not None and len(sub):
            counts = sub[country_col].value_counts(dropna=False).to_dict()
            print(f"  countries: {counts}")


def parse_prepare_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare inputs for Bondora conformal experiments.")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-master", help="Merge main pipeline outputs into a conformal master file.")
    build.add_argument("--data-dir", default="data")
    build.add_argument("--predictions", default=None)
    build.add_argument("--closed", default=None)
    build.add_argument("--out", default=None)
    build.set_defaults(func=build_master)

    check = sub.add_parser("check-tabpfn", help="Check TabPFN prediction coverage on held-out rows.")
    check.add_argument("--data-dir", default="data")
    check.add_argument("--master", default=None)
    check.add_argument("--predictions", default=None)
    check.set_defaults(func=check_tabpfn)

    diag = sub.add_parser("diagnose-window", help="Show which columns restrict the conformal universe.")
    diag.add_argument("--data-dir", default="data")
    diag.add_argument("--master", default=None)
    diag.add_argument("--predictions", default=None)
    diag.set_defaults(func=diagnose_window)

    return parser.parse_args()


def prepare_cli_main() -> None:
    args = parse_prepare_args()
    args.func(args)


# Final presentation figures

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
    "savefig.facecolor": "none",
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
                 f"        Run bondora_credit_risk_analysis.py pipeline first.\n")
    df = pd.read_csv(CLOSED_LOANS_PATH, low_memory=False)
    print(f"[load] closed_loans: {len(df):,} rows, {len(df.columns)} cols")
    return df


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        sys.exit(f"[error] not found: {PREDICTIONS_PATH}\n"
                 f"        Run bondora_credit_risk_analysis.py pipeline first.\n")
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

def plot_country_default_rates(closed: pd.DataFrame) -> pd.DataFrame:
    print("\n[figure] Three-country default rate")

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

    _save(fig, "country_default_rates")
    plt.close(fig)
    return summary.reset_index().rename(columns={country_col: "country"})


# Figure 2 : 5-label bars + Jaccard card

def plot_default_label_sensitivity(closed: pd.DataFrame) -> pd.DataFrame:
    print("\n[figure] Default-label sensitivity")

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

    _save(fig, "default_label_sensitivity")
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


def _plot_calibration_panel(
    ax,
    base: pd.DataFrame,
    ycol: str,
    mcols: Dict[str, Optional[str]],
    hname: str,
    frontier: set[str],
) -> List[dict]:
    """Draw one calibration panel and return its metric rows."""
    rows = []
    xs = np.linspace(0, 1, 50)
    ax.fill_between(xs, xs, 1.0, color="#E15759", alpha=0.04, linewidth=0,
                    zorder=1)
    ax.plot([0, 1], [0, 1], "--", color=REFERENCE_LINE, linewidth=1.2,
            label="Perfect calibration", zorder=2)

    full_data_models = [m for m in mcols if m not in frontier]
    fulldata_cols = [mcols[m] for m in full_data_models if mcols.get(m) is not None]
    common = comparison_frame(base, fulldata_cols, ycol)
    n_common = len(common)
    base_rate = float(common[ycol].astype(int).mean()) if n_common else float("nan")
    print(f"       {hname:<22s}: full-data common n={n_common:,}  "
          f"base_rate={base_rate:.4f}")

    metric_rows = []
    for name, color, ls, marker, lw, ms, alpha, z in _FIG3_MODEL_SPECS:
        mcol = mcols.get(name)
        if mcol is None or mcol not in base.columns:
            continue
        if name in frontier:
            d = base[[mcol, ycol]].dropna()
        elif n_common > 0:
            d = common[[mcol, ycol]]
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
        rows.append({"horizon": hname, "model": name,
                     "ECE": e, "Brier": bs, "n": len(d),
                     "base_rate": float(y_true.mean())})

    metric_rows.sort(key=lambda r: r[1])
    header = f"{'':10s}{'ECE':>7s}{'Brier':>8s}"
    sep = "-" * len(header)
    out_lines = [header, sep]
    for name, e, bs, nrow in metric_rows:
        tag = "" if nrow == n_common else " *"
        out_lines.append(f"{name:<10s}{e:7.3f}{bs:8.3f}{tag}")
    out_lines.append(sep)
    out_lines.append(f"n = {n_common:,}")
    for dn in sorted({nrow for *_, nrow in metric_rows if nrow != n_common}):
        out_lines.append(f"* n = {dn:,}")
    ax.text(0.03, 0.97, "\n".join(out_lines),
            transform=ax.transAxes, ha="left", va="top",
            family="monospace", fontsize=9.8,
            bbox=dict(boxstyle="round,pad=0.55",
                      facecolor="none", edgecolor="#777",
                      linewidth=0.9))

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Mean predicted PD")
    ax.set_ylabel("Observed default rate")
    ax.set_title(hname, pad=10)
    ax.legend(loc="lower right", fontsize=9.5,
              frameon=True, framealpha=0.92,
              fancybox=False, edgecolor="#999")
    return rows


def plot_pod_calibration(preds: pd.DataFrame) -> pd.DataFrame:
    print("\n[figure] Calibration by model and horizon")

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

    all_rows = []
    horizons = [
        ("1-year horizon", y1_col, horizon_cols["1y"],
         "pod_calibration_1year"),
        ("Lifetime horizon (L3)", ylife_col, horizon_cols["lifetime"],
         "pod_calibration_lifetime"),
    ]

    for hname, ycol, mcols, stem in horizons:
        fig, ax = plt.subplots(figsize=(7.2, 5.8))
        all_rows.extend(_plot_calibration_panel(ax, base, ycol, mcols, hname,
                                                FRONTIER))
        fig.suptitle(
            f"Calibration by model: {hname}",
            fontsize=13.5, fontweight="bold", y=1.02,
        )
        fig.text(
            0.5, -0.03,
            "Full-data models use common rows; TabPFN-3 uses its available "
            "prediction rows.",
            ha="center", fontsize=10.5, style="italic", color=TEXT_MUTED,
        )
        _save(fig, stem)
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


def plot_brier_score_decomposition(preds: pd.DataFrame) -> pd.DataFrame:
    print("\n[figure] Brier component comparison")

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

    _save(fig, "brier_score_decomposition")
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


def plot_entry_vintage_default_rates(closed: pd.DataFrame) -> pd.DataFrame:
    print("\n[figure] Entry-vintage and 2017 default rates")

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

    _save(fig, "entry_vintage_default_rates")
    plt.close(fig)
    return table


# I/O HELPERS

def _save(fig, stem: str) -> None:
    out = FIG_DIR / f"{stem}.png"
    save_transparent_figure(fig, out, dpi=220)
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

def phase_a_figures_main():
    parser = argparse.ArgumentParser(
        description="Create figures for the Bondora credit-risk analysis",
    )
    parser.add_argument(
        "--only",
        choices=[
            "country-defaults",
            "label-sensitivity",
            "calibration",
            "brier",
            "entry-vintages",
        ],
                        help="Run only one figure")
    args = parser.parse_args()

    print("=" * 72)
    print("Bondora analysis figures")
    print(f"Color palette: EE={COUNTRY_COLORS['EE']}  "
          f"FI={COUNTRY_COLORS['FI']}  ES={COUNTRY_COLORS['ES']}")
    print("=" * 72)

    closed = None
    preds = None
    if args.only in (None, "country-defaults", "label-sensitivity", "entry-vintages"):
        closed = load_closed_loans()
    if args.only in (None, "calibration", "brier"):
        preds = load_predictions()

    parts: Dict[str, pd.DataFrame] = {}
    if args.only in (None, "country-defaults"):
        parts["country_default_rates"] = plot_country_default_rates(closed)
    if args.only in (None, "label-sensitivity"):
        parts["default_label_sensitivity"] = plot_default_label_sensitivity(closed)
    if args.only in (None, "calibration"):
        parts["pod_calibration"] = plot_pod_calibration(preds)
    if args.only in (None, "brier"):
        parts["brier_score_decomposition"] = plot_brier_score_decomposition(preds)
    if args.only in (None, "entry-vintages"):
        parts["entry_vintage_default_rates"] = plot_entry_vintage_default_rates(closed)

    write_summary(parts)

    print("\n" + "=" * 72)
    print(f"Done. Figures in:  {FIG_DIR}")
    print("=" * 72)


def main():
    """Dispatch the compact submission script into its analysis stages."""
    command = sys.argv[1] if len(sys.argv) > 1 else "pipeline"
    aliases = {
        "pipeline": pipeline_main,
        "figures": phase_a_figures_main,
        "build-master": prepare_cli_main,
        "check-tabpfn": prepare_cli_main,
        "diagnose-window": prepare_cli_main,
    }
    if command not in aliases:
        valid = ", ".join(sorted(aliases))
        raise SystemExit(f"Unknown command '{command}'. Valid commands: {valid}")
    if command in {"pipeline", "figures"}:
        sys.argv = [sys.argv[0], *sys.argv[2:]]
    aliases[command]()


if __name__ == "__main__":
    main()
