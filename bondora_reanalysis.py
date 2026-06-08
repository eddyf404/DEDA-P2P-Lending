"""Bondora credit-risk reanalysis.

The pipeline constructs default labels, audits candidate features, trains
baseline models, and exports the tables and figures used in the analysis.
"""


import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore", category=FutureWarning)

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
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
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
        os.path.join(fig_dir, "feature_audit_auc_distribution.png"),
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
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
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
        os.path.join(fig_dir, "model_auc_by_horizon.png"),
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
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
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
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
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
        output_path=os.path.join(fig_dir, "vintage_2015_country_gap.png"),
    )
    plot_vintage_gap_lines(
        vintage,
        output_path=os.path.join(fig_dir, "vintage_gap_by_country.png"),
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
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
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
        os.path.join(fig_dir, "conformal_coverage_by_country.png"),
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

def parse_args():
    parser = argparse.ArgumentParser(
        description="Bondora credit-risk reanalysis pipeline"
    )
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="Path to LoanData.csv")
    parser.add_argument("--fig-dir", default="fig", help="Figure output directory")
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


def main():
    args = parse_args()
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


if __name__ == "__main__":
    main()
