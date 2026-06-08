# -*- coding: utf-8 -*-
"""Run the hosted TabPFN classifier on prepared Bondora data.

This script sends preprocessed feature matrices to the Prior Labs API. Remote
processing is disabled unless ``--allow-remote-processing`` is supplied.
Authentication is read from ``PRIORLABS_API_KEY`` or ``TABPFN_API_KEY``.
"""

import argparse
import inspect
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent / "data"
FIG_DIR = Path(__file__).parent / "fig"
FIG_DIR.mkdir(exist_ok=True)

COUNTRIES = ["EE", "FI", "ES"]

# Hard blacklist: these fields must never be used as model inputs.
NEVER_FEATURES = {
    # identifiers / split helpers
    "LoanId",
    "LoanDate",
    "split",

    # 1-year target and fields derived from the default event date
    "default_1y",
    "y_1y",
    "fully_observed_1y",
    "days_to_default",

    # lifetime targets and constructed label definitions
    "y_lifetime",
    "L1_strict_late",
    "L2_status_or_default",
    "L3_default_date",
    "L4_ever_60d_late",
    "L5_default_excl_cured",

    # raw outcome fields
    "Status",
    "DefaultDate",
    "WorseLateCategory",

    # Bondora scores are baselines, not inputs to the benchmark model
    "ProbabilityOfDefault",
    "ExpectedLoss",
    "LossGivenDefault",
    "ExpectedReturn",

    # obvious post-origination collection / arrears fields
    "CurrentDebtDaysPrimary",
    "CurrentDebtDaysSecondary",
    "DebtOccuredOn",
    "DebtOccuredOnForSecondary",
    "ContractEndDate",
    "LastPaymentOn",
    "LoanStatusActiveFrom",
    "Restructured",
    "WorkoutProcessingType",
    "PrincipalOverdueBySchedule",
    "NextPaymentNr",
    "ActiveScheduleFirstPaymentReached",
}


def is_never_feature(name: str) -> bool:
    """Return True for fields that are target-derived or clearly post-origination."""
    if name in NEVER_FEATURES:
        return True

    lowered = name.lower()
    blocked_tokens = [
        "default",
        "worselate",
        "recovery",
        "writeoff",
        "debtservicing",
        "postdefault",
        "pastdue",
        "overdue",
        "latecategory",
    ]
    return any(token in lowered for token in blocked_tokens)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing loans_clean.csv, feature_columns.txt, and step5_predictions.csv.",
    )
    p.add_argument(
        "--dry-run-features",
        action="store_true",
        help="Only load/filter features and print the final feature list; do not call TabPFN.",
    )
    p.add_argument(
        "--tabpfn-n",
        type=int,
        default=8000,
        help="Stratified train subsample size sent to the hosted TabPFN API.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--n-estimators",
        type=int,
        default=8,
        help="TabPFN ensemble estimators, if supported by the installed client.",
    )
    p.add_argument(
        "--predict-batch-size",
        type=int,
        default=512,
        help="Rows per API predict_proba call. Smaller values are more stable for hosted API runs.",
    )
    p.add_argument(
        "--max-predict-retries",
        type=int,
        default=3,
        help="Retry each prediction chunk this many times before failing the horizon.",
    )
    p.add_argument(
        "--retry-sleep-s",
        type=float,
        default=10.0,
        help="Seconds to wait between failed prediction-chunk retries.",
    )
    p.add_argument(
        "--skip-cal-pred",
        action="store_true",
        help="Only predict the test split. Faster and more stable, but no TabPFN cal predictions for conformal transfer.",
    )
    p.add_argument(
        "--top-k-cat",
        type=int,
        default=50,
        help="Keep top-K categorical levels based on train split only.",
    )
    p.add_argument(
        "--allow-remote-processing",
        action="store_true",
        help=(
            "Confirm that the selected feature data may be sent to the "
            "Prior Labs hosted API."
        ),
    )
    p.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Hosted model identifier. Leave empty to use the API default "
            "latest TabPFN model, or pass a value listed by the client."
        ),
    )
    p.add_argument(
        "--thinking-effort",
        type=str,
        default=None,
        choices=[None, "low", "medium", "high"],
        help="Optional TabPFN-3-Plus thinking mode effort, if supported.",
    )
    p.add_argument(
        "--thinking-timeout-s",
        type=int,
        default=None,
        help="Optional thinking-mode timeout in seconds, if supported.",
    )
    return p.parse_args()


def ece_score(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    binids = np.clip(binids, 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        mask = binids == b
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(bin_acc - bin_conf)
    return float(ece)


def load_v2_outputs():
    """Read existing v2 outputs and rebuild the frame needed for TabPFN."""
    print("[load] reading v2 outputs ...")

    pred_path = DATA_DIR / "step5_predictions.csv"
    clean_path = DATA_DIR / "loans_clean.csv"
    feat_file = DATA_DIR / "feature_columns.txt"

    pred = pd.read_csv(pred_path, parse_dates=["LoanDate"])
    clean = pd.read_csv(clean_path, parse_dates=["LoanDate"], low_memory=False)

    safe_feats = []
    with open(feat_file, encoding="utf-8") as f:
        for line in f:
            if line.startswith("feature\t"):
                safe_feats.append(line.split("\t")[1].strip())

    print(f"  raw feature_columns entries: {len(safe_feats)}")

    audit_p = DATA_DIR / "step4_feature_audit.csv"
    if audit_p.exists():
        audit = pd.read_csv(audit_p)
        if {"feature", "audit_flag"}.issubset(audit.columns):
            approved = set(
                audit.loc[audit["audit_flag"] == "safe", "feature"].tolist()
            )
            audit_drop = set(safe_feats) - approved
            safe_feats = [f for f in safe_feats if f in approved]
        else:
            audit_drop = set(audit.loc[
                (audit["auc"] >= 0.85) | (audit["auc"] <= 0.15),
                "feature"
            ].tolist())
            safe_feats = [f for f in safe_feats if f not in audit_drop]
        print(
            f"  step4 audit drops: {len(audit_drop)}; "
            f"remaining candidate features: {len(safe_feats)}"
        )
    else:
        print("  warning: step4_feature_audit.csv not found; using hard blacklist only.")

    before_hard_filter = list(safe_feats)
    hard_drops = [f for f in before_hard_filter if is_never_feature(f)]
    safe_feats = [f for f in before_hard_filter if not is_never_feature(f)]
    if hard_drops:
        print("  hard blacklist drops:")
        for f in sorted(hard_drops):
            print(f"    - {f}")

    bad_remaining = sorted(f for f in safe_feats if is_never_feature(f))
    if bad_remaining:
        raise RuntimeError(
            "Leaky target-derived features still present after filtering: "
            + ", ".join(bad_remaining)
        )

    pred_keys = pred[["LoanId", "split", "y_1y", "y_lifetime", "fully_observed_1y"]].copy()
    overlap = set(pred_keys.columns) - {"LoanId"}
    clean = clean.drop(columns=[c for c in overlap if c in clean.columns])
    df = clean.merge(pred_keys, on="LoanId", how="inner")

    missing_feats = [f for f in safe_feats if f not in df.columns]
    if missing_feats:
        print(f"  warning: dropping {len(missing_feats)} features not found in merged frame")
        safe_feats = [f for f in safe_feats if f in df.columns]

    if not safe_feats:
        raise ValueError("No usable safe features found after filtering.")

    final_feature_path = DATA_DIR / "tabpfn3_safe_features.txt"
    with open(final_feature_path, "w", encoding="utf-8") as f:
        for col in sorted(safe_feats):
            f.write(f"{col}\n")

    print(f"  merged frame: {len(df):,} rows × {len(safe_feats)} safe features")
    print(f"  wrote final feature list: {final_feature_path}")
    return df, pred, safe_feats


class FeaturePreprocessor:
    """
    Fit only on train split to avoid cal/test preprocessing leakage.

    Preprocessing consists of:
      - datetime expansion
      - categorical top-K capping
      - ordinal encoding
      - numeric median imputation

    The implementation matches the preprocessing used by the local baselines.
    """

    def __init__(self, feat_cols, top_k_cat=50):
        self.feat_cols = list(feat_cols)
        self.top_k_cat = top_k_cat

        self.datetime_cols_ = None
        self.num_cols_ = None
        self.cat_cols_ = None
        self.final_cols_ = None

        self.top_levels_ = {}
        self.oe_ = None
        self.imp_ = None

    def _expand_datetime(self, X, fit=False):
        X = X.copy()

        if fit:
            self.datetime_cols_ = [
                c for c in X.columns if pd.api.types.is_datetime64_any_dtype(X[c])
            ]

        for c in self.datetime_cols_:
            dt = pd.to_datetime(X[c], errors="coerce")
            X[f"{c}_year"] = dt.dt.year
            X[f"{c}_month"] = dt.dt.month
            X[f"{c}_days"] = (dt - pd.Timestamp("1970-01-01")).dt.days
            X = X.drop(columns=[c])

        return X

    def fit(self, df_train):
        X = df_train[self.feat_cols].copy()
        X = self._expand_datetime(X, fit=True)

        self.num_cols_ = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        self.cat_cols_ = [c for c in X.columns if c not in self.num_cols_]
        self.final_cols_ = list(X.columns)

        if self.num_cols_:
            X[self.num_cols_] = X[self.num_cols_].replace([np.inf, -np.inf], np.nan)

            # Avoid all-missing numeric columns breaking SimpleImputer behavior.
            for c in self.num_cols_:
                if X[c].notna().sum() == 0:
                    X[c] = 0.0

            self.imp_ = SimpleImputer(strategy="median")
            self.imp_.fit(X[self.num_cols_])

        if self.cat_cols_:
            X_cat = pd.DataFrame(index=X.index)
            for c in self.cat_cols_:
                s = X[c].astype("string").fillna("__MISSING__")
                top = set(s.value_counts().head(self.top_k_cat).index)
                self.top_levels_[c] = top
                X_cat[c] = np.where(s.isin(top), s, "__OTHER__")

            self.oe_ = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            self.oe_.fit(X_cat.astype(str))

        return self

    def transform(self, df):
        X = df[self.feat_cols].copy()
        X = self._expand_datetime(X, fit=False)

        # Ensure same columns/order as fit.
        for c in self.final_cols_:
            if c not in X.columns:
                X[c] = np.nan
        X = X[self.final_cols_]

        if self.num_cols_:
            X[self.num_cols_] = X[self.num_cols_].replace([np.inf, -np.inf], np.nan)
            for c in self.num_cols_:
                if X[c].notna().sum() == 0:
                    X[c] = 0.0
            X[self.num_cols_] = self.imp_.transform(X[self.num_cols_])

        if self.cat_cols_:
            X_cat = pd.DataFrame(index=X.index)
            for c in self.cat_cols_:
                s = X[c].astype("string").fillna("__MISSING__")
                top = self.top_levels_[c]
                X_cat[c] = np.where(s.isin(top), s, "__OTHER__")

            X[self.cat_cols_] = self.oe_.transform(X_cat.astype(str))

        X = X.fillna(-1)
        return X.values.astype(np.float32)


def stratified_subsample(df, target_col, n, seed):
    """
    Proportional stratified subsample within train split by target × country.
    Not class-balanced; it preserves approximate original proportions.
    """
    train = df[(df["split"] == "train") & df[target_col].notna()].copy()

    if len(train) <= n:
        return train

    rng = np.random.RandomState(seed)
    train["_strat"] = (
        train[target_col].astype(int).astype(str)
        + "_"
        + train["Country"].astype("string").fillna("__MISSING__").astype(str)
    )

    counts = train["_strat"].value_counts()
    raw = counts / counts.sum() * n
    alloc = np.floor(raw).astype(int)

    # Ensure non-empty allocation where possible.
    if n >= len(alloc):
        alloc[alloc == 0] = 1
    else:
        alloc[:] = 0
        for k in raw.sort_values(ascending=False).head(n).index:
            alloc.loc[k] = 1

    # Adjust to exactly n.
    while alloc.sum() < n:
        remainders = (raw - np.floor(raw)).sort_values(ascending=False)
        for k in remainders.index:
            if alloc.sum() >= n:
                break
            if alloc.loc[k] < counts.loc[k]:
                alloc.loc[k] += 1

    while alloc.sum() > n:
        remainders = (raw - np.floor(raw)).sort_values(ascending=True)
        for k in remainders.index:
            if alloc.sum() <= n:
                break
            if alloc.loc[k] > 1:
                alloc.loc[k] -= 1

    parts = []
    for strat, g in train.groupby("_strat", group_keys=False):
        k = int(min(alloc.loc[strat], len(g)))
        if k > 0:
            parts.append(g.sample(n=k, random_state=rng.randint(0, 2**31 - 1)))

    sampled = pd.concat(parts, axis=0)

    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=seed)
    elif len(sampled) < n:
        remaining = train.drop(index=sampled.index)
        extra = remaining.sample(n=n - len(sampled), random_state=seed)
        sampled = pd.concat([sampled, extra], axis=0)

    return sampled.drop(columns=["_strat"])



def get_priorlabs_token():
    """Return a Prior Labs API token from the environment, if available."""
    return (
        os.getenv("PRIORLABS_API_KEY")
        or os.getenv("TABPFN_API_KEY")
    )


def make_tabpfn3_classifier(
    n_estimators,
    seed,
    model_path=None,
    thinking_effort=None,
    thinking_timeout_s=None,
):
    """
    Create an API-backed TabPFN classifier via tabpfn-client.

    Fit and prediction calls are handled by the hosted Prior Labs API client.
    """
    try:
        import tabpfn_client
        from tabpfn_client import TabPFNClassifier
    except Exception as e:
        raise RuntimeError(
            "Could not import the hosted TabPFN client. Run: "
            "pip install --upgrade tabpfn-client"
        ) from e

    token = get_priorlabs_token()
    if token:
        if hasattr(tabpfn_client, "set_access_token"):
            tabpfn_client.set_access_token(token)
        else:
            raise RuntimeError(
                "Installed tabpfn-client does not expose set_access_token(). "
                "Run: pip install --upgrade tabpfn-client"
            )
    else:
        print(
            "  warning: no PRIORLABS_API_KEY/TABPFN_API_KEY found. "
            "tabpfn-client may open an interactive login flow."
        )

    sig = inspect.signature(TabPFNClassifier)
    candidate_kwargs = {
        "random_state": seed,
        "n_estimators": n_estimators,
        "model_path": model_path,
        "thinking_effort": thinking_effort,
        "thinking_timeout_s": thinking_timeout_s,
    }
    kwargs = {
        k: v
        for k, v in candidate_kwargs.items()
        if v is not None and k in sig.parameters
    }

    ignored = [
        k for k, v in candidate_kwargs.items()
        if v is not None and k not in sig.parameters
    ]
    if ignored:
        print(f"  note: installed tabpfn-client ignores unsupported args: {ignored}")

    return TabPFNClassifier(**kwargs)

def positive_class_index(clf):
    classes = np.asarray(getattr(clf, "classes_", [0, 1]))
    pos = np.where(classes == 1)[0]
    if len(pos) != 1:
        raise ValueError(f"Could not find positive class 1 in clf.classes_: {classes}")
    return int(pos[0])


def predict_proba_chunks(clf, X, batch_size=512, max_retries=3, retry_sleep_s=10.0):
    """Predict a preprocessed matrix in small API calls with retry."""
    if X is None or len(X) == 0:
        return np.array([], dtype=np.float32)

    pos_idx = positive_class_index(clf)
    out = []
    n = len(X)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = X[start:end]

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                proba = clf.predict_proba(chunk)
                out.append(proba[:, pos_idx])
                print(f"    predicted rows {start:,}-{end - 1:,} / {n:,}")
                break
            except Exception as e:
                last_error = e
                print(
                    f"    warning: predict chunk {start:,}-{end - 1:,} failed "
                    f"(attempt {attempt}/{max_retries}): {e}"
                )
                if attempt < max_retries:
                    time.sleep(retry_sleep_s)
        else:
            raise RuntimeError(
                f"Prediction failed after {max_retries} attempts for rows "
                f"{start:,}-{end - 1:,}"
            ) from last_error

    return np.concatenate(out).astype(np.float32)


def predict_frame_chunks(
    clf,
    pre,
    frame,
    batch_size=512,
    max_retries=3,
    retry_sleep_s=10.0,
):
    """Transform and predict a dataframe in chunks, avoiding one huge X_test upload."""
    if frame is None or len(frame) == 0:
        return np.array([], dtype=np.float32)

    pos_idx = positive_class_index(clf)
    out = []
    n = len(frame)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        X_chunk = pre.transform(frame.iloc[start:end])

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                proba = clf.predict_proba(X_chunk)
                out.append(proba[:, pos_idx])
                print(f"    predicted rows {start:,}-{end - 1:,} / {n:,}")
                break
            except Exception as e:
                last_error = e
                print(
                    f"    warning: predict chunk {start:,}-{end - 1:,} failed "
                    f"(attempt {attempt}/{max_retries}): {e}"
                )
                if attempt < max_retries:
                    time.sleep(retry_sleep_s)
        else:
            raise RuntimeError(
                f"Prediction failed after {max_retries} attempts for rows "
                f"{start:,}-{end - 1:,}"
            ) from last_error

    return np.concatenate(out).astype(np.float32)


def main():
    args = parse_args()

    global DATA_DIR
    DATA_DIR = Path(args.data_dir)

    np.random.seed(args.seed)

    model_name = "tabpfn3"
    print(
        f"[config] model={model_name} backend=priorlabs_api "
        f"n_estimators={args.n_estimators} model_path={args.model_path or 'api_default'}"
    )

    df, pred, safe_feats = load_v2_outputs()

    if args.dry_run_features:
        print("\n[dry-run] final TabPFN input features:")
        for col in sorted(safe_feats):
            print(f"  - {col}")
        print("\nDry run complete. No API call was made.")
        return

    if not args.allow_remote_processing:
        raise SystemExit(
            "Remote processing is disabled. Review the feature list with "
            "`--dry-run-features`, confirm the dataset terms, then rerun with "
            "`--allow-remote-processing`."
        )

    if not get_priorlabs_token():
        raise SystemExit(
            "No API token found. Set PRIORLABS_API_KEY or TABPFN_API_KEY "
            "in the environment."
        )

    pred_out = pred.set_index("LoanId").copy()

    new_metrics = []
    updated_horizons = []

    for horizon, target_col, eligible_col in [
        ("1y", "y_1y", "fully_observed_1y"),
        ("lifetime", "y_lifetime", None),
    ]:
        print(f"\n=== TabPFN-3 on {horizon} ===")

        eligible = df[target_col].notna()
        if eligible_col == "fully_observed_1y":
            eligible &= df["fully_observed_1y"].fillna(False).astype(bool)

        full_train = df[(df["split"] == "train") & eligible].copy()
        sub_train = stratified_subsample(df[eligible], target_col, args.tabpfn_n, args.seed)

        idx_cal = (df["split"] == "cal") & eligible
        idx_test = (df["split"] == "test") & eligible

        col = f"pred_{model_name}_{horizon}"

        print(
            f"  full train: {len(full_train):,}  "
            f"train subsample: {len(sub_train):,}  "
            f"cal: {idx_cal.sum():,}  test: {idx_test.sum():,}"
        )

        if len(sub_train) == 0 or idx_test.sum() == 0:
            print(f"  warning: skipping {horizon}: empty train or test set")
            continue

        y_train = sub_train[target_col].astype(int).values
        if len(np.unique(y_train)) < 2:
            print(f"  warning: skipping {horizon}: train subsample has one class")
            continue

        print("  [prep] fitting preprocessing on train split only ...")
        pre = FeaturePreprocessor(safe_feats, top_k_cat=args.top_k_cat)
        pre.fit(full_train)

        X_train = pre.transform(sub_train)

        print(
            f"  X_train: {X_train.shape}  "
            f"cal rows: {idx_cal.sum():,}  test rows: {idx_test.sum():,}  "
            f"predict batch: {args.predict_batch_size:,}"
        )

        t0 = time.time()
        try:
            clf = make_tabpfn3_classifier(
                n_estimators=args.n_estimators,
                seed=args.seed,
                model_path=args.model_path,
                thinking_effort=args.thinking_effort,
                thinking_timeout_s=args.thinking_timeout_s,
            )
            clf.fit(X_train, y_train)

            p_cal = None
            if not args.skip_cal_pred and idx_cal.sum() > 0:
                print("  [predict] cal split ...")
                p_cal = predict_frame_chunks(
                    clf,
                    pre,
                    df.loc[idx_cal],
                    batch_size=args.predict_batch_size,
                    max_retries=args.max_predict_retries,
                    retry_sleep_s=args.retry_sleep_s,
                )

            print("  [predict] test split ...")
            p_test = predict_frame_chunks(
                clf,
                pre,
                df.loc[idx_test],
                batch_size=args.predict_batch_size,
                max_retries=args.max_predict_retries,
                retry_sleep_s=args.retry_sleep_s,
            )

        except Exception as e:
            print(f"  warning: TabPFN-3 failed on {horizon}: {e}")
            print(f"     Skipping {horizon}.")
            continue

        print(f"  TabPFN-3 {horizon}: {(time.time() - t0) / 60:.1f} min")

        if p_cal is not None:
            cal_ids = df.loc[idx_cal, "LoanId"].values
        test_ids = df.loc[idx_test, "LoanId"].values

        updated_col = pd.Series(np.nan, index=pred_out.index, dtype=float)
        if p_cal is not None:
            updated_col.loc[cal_ids] = p_cal
        updated_col.loc[test_ids] = p_test
        pred_out[col] = updated_col
        updated_horizons.append(horizon)

        y_test = df.loc[idx_test, target_col].astype(int).values
        auc = roc_auc_score(y_test, p_test)
        ece = ece_score(y_test, p_test, n_bins=10)
        brier = brier_score_loss(y_test, p_test)

        print(f"  Test AUC={auc:.4f}  ECE={ece:.3f}  Brier={brier:.3f}")

        new_metrics.append(
            dict(
                model=model_name,
                horizon=horizon,
                auc=auc,
                ece=ece,
                brier=brier,
                n_train=len(sub_train),
                n_test=int(idx_test.sum()),
                n_features=len(safe_feats),
                backend="priorlabs_api",
                model_path=args.model_path or "api_default",
                n_estimators=args.n_estimators,
            )
        )

    if updated_horizons:
        pred_path = DATA_DIR / "step5_predictions.csv"
        pred_tmp = DATA_DIR / "step5_predictions.csv.tmp"
        pred_out.reset_index().to_csv(pred_tmp, index=False)
        os.replace(pred_tmp, pred_path)
        print(
            "\nUpdated step5_predictions.csv for: "
            + ", ".join(updated_horizons)
        )
    else:
        print("\nNo successful horizons; existing predictions were not changed.")

    if new_metrics:
        m_path = DATA_DIR / "step5_model_horizon_metrics.csv"
        old = pd.read_csv(m_path) if m_path.exists() else pd.DataFrame()
        new_df = pd.DataFrame(new_metrics)

        # Replace previous tabpfn3 rows for these horizons instead of repeatedly appending duplicates.
        if not old.empty and {"model", "horizon"}.issubset(old.columns):
            old = old[
                ~(
                    (old["model"] == model_name)
                    & (old["horizon"].isin(new_df["horizon"]))
                )
            ]

        metrics_tmp = m_path.with_suffix(m_path.suffix + ".tmp")
        pd.concat([old, new_df], ignore_index=True).to_csv(
            metrics_tmp,
            index=False,
        )
        os.replace(metrics_tmp, m_path)
        print(f"Wrote {len(new_metrics)} TabPFN-3 metric rows to {m_path}")

    print("\nRun make_figures.py to refresh figures.")


if __name__ == "__main__":
    main()
