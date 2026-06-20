"""Prepare derived inputs for the conformal-inference experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HORIZONS = ("1y", "lifetime")
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

LABEL_ALIASES = {
    "L1": ["L1_strict_late", "L1", "label_L1"],
    "L2": ["L2_status_or_default", "L2", "label_L2"],
    "L3": ["L3_default_date", "L3", "label_L3", "y_lifetime"],
    "L4": ["L4_ever_60d_late", "L4", "label_L4"],
    "L5": ["L5_default_excl_cured", "L5", "label_L5"],
}


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


def merge_without_duplicate_columns(left: pd.DataFrame, right: pd.DataFrame, on: str) -> pd.DataFrame:
    keep = [on] + [c for c in right.columns if c != on and c not in left.columns]
    return left.merge(right[keep], on=on, how="left")


def add_public_label_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for short, names in LABEL_ALIASES.items():
        col = first_present(out, names)
        if col is not None:
            out[f"label_{short}"] = out[col]

    if "label_L5" not in out.columns:
        dd_col = first_present(out, ["DefaultDate", "default_date"])
        status_col = first_present(out, ["Status", "status", "LoanStatus"])
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

    for horizon in HORIZONS:
        flag = f"conformal_eligible_{horizon}"
        if flag not in out.columns:
            continue
        for score_name, by_horizon in SCORE_ALIASES.items():
            col = first_present(out, by_horizon[horizon])
            if col is not None:
                out[f"{flag}_{score_name}"] = out[flag] & out[col].notna()
    return out


def build_master(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    predictions = read_table(Path(args.predictions or data_dir / "step5_predictions.csv"))
    closed = read_table(Path(args.closed or data_dir / "closed_loans.csv"))

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
    for horizon in HORIZONS:
        flag = f"conformal_eligible_{horizon}"
        if flag in master.columns:
            print(f"  {flag}: {int(master[flag].sum()):,} rows")


def load_master_or_predictions(args: argparse.Namespace) -> pd.DataFrame:
    path = Path(args.master or Path(args.data_dir) / "conformal_master.csv")
    if path.exists():
        return read_table(path)
    return read_table(Path(args.predictions or Path(args.data_dir) / "step5_predictions.csv"))


def check_tabpfn(args: argparse.Namespace) -> None:
    df = load_master_or_predictions(args)
    split = df["split"].astype(str).str.lower() if "split" in df.columns else pd.Series("", index=df.index)
    heldout = split.isin(["cal", "test", "calibration", "eval", "evaluation"])

    print("TabPFN prediction availability")
    for horizon in HORIZONS:
        col = first_present(df, SCORE_ALIASES["tabpfn"][horizon])
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
    date_col = first_present(df, ["LoanDate", "loandate", "loan_date", "origination_date"])
    country_col = first_present(df, ["Country", "country", "loancountry"])
    if date_col is None:
        raise SystemExit("No loan-date column found.")

    print("One-year conformal-universe diagnostics")
    masks = []
    if "y_1y" in df.columns:
        masks.append(("1y label observed", df["y_1y"].notna()))
    for name, aliases in [
        ("LR score present", SCORE_ALIASES["lr"]["1y"]),
        ("GBDT score present", SCORE_ALIASES["gbdt"]["1y"]),
        ("TabPFN score present", SCORE_ALIASES["tabpfn"]["1y"]),
        ("Bondora PoD present", SCORE_ALIASES["pod"]["1y"]),
    ]:
        col = first_present(df, aliases)
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


def parse_args() -> argparse.Namespace:
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


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
