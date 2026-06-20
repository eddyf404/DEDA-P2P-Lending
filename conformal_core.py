"""Core utilities for binary conformal prediction.

The functions in this module assume an already fitted probability score
``p_hat(x) = P(Y=1 | X=x)``. They do not train models.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


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
