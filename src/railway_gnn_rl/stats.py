"""Statistics helpers.

* mean_ci_t:            mean with a Student-t confidence interval (n-1 dof);
                        used across training seeds (n = 5 or 3).
* bootstrap_ci:         percentile bootstrap CI of the mean (resampling values).
* paired_bootstrap_ci:  percentile bootstrap CI of the mean per-scenario
                        difference a_i - b_i (scenarios resampled as pairs).
* wilcoxon_signed_rank: two-sided Wilcoxon signed-rank test on a_i - b_i
                        (zero differences dropped; scipy, exact when possible).
* cohens_dz:            mean(d) / sd(d) of the paired differences (effect size).
* optimality_gap:       (mean W_policy - mean W_opt) / mean W_opt (ratio of means,
                        defined even when individual scenarios have W_opt = 0).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps


def _arr(values) -> np.ndarray:
    a = np.asarray(list(values), dtype=float)
    if a.size == 0:
        raise ValueError("empty input")
    return a


def mean_ci_t(values, conf: float = 0.95) -> tuple[float, float, float]:
    a = _arr(values)
    m = float(a.mean())
    if a.size < 2:
        return m, float("nan"), float("nan")
    half = float(sps.t.ppf(0.5 + conf / 2, a.size - 1) * a.std(ddof=1) / math.sqrt(a.size))
    return m, m - half, m + half


def bootstrap_ci(values, n_boot: int = 10_000, conf: float = 0.95, seed: int = 0):
    a = _arr(values)
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    lo, hi = np.quantile(means, [0.5 - conf / 2, 0.5 + conf / 2])
    return float(a.mean()), float(lo), float(hi)


def paired_bootstrap_ci(a, b, n_boot: int = 10_000, conf: float = 0.95, seed: int = 0):
    a, b = _arr(a), _arr(b)
    if a.shape != b.shape:
        raise ValueError("paired samples must have equal length")
    return bootstrap_ci(a - b, n_boot, conf, seed)


def wilcoxon_signed_rank(a, b) -> dict:
    a, b = _arr(a), _arr(b)
    if a.shape != b.shape:
        raise ValueError("paired samples must have equal length")
    d = a - b
    nz = int(np.count_nonzero(d))
    if nz == 0:
        return {"statistic": float("nan"), "p_value": 1.0, "n_nonzero": 0}
    res = sps.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue), "n_nonzero": nz}


def cohens_dz(a, b) -> float:
    d = _arr(a) - _arr(b)
    sd = d.std(ddof=1) if d.size > 1 else 0.0
    return float(d.mean() / sd) if sd > 0 else float("nan")


def optimality_gap(w_policy, w_opt) -> float:
    mp, mo = float(_arr(w_policy).mean()), float(_arr(w_opt).mean())
    return (mp - mo) / mo if mo > 0 else float("nan")


def moving_average(values, window: int) -> list[float]:
    """Trailing moving average; the first window-1 points average what is available."""
    if window < 1:
        raise ValueError("window must be >= 1")
    v = list(map(float, values))
    out, s = [], 0.0
    for i, x in enumerate(v):
        s += x
        if i >= window:
            s -= v[i - window]
        out.append(s / min(i + 1, window))
    return out


def converged(val_means, k: int = 3, rel_tol: float = 0.05) -> bool:
    """Convergence criterion: the last k validation means lie within a band of
    width rel_tol * (their mean)."""
    v = list(map(float, val_means))
    if len(v) < k:
        return False
    tail = v[-k:]
    return (max(tail) - min(tail)) <= rel_tol * abs(sum(tail) / k)
