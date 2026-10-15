"""Cycle 11: confidence intervals, paired statistics and helper functions."""

import math

import pytest

from railway_gnn_rl.stats import (bootstrap_ci, cohens_dz, converged, mean_ci_t, moving_average,
                                  optimality_gap, paired_bootstrap_ci, wilcoxon_signed_rank)


def test_t_interval_hand_calculated():
    # mean 3, sd = sqrt(2.5) = 1.5811, se = 0.70711, t_{0.975,4} = 2.776445 -> half width 1.96324
    m, lo, hi = mean_ci_t([1, 2, 3, 4, 5])
    assert m == 3
    assert lo == pytest.approx(3 - 1.963243, abs=1e-5)
    assert hi == pytest.approx(3 + 1.963243, abs=1e-5)


def test_t_interval_single_value_and_empty_input():
    m, lo, hi = mean_ci_t([4.0])
    assert m == 4.0 and math.isnan(lo) and math.isnan(hi)
    with pytest.raises(ValueError):
        mean_ci_t([])


def test_bootstrap_ci_constant_data_and_reproducibility():
    assert bootstrap_ci([2, 2, 2, 2]) == (2.0, 2.0, 2.0)
    data = [1, 5, 2, 8, 3, 3, 9, 4]
    a, b = bootstrap_ci(data, seed=1), bootstrap_ci(data, seed=1)
    assert a == b
    m, lo, hi = a
    assert m == pytest.approx(4.375) and lo < m < hi


def test_paired_bootstrap_uses_per_scenario_differences():
    a = [10, 12, 9, 11, 10]
    b = [9, 11, 8, 10, 9]           # every difference is exactly +1
    assert paired_bootstrap_ci(a, b) == (1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1, 2], [1])


def test_wilcoxon_known_exact_value_and_all_ties():
    a = list(range(11, 21))
    b = list(range(1, 11))          # 10 positive differences -> exact two-sided p = 2/2^10
    res = wilcoxon_signed_rank(a, b)
    assert res["p_value"] == pytest.approx(2 / 1024)
    assert res["n_nonzero"] == 10
    same = wilcoxon_signed_rank([3, 4, 5], [3, 4, 5])
    assert same["p_value"] == 1.0 and same["n_nonzero"] == 0


def test_cohens_dz_hand_calculated():
    # differences 1, 2, 3 -> mean 2, sd 1 -> d_z = 2
    assert cohens_dz([2, 4, 6], [1, 2, 3]) == pytest.approx(2.0)
    assert math.isnan(cohens_dz([1, 2], [1, 2]))       # zero variance of differences


def test_optimality_gap_is_ratio_of_means():
    assert optimality_gap([12, 8], [10, 6]) == pytest.approx((10 - 8) / 8)
    assert math.isnan(optimality_gap([1], [0]))


def test_moving_average_trailing_window():
    assert moving_average([1, 2, 3, 4, 5], 2) == [1.0, 1.5, 2.5, 3.5, 4.5]
    assert moving_average([], 3) == []
    with pytest.raises(ValueError):
        moving_average([1], 0)


def test_convergence_criterion():
    assert converged([20.0, 15.0, 14.0, 14.2, 13.9], k=3, rel_tol=0.05)   # range 0.3 <= 0.7
    assert not converged([20.0, 14.0, 12.0, 10.0], k=3, rel_tol=0.05)
    assert not converged([10.0, 10.0], k=3)                              # too few points
