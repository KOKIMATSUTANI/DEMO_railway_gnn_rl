"""Cycle 13: figure generation from recorded metrics (file creation, not pixels)."""

import pytest

from railway_gnn_rl.plots import (plot_comparison, plot_paired_differences, plot_return_curve,
                                  plot_sensitivity, plot_step_trace, plot_waiting_curve)


def fake_rows():
    rows = []
    for kind in ("gnn", "mlp"):
        for seed in range(3):
            for ep in range(40):
                w = 20 - ep * 0.1 + seed
                rows.append({"kind": kind, "seed": seed, "episode": ep,
                             "episode_return": -w, "total_waiting": w})
    return rows


def is_png(p):
    return p.exists() and p.stat().st_size > 1000 and p.read_bytes()[:4] == b"\x89PNG"


def test_training_curves_are_written(tmp_path):
    a, b = tmp_path / "A.png", tmp_path / "B.png"
    plot_return_curve(fake_rows(), a, window=5)
    plot_waiting_curve(fake_rows(), b, refs={"FCFS (evaluation set)": 15.0,
                                             "exact optimum (evaluation set)": 13.0}, window=5)
    assert is_png(a) and is_png(b)


def test_step_trace_and_cumulative_return_are_separate_files(tmp_path):
    c1, c2 = tmp_path / "C1.png", tmp_path / "C2.png"
    plot_step_trace([0, -1, -2, 0, -1, 0, 0], c1, c2, title="evaluation scenario 0")
    assert is_png(c1) and is_png(c2)


def test_comparison_paired_and_sensitivity_figures(tmp_path):
    summary = {"fcfs": {"mean": 15.0, "ci_lo": 13.0, "ci_hi": 17.0},
               "optimum": {"mean": 13.0, "ci_lo": 11.5, "ci_hi": 14.6}}
    plot_comparison(summary, tmp_path / "D.png")
    plot_paired_differences([-2, 0, 0, 1, -1, 0, -3], tmp_path / "E.png", label="GNN - FCFS")
    table = [{"setting": "base", "val_mean": 14.0, "val_lo": 13.0, "val_hi": 15.0},
             {"setting": "layers=1", "val_mean": 14.5, "val_lo": 13.2, "val_hi": 15.8}]
    plot_sensitivity(table, tmp_path / "F.png")
    for n in "DEF":
        assert is_png(tmp_path / f"{n}.png")


def test_empty_or_invalid_inputs_raise(tmp_path):
    with pytest.raises(ValueError):
        plot_return_curve([], tmp_path / "x.png")
    with pytest.raises(ValueError):
        plot_step_trace([], tmp_path / "a.png", tmp_path / "b.png")
    with pytest.raises(ValueError):
        plot_comparison({}, tmp_path / "d.png")
    with pytest.raises(ValueError):
        plot_paired_differences([], tmp_path / "e.png")
    with pytest.raises(KeyError):
        plot_return_curve([{"kind": "gnn", "seed": 0}], tmp_path / "y.png")
