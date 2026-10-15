"""Cycles 9, 11, 12: experiment pipeline (metrics, CSV, scenario sets, summaries)."""

import csv

import pytest
import torch

from railway_gnn_rl.baselines import evaluate, fcfs
from railway_gnn_rl.experiment import (TRAIN_COLUMNS, TrainConfig, evaluation_rows, load_scenarios,
                                       save_scenarios, scenario_set, summarize, train_run,
                                       training_scenario, write_csv)


def test_scenario_sets_have_documented_sizes_and_disjoint_seeds():
    ev, val, ho = scenario_set("evaluation"), scenario_set("validation"), scenario_set("heldout")
    assert (len(ev), len(val), len(ho)) == (50, 20, 20)
    assert all(tr.departure <= 4 for sc in ho for tr in sc.trains)
    assert scenario_set("evaluation") == ev                          # fixed seed -> identical
    train = {training_scenario(s, k) for s in range(5) for k in range(500)}
    assert not (train & set(ev)) and not (train & set(val)) and not (set(val) & set(ev))


def test_scenarios_round_trip_through_disk(tmp_path):
    ev = scenario_set("evaluation")
    p = tmp_path / "evaluation.json"
    save_scenarios(ev, p)
    assert load_scenarios(p) == ev


def test_write_csv_columns_rows_and_invalid_input(tmp_path):
    p = tmp_path / "x.csv"
    write_csv([{"a": 1, "b": 2}, {"a": 3, "b": 4}], p, ["a", "b"])
    rows = list(csv.DictReader(open(p)))
    assert [r["a"] for r in rows] == ["1", "3"] and list(rows[0]) == ["a", "b"]
    with pytest.raises(ValueError):
        write_csv([], p, ["a"])
    with pytest.raises(KeyError):
        write_csv([{"a": 1}], p, ["a", "b"])


def test_train_run_records_one_row_per_episode_with_consistent_metrics():
    cfg = TrainConfig(episodes=12, val_every=6, hidden=16)
    policy, rows, val_curve = train_run(cfg, train_seed=0)
    assert len(rows) == 12 and list(rows[0]) == TRAIN_COLUMNS
    assert [r["episode"] for r in rows] == list(range(12))
    for r in rows:
        assert r["episode_return"] == -r["total_waiting"]            # gamma=1, no penalties
        assert r["termination"] == "success"
    assert [v["episode"] for v in val_curve] == [6, 12]
    assert all(v["val_mean_waiting"] >= 0 for v in val_curve)
    assert isinstance(policy, torch.nn.Module)


def test_train_run_is_reproducible_for_a_seed():
    cfg = TrainConfig(episodes=6, val_every=6, hidden=16)
    _, r1, _ = train_run(cfg, 3)
    _, r2, _ = train_run(cfg, 3)
    assert r1 == r2


def test_evaluation_rows_and_summary_with_paired_statistics():
    scen = scenario_set("evaluation")[:8]
    recs = evaluate(fcfs, scen)
    rows = evaluation_rows("fcfs", recs, "evaluation")
    assert len(rows) == 8 and rows[0]["dispatcher"] == "fcfs" and rows[3]["scenario"] == 3
    w_fcfs = [r.total_waiting for r in recs]
    w_better = [max(0, w - 1) for w in w_fcfs]
    s = summarize({"fcfs": w_fcfs, "better": w_better}, w_opt=w_better)
    assert s["fcfs"]["mean"] == pytest.approx(sum(w_fcfs) / 8)
    assert s["better"]["gap"] == pytest.approx(0.0)
    assert s["better"]["vs_fcfs"]["mean_diff"] < 0
    assert s["fcfs"]["ci_lo"] <= s["fcfs"]["mean"] <= s["fcfs"]["ci_hi"]
