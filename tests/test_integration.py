"""Cycle 14: end-to-end integration tests."""

import csv
import json

import pytest
import torch

from railway_gnn_rl.agent import ReinforceTrainer, action_mask, select_action
from railway_gnn_rl.baselines import evaluate
from railway_gnn_rl.agent import policy_dispatcher
from railway_gnn_rl.cli import main
from railway_gnn_rl.graph_obs import build_observation
from railway_gnn_rl.models import make_policy
from railway_gnn_rl.scenarios import generate_scenario
from railway_gnn_rl.sim import RailwaySim

pytestmark = pytest.mark.integration


def test_environment_policy_update_episode_and_evaluation_end_to_end():
    torch.manual_seed(0)
    policy = make_policy("gnn", 16, 2)
    sc = next(generate_scenario(s) for s in range(50)
              if (lambda sim: (sim.reset(), sim.decision is not None)[1])(RailwaySim(generate_scenario(s))))
    sim = RailwaySim(sc)
    sim.reset()
    obs = build_observation(sim)
    assert obs.x.shape == (7, obs.x.shape[1]) and obs.edge_index.shape == (2, 16)
    total = sim.reward_before_first_decision
    while not sim.done:                                     # several legal actions
        a, logp, _ = select_action(policy, sim, "sample", torch.Generator().manual_seed(1))
        assert a in sim.legal_actions() and torch.isfinite(logp)
        assert bool(action_mask(sim).any())
        total += sim.apply(a)
    assert total == sim.episode_return == -sim.total_waiting
    before = [p.detach().clone() for p in policy.parameters()]
    m = ReinforceTrainer(policy, lr=1e-2, seed=0).train_episode(sc)   # >= 1 valid update
    assert m["updated"]
    assert any(not torch.equal(b, p) for b, p in zip(before, policy.parameters()))
    recs = evaluate(policy_dispatcher(policy), [generate_scenario(s) for s in range(5)])
    assert len(recs) == 5 and all(r.termination == "success" for r in recs)


def test_cli_pipeline_smoke_run_writes_all_outputs(tmp_path):
    out = tmp_path / "out"
    rc = main(["all", "--out", str(out), "--episodes", "6", "--val-every", "3",
               "--seeds", "0", "1", "--sens-seeds", "0", "--eval-limit", "6", "--workers", "1"])
    assert rc == 0
    for f in ["config.json", "train_metrics.csv", "validation_curve.csv", "optimum.csv",
              "eval_per_scenario.csv", "summary.json", "summary.md", "sensitivity.csv",
              "step_trace.csv", "scenarios/evaluation.json",
              "checkpoints/gnn_seed0.pt", "checkpoints/mlp_seed1.pt"]:
        assert (out / f).exists(), f
    for fig in "ABCDEF":
        assert list((out / "figures").glob(f"{fig}_*.png")), fig
    rows = list(csv.DictReader(open(out / "train_metrics.csv")))
    assert len(rows) == 2 * 2 * 6                                      # kinds x seeds x episodes
    ev = list(csv.DictReader(open(out / "eval_per_scenario.csv")))
    names = {r["dispatcher"] for r in ev}
    assert {"fcfs", "fixed_priority", "longest_waiting", "random", "optimum", "gnn", "mlp"} <= names
    per = {}
    for r in ev:
        if r["set"] == "evaluation":
            per.setdefault((r["dispatcher"], r["seed"]), []).append(int(r["scenario"]))
    assert all(sorted(v) == list(range(6)) for v in per.values())      # identical scenarios
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["reproducibility"]["max_abs_diff"] <= 1e-9
    assert "torch" in cfg["versions"] and "train" in cfg
    assert all(r["termination"] == "success" for r in ev)
