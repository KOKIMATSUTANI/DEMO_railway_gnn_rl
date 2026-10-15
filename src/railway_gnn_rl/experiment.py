"""Experiment pipeline: scenario sets, training runs, evaluation, summaries.

Scenario sets (all from `generate_scenario`, disjoint seed ranges):
  training   seed 1_000_000 + 100_000 * train_seed + episode, departures [0, 8]
  validation seeds 800_000 .. 800_019 (20), departures [0, 8]  - monitoring/convergence
  evaluation seeds 900_000 .. 900_049 (50), departures [0, 8]  - final comparison only
  heldout    seeds 700_000 .. 700_019 (20), departures [0, 4]  - shifted (denser) distribution
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .agent import ReinforceTrainer, policy_dispatcher
from .baselines import evaluate
from .models import make_policy
from .scenarios import DEP_HIGH, DEP_LOW, HELDOUT_DEP_HIGH, generate_scenario
from .sim import Infrastructure, Scenario, Section, TrainSpec
from .stats import bootstrap_ci, cohens_dz, optimality_gap, paired_bootstrap_ci, wilcoxon_signed_rank

SETS = {
    "validation": (800_000, 20, DEP_HIGH),
    "evaluation": (900_000, 50, DEP_HIGH),
    "heldout": (700_000, 20, HELDOUT_DEP_HIGH),
}

TRAIN_COLUMNS = ["kind", "seed", "episode", "episode_return", "total_waiting", "total_delay",
                 "decisions", "holds", "steps", "termination", "loss", "entropy",
                 "baseline_return", "advantage", "grad_norm"]

EVAL_COLUMNS = ["dispatcher", "seed", "set", "scenario", "total_waiting", "total_delay",
                "episode_return", "decisions", "holds", "steps", "termination"]


@dataclass
class TrainConfig:
    kind: str = "gnn"
    hidden: int = 32
    layers: int = 2
    lr: float = 1e-3
    entropy_coef: float = 0.01
    adv_scale: float = 10.0
    grad_clip: float = 1.0
    delay_weight: float = 0.0
    episodes: int = 500
    val_every: int = 50


# ------------------------------------------------------------------ scenarios
def scenario_set(name: str) -> list[Scenario]:
    base, n, hi = SETS[name]
    return [generate_scenario(base + k, DEP_LOW, hi) for k in range(n)]


def training_scenario(train_seed: int, episode: int) -> Scenario:
    return generate_scenario(1_000_000 + 100_000 * train_seed + episode)


def scenario_to_dict(sc: Scenario) -> dict:
    return asdict(sc)


def scenario_from_dict(d: dict) -> Scenario:
    inf = d["infra"]
    infra = Infrastructure(tuple(inf["node_names"]), tuple(Section(**s) for s in inf["sections"]),
                           None if inf["node_capacity"] is None else tuple(inf["node_capacity"]))
    trains = tuple(TrainSpec(t["id"], tuple(t["route"]), t["departure"], t["slack"])
                   for t in d["trains"])
    return Scenario(infra, trains, d["max_time"])


def save_scenarios(scenarios, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([scenario_to_dict(s) for s in scenarios], indent=1))


def load_scenarios(path) -> list[Scenario]:
    return [scenario_from_dict(d) for d in json.loads(Path(path).read_text())]


# ------------------------------------------------------------------------ CSV
def write_csv(rows, path, columns) -> None:
    if not rows:
        raise ValueError("no rows to write")
    for r in rows:
        missing = [c for c in columns if c not in r]
        if missing:
            raise KeyError(f"row lacks columns {missing}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _clean(v):
    return None if isinstance(v, float) and math.isnan(v) else v


# ------------------------------------------------------------------- training
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_run(cfg: TrainConfig, train_seed: int):
    """Train one policy. Returns (policy, per-episode rows, validation curve)."""
    seed_everything(train_seed)
    policy = make_policy(cfg.kind, cfg.hidden, cfg.layers)
    trainer = ReinforceTrainer(policy, cfg.lr, cfg.entropy_coef, cfg.adv_scale, cfg.grad_clip,
                               cfg.delay_weight, seed=train_seed)
    validation = scenario_set("validation")
    rows, val_curve = [], []
    for ep in range(cfg.episodes):
        m = trainer.train_episode(training_scenario(train_seed, ep))
        row = {"kind": cfg.kind, "seed": train_seed, "episode": ep}
        row.update({c: _clean(m[c]) for c in TRAIN_COLUMNS[3:]})
        rows.append(row)
        if (ep + 1) % cfg.val_every == 0:
            policy.eval()
            recs = evaluate(policy_dispatcher(policy), validation)
            val_curve.append({"kind": cfg.kind, "seed": train_seed, "episode": ep + 1,
                              "val_mean_waiting": float(np.mean([r.total_waiting for r in recs]))})
    policy.eval()
    return policy, rows, val_curve


# ----------------------------------------------------------------- evaluation
def evaluation_rows(name, records, set_name, seed=None) -> list[dict]:
    return [{"dispatcher": name, "seed": seed, "set": set_name, "scenario": k,
             "total_waiting": r.total_waiting, "total_delay": r.total_delay,
             "episode_return": r.episode_return, "decisions": r.decisions, "holds": r.holds,
             "steps": r.steps, "termination": r.termination}
            for k, r in enumerate(records)]


def _paired(a, b) -> dict:
    m, lo, hi = paired_bootstrap_ci(a, b)
    w = wilcoxon_signed_rank(a, b)
    return {"mean_diff": m, "ci_lo": lo, "ci_hi": hi, "wilcoxon_p": w["p_value"],
            "n_nonzero": w["n_nonzero"], "cohens_dz": cohens_dz(a, b)}


def summarize(per_scenario: dict, w_opt) -> dict:
    """per_scenario: dispatcher -> list of total waiting per scenario (same order).
    Returns mean, sd, 95% percentile-bootstrap CI over scenarios, optimality gap,
    and paired comparisons (difference = dispatcher - reference) vs FCFS and optimum."""
    out = {}
    for name, w in per_scenario.items():
        m, lo, hi = bootstrap_ci(w)
        entry = {"mean": m, "sd": float(np.std(w, ddof=1)) if len(w) > 1 else float("nan"),
                 "ci_lo": lo, "ci_hi": hi, "n": len(w),
                 "gap": optimality_gap(w, w_opt) if w_opt is not None else float("nan")}
        if "fcfs" in per_scenario and name != "fcfs":
            entry["vs_fcfs"] = _paired(w, per_scenario["fcfs"])
        if w_opt is not None:
            entry["vs_opt"] = _paired(w, w_opt)
        out[name] = entry
    return out
