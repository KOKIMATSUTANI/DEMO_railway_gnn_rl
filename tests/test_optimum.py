"""Cycle 10b: exact-optimum solver, cross-checked against the simulator."""

import copy
import random

from railway_gnn_rl.baselines import evaluate, fcfs, fixed_priority, longest_waiting, make_random
from railway_gnn_rl.optimum import replay, solve
from railway_gnn_rl.scenarios import generate_scenario
from railway_gnn_rl.sim import HOLD, RailwaySim, Scenario, TrainSpec
from conftest import line_infra
from test_sim_hold import TRAINS, crossing_infra


def brute_force(sim):
    """Plain exhaustive search without memoisation (reference implementation)."""
    if sim.done:
        return sim.total_waiting
    best = None
    for a in sim.legal_actions():
        s2 = copy.deepcopy(sim)
        s2.apply(a)
        v = brute_force(s2)
        best = v if best is None else min(best, v)
    return best


def test_hand_solvable_crossing_optimum_is_six_with_hold_first():
    scen = Scenario(crossing_infra(), tuple(TrainSpec(i, r, d) for i, (r, d) in enumerate(TRAINS)))
    res = solve(scen)
    assert res.status == "optimal"
    assert res.total_waiting == 6
    assert res.actions[0] == HOLD
    assert replay(scen, res.actions).total_waiting == 6


def test_memoised_solver_matches_brute_force_on_tiny_scenarios():
    rng = random.Random(0)
    for _ in range(25):
        trains = tuple(TrainSpec(i, r, rng.randint(0, 3)) for i, r in
                       enumerate([(0, 1, 2), (2, 1, 0), (0, 1, 2), (2, 1)]))
        scen = Scenario(line_infra(), trains)
        sim = RailwaySim(scen)
        sim.reset()
        res = solve(scen)
        assert res.total_waiting == brute_force(sim)


def test_optimal_action_sequence_replays_to_reported_value_and_beats_all_rules():
    scen = [generate_scenario(s) for s in range(12)]
    rules = {name: evaluate(d, scen) for name, d in
             [("fcfs", fcfs), ("prio", fixed_priority), ("wait", longest_waiting),
              ("rand", make_random(0))]}
    for k, sc in enumerate(scen):
        res = solve(sc)
        assert res.status == "optimal"
        rep = replay(sc, res.actions)
        assert rep.total_waiting == res.total_waiting and rep.termination == "success"
        for recs in rules.values():
            assert res.total_waiting <= recs[k].total_waiting


def test_node_limit_reports_not_computed_instead_of_guessing():
    res = solve(generate_scenario(37), node_limit=3)
    assert res.status == "not_computed" and res.total_waiting is None
