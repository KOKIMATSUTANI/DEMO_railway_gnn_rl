"""Cycles 9-10a: rule-based dispatchers, metrics collection and evaluation."""

from conftest import make_sim
from railway_gnn_rl.baselines import evaluate, fcfs, fixed_priority, longest_waiting, make_random
from railway_gnn_rl.scenarios import generate_scenario
from railway_gnn_rl.sim import HOLD

A, B, C = 0, 1, 2


def t3_decision():
    # From the graph test: at t=3, T0 (at B since t=2, waited 1) and T2 (at C since t=0,
    # waited 3) compete for s1.
    sim = make_sim([((A, B, C), 0), ((C, B), 0), ((C, B), 0)])
    sim.apply(1)
    assert sim.decision.t == 3 and sim.decision.candidates == (0, 2)
    return sim


def test_fcfs_serves_the_train_that_reached_its_node_first():
    sim = t3_decision()
    assert sim.trains[0].at_node_since == 2 and sim.trains[2].at_node_since == 0
    assert fcfs(sim) == 2


def test_fcfs_tie_breaks_by_train_id_and_never_holds():
    sim = make_sim([((A, B), 0), ((A, B), 0)])
    assert fcfs(sim) == 0
    for seed in range(20):
        from railway_gnn_rl.sim import RailwaySim
        s = RailwaySim(generate_scenario(seed)); s.reset()
        while not s.done:
            a = fcfs(s)
            assert a != HOLD and a in s.decision.candidates
            s.apply(a)


def test_fixed_priority_picks_lowest_train_id():
    assert fixed_priority(t3_decision()) == 0


def test_longest_waiting_rule_picks_largest_waiting_counter():
    sim = t3_decision()
    assert (sim.trains[0].waiting, sim.trains[2].waiting) == (1, 3)
    assert longest_waiting(sim) == 2


def test_random_dispatcher_is_seeded_and_legal_including_hold():
    scen = [generate_scenario(s) for s in range(15)]
    r1 = evaluate(make_random(4), scen)
    r2 = evaluate(make_random(4), scen)
    assert [r.total_waiting for r in r1] == [r.total_waiting for r in r2]
    assert sum(r.holds for r in r1) > 0


def test_evaluate_returns_one_consistent_record_per_scenario():
    scen = [generate_scenario(s) for s in range(10)]
    recs = evaluate(fcfs, scen)
    assert len(recs) == 10
    for r in recs:
        assert r.termination == "success"
        assert r.episode_return == -r.total_waiting
        assert r.total_delay <= r.total_waiting      # delay = max(0, waiting - slack) per train
        assert r.steps == len(r.step_rewards) and r.holds == 0
