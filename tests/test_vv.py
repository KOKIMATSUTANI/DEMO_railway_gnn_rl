"""Section 10: simulator verification & validation on the demo network.

The three hand-calculated scenarios required by the spec live in other files:
  * no conflict:                test_sim_clock.py::test_clock_advances_exactly_one_minute_per_step_and_train_arrives_on_time
  * one conflict (same dir.):   test_sim_clock.py::test_competing_requests_expose_a_decision_with_ascending_candidates
  * single-track opposing:      test_sim_occupancy.py::test_single_track_opposing_requests_are_a_conflict
                                and test_sim_hold.py (crossing with HOLD, W = 6 vs 8)
The optimum cross-check (replay of the solver's action sequence) is in test_optimum.py.
"""

import random

from railway_gnn_rl.scenarios import HELDOUT_DEP_HIGH, generate_scenario
from railway_gnn_rl.sim import RailwaySim


def random_runs(n=80, dep_high=None, seed=0):
    rng = random.Random(seed)
    for s in range(n):
        sc = generate_scenario(s) if dep_high is None else generate_scenario(s, 0, dep_high)
        sim = RailwaySim(sc)
        sim.reset()
        actions = []
        while not sim.done:
            a = rng.choice(sim.legal_actions())
            actions.append(a)
            sim.apply(a)
        yield sc, sim, actions


def test_invariants_capacity_position_conservation_and_final_occupancy():
    for dep_high in (None, HELDOUT_DEP_HIGH):
        for sc, sim, _ in random_runs(dep_high=dep_high):
            secs = sc.infra.sections
            for snap in sim.history:
                assert len(snap.trains) == len(sc.trains)               # trains conserved
                on = {}
                for s, occ in zip(secs, snap.occupants):
                    dirs = [snap.trains[j].direction for j in occ]
                    if s.kind == "single":
                        assert len(occ) <= s.capacity and len(set(dirs)) <= 1
                    else:
                        assert dirs.count(1) <= s.capacity and dirs.count(-1) <= s.capacity
                    for j in occ:
                        assert j not in on                              # at most one section
                        on[j] = s.id
                for j, ts in enumerate(snap.trains):
                    # a train is on a section iff it is listed as occupant of exactly that section
                    assert (ts.status == "on_section") == (j in on)
                    if j in on:
                        assert ts.position == ("section", on[j])
            assert sim.termination == "success"
            assert all(len(o) == 0 for o in sim.occupants)              # empty at the end


def test_determinism_same_scenario_and_actions_give_identical_trajectories():
    for sc, sim, actions in random_runs(n=30, seed=7):
        again = RailwaySim(sc)
        again.reset()
        for a in actions:
            again.apply(a)
        assert again.history == sim.history
        assert again.events == sim.events
        assert again.step_rewards == sim.step_rewards


def test_time_step_consistency_no_sub_step_events():
    # With integer travel times every arrival happens exactly travel_time minutes after
    # entry, i.e. on the 1-minute grid: dt = 1 introduces no discretisation error.
    for sc, sim, _ in random_runs(n=40, seed=3):
        entered = {}
        for e in sim.events:
            assert isinstance(e.t, int)
            if e.kind == "enter":
                entered[(e.train, e.section)] = e.t
            elif e.kind == "release":
                t_in = entered.pop((e.train, e.section))
                assert e.t - t_in == sc.infra.sections[e.section].travel_time
        assert sim.step_times == list(range(sim.t + 1))


def test_waiting_monotone_non_negative_and_bounded():
    for sc, sim, _ in random_runs(n=60, seed=5):
        ws = [s.total_waiting for s in sim.history]
        assert ws[0] >= 0 and all(b >= a for a, b in zip(ws, ws[1:]))
        assert 0 <= sim.total_waiting <= len(sc.trains) * sim.max_time   # documented bound
        assert sim.t <= sim.max_time


def test_no_deadlock_or_truncation_on_demo_network():
    terms = [sim.termination for dep in (None, HELDOUT_DEP_HIGH)
             for _, sim, _ in random_runs(n=100, dep_high=dep, seed=11)]
    assert set(terms) == {"success"}
