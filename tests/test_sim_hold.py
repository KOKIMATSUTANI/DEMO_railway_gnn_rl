"""Cycle 3b: the HOLD action (crossing decisions on constrained sections)."""

import random

import pytest

from conftest import make_sim, run_to_end
from railway_gnn_rl.scenarios import generate_scenario
from railway_gnn_rl.sim import HOLD, Infrastructure, RailwaySim, Section

A, B, C = 0, 1, 2


def crossing_infra(s1_travel=5):
    # A --s0 (double, 2 per dir, 1 min)-- B --s1 (single, 2 following, s1_travel min)-- C
    return Infrastructure(
        node_names=("A", "B", "C"),
        sections=(Section(0, A, B, 1, 2, "double"), Section(1, B, C, s1_travel, 2, "single")),
    )


TRAINS = [((C, B), 0), ((A, B, C), 0), ((A, B, C), 0)]


def test_hold_is_offered_when_an_approaching_train_would_be_blocked():
    sim = make_sim(TRAINS, crossing_infra())
    # t=0: T1, T2 entered s0 (auto) and reach B at t=1 < 5 = travel time of s1.
    d = sim.decision
    assert (d.t, d.section, d.candidates, d.can_hold) == (0, 1, (0,), True)


def test_granting_immediately_hand_calculation():
    sim = make_sim(TRAINS, crossing_infra())
    sim.apply(0)       # T0 occupies s1 during t=0..4, arrives B at 5
    run_to_end(sim)
    # T1, T2 reach B at 1, are blocked t=1..4 (4 min each), enter together at 5, arrive C at 10.
    assert [sim.trains[i].arrival_time for i in range(3)] == [5, 10, 10]
    assert sim.total_waiting == 8


def test_holding_hand_calculation_reduces_total_waiting():
    sim = make_sim(TRAINS, crossing_infra())
    sim.apply(HOLD)    # T0 waits at t=0
    d = sim.decision   # t=1: all three request s1, no approaching trains left
    assert (d.t, d.candidates, d.can_hold) == (1, (0, 1, 2), False)
    sim.apply(1)       # T1 enters; T2 (same direction) still fits, T0 is now blocked
    d = sim.decision
    assert (d.t, d.candidates, d.can_hold) == (1, (2,), True)
    sim.apply(2)
    run_to_end(sim)
    # T1, T2 on s1 during 1..5, arrive C at 6; T0 waits 0..5 (6 min), enters 6, arrives B at 11.
    assert [sim.trains[i].arrival_time for i in range(3)] == [11, 6, 6]
    assert sim.total_waiting == 6
    assert sim.episode_return == -6.0


def test_no_hold_if_approaching_train_arrives_after_section_would_be_clear():
    # s1 takes only 1 minute: T0 clears s1 before T1/T2 (arriving at t=1) need it.
    sim = make_sim(TRAINS, crossing_infra(s1_travel=1))
    assert sim.done and sim.total_waiting == 0


def test_no_hold_without_competitors_and_hold_rejected_when_not_offered():
    sim = make_sim([((A, B), 0), ((A, B), 0)])
    d = sim.decision
    assert d.can_hold is False
    with pytest.raises(ValueError):
        sim.apply(HOLD)


def test_random_policies_with_holds_always_finish_within_the_bound():
    rng = random.Random(3)
    holds = 0
    for seed in range(60):
        sim = RailwaySim(generate_scenario(seed))
        sim.reset()
        while not sim.done:
            acts = list(sim.decision.candidates) + ([HOLD] if sim.decision.can_hold else [])
            a = rng.choice(acts)
            holds += a == HOLD
            sim.apply(a)
        assert sim.termination == "success"
        assert sim.t <= sim.max_time
    assert holds > 0
