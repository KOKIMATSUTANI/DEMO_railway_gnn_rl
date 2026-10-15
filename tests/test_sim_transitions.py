"""Cycle 2: railway environment state transitions."""

import pytest

from conftest import make_sim, run_to_end

A, B, C = 0, 1, 2


def test_per_step_snapshots_follow_the_hand_calculated_state_sequence():
    # T0 departs A at t=1: s0 (2 min) then s1 (3 min) -> arrives C at 6.
    sim = make_sim([((A, B, C), 1)])
    run_to_end(sim)
    seq = [(snap.t, snap.trains[0].status, snap.trains[0].position, snap.trains[0].remaining)
           for snap in sim.history]
    assert seq == [
        (0, "pending", None, 0),
        (1, "on_section", ("section", 0), 2),
        (2, "on_section", ("section", 0), 1),
        (3, "on_section", ("section", 1), 3),   # arrived at B and left again in the same step
        (4, "on_section", ("section", 1), 2),
        (5, "on_section", ("section", 1), 1),
        (6, "finished", None, 0),
    ]


def test_waiting_train_position_is_its_node_and_route_index_advances():
    # T0 occupies s0 A->B from t=0..2; T1 waits at B for s0 during t=1.
    sim = make_sim([((A, B, C), 0), ((B, A), 1)])
    run_to_end(sim)
    snap1 = sim.history[1]
    assert snap1.trains[1].status == "at_node"
    assert snap1.trains[1].position == ("node", B)
    snap2 = sim.history[2]
    assert snap2.trains[0].position == ("section", 1)
    assert snap2.trains[0].route_idx == 1
    assert sim.trains[0].route_idx == 2 and sim.trains[1].route_idx == 1


def test_apply_rejects_illegal_train_and_leaves_state_unchanged():
    sim = make_sim([((A, B), 0), ((A, B), 0), ((B, C), 5)])
    before = (sim.t, sim.decision, [vars(s).copy() for s in sim.trains])
    with pytest.raises(ValueError):
        sim.apply(2)            # train 2 is not a candidate
    assert (sim.t, sim.decision, [vars(s).copy() for s in sim.trains]) == before


def test_apply_without_pending_decision_raises():
    sim = make_sim([((A, B), 0)])
    assert sim.done
    with pytest.raises(RuntimeError):
        sim.apply(0)


def test_decision_point_state_reflects_trains_in_motion():
    # T0 enters s1 (B->C) at 0. T1, T2 both depart A at 1 and compete for s0.
    sim = make_sim([((B, C), 0), ((A, B), 1), ((A, B), 1)])
    assert sim.decision.t == 1
    assert sim.trains[0].status == "on_section" and sim.trains[0].remaining == 2
    assert sim.position(1) == ("node", A) and sim.position(2) == ("node", A)
