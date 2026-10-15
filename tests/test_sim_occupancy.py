"""Cycle 3: shared-section occupancy constraints and deadlock detection."""

import random

from conftest import line_infra, make_sim, run_to_end
from railway_gnn_rl.sim import Infrastructure, Section

A, B, C = 0, 1, 2


def test_double_track_lets_opposite_trains_pass_without_decision():
    infra = line_infra(kind0="double", cap0=1)
    sim = make_sim([((A, B), 0), ((B, A), 0)], infra)
    assert run_to_end(sim) == []
    assert sim.trains[0].arrival_time == 2 and sim.trains[1].arrival_time == 2
    assert sim.total_waiting == 0


def test_double_track_capacity_is_per_direction():
    infra = line_infra(kind0="double", cap0=1)
    sim = make_sim([((A, B), 0), ((A, B), 0), ((B, A), 0)], infra)
    # T2 (B->A) enters freely; T0 and T1 (A->B) compete for the single A->B track.
    assert sim.decision.candidates == (0, 1)
    assert sim.trains[2].status == "on_section"


def test_single_track_capacity_two_allows_following_but_not_opposing_moves():
    infra = line_infra(kind0="single", cap0=2)
    sim = make_sim([((A, B), 0), ((A, B), 0), ((B, A), 1)], infra)
    seen = run_to_end(sim)
    assert seen == []                                   # both A->B trains fit together
    assert sim.trains[0].arrival_time == 2 and sim.trains[1].arrival_time == 2
    # T2 is ready at 1 but s0 is occupied in the opposite direction until t=2.
    assert sim.trains[2].waiting == 1 and sim.trains[2].arrival_time == 4


def test_single_track_opposing_requests_are_a_conflict():
    sim = make_sim([((A, B), 0), ((B, A), 0)])
    assert sim.decision.section == 0 and sim.decision.candidates == (0, 1)
    sim.apply(1)
    run_to_end(sim)
    assert sim.trains[1].arrival_time == 2 and sim.trains[0].arrival_time == 4
    assert sim.total_waiting == 2


def test_capacity_is_never_exceeded_under_random_dispatching():
    rng = random.Random(0)
    for kind, cap in [("single", 1), ("single", 2), ("double", 1), ("double", 2)]:
        infra = line_infra(kind0=kind, cap0=cap, kind1=kind, cap1=cap)
        for _ in range(30):
            trains = [((A, B, C), rng.randint(0, 4)) for _ in range(3)] + \
                     [((C, B, A), rng.randint(0, 4)) for _ in range(3)]
            sim = make_sim(trains, infra)
            run_to_end(sim, lambda d: rng.choice(d.candidates))
            assert sim.termination == "success"
            for snap in sim.history:
                for s, occ in zip(infra.sections, snap.occupants):
                    dirs = [snap.trains[j].direction for j in occ]
                    if kind == "single":
                        assert len(occ) <= cap and len(set(dirs)) <= 1
                    else:
                        assert dirs.count(1) <= cap and dirs.count(-1) <= cap


def deadlock_infra(node_capacity):
    # P --s0(1)-- X --s1(1)-- Y --s2(1)-- Q ; stations X and Y hold one train each.
    return Infrastructure(
        node_names=("P", "X", "Y", "Q"),
        sections=(Section(0, 0, 1, 1, 1, "single"), Section(1, 1, 2, 1, 1, "single"),
                  Section(2, 2, 3, 1, 1, "single")),
        node_capacity=node_capacity,
    )


def test_node_capacity_blocks_entry_towards_a_full_station():
    # s0 is double track with 2 tracks per direction, so the *section* admits both
    # trains; station X (capacity 1) does not: entry reserves a slot at the target node.
    infra = Infrastructure(
        node_names=("P", "X", "Q"),
        sections=(Section(0, 0, 1, 1, 2, "double"), Section(1, 1, 2, 1, 1, "single")),
        node_capacity=(None, 1, None),
    )
    sim = make_sim([((0, 1, 2), 0), ((0, 1, 2), 0)], infra)
    assert sim.decision.section == 0 and sim.decision.candidates == (0, 1)
    sim.apply(0)
    run_to_end(sim)
    # T0: enter 0, at X at 1, enters s1 at 1 (X freed), arrives Q at 2.
    # T1: may enter s0 only at t=1 (X freed in the same step) -> X at 2, Q at 3.
    assert sim.trains[0].arrival_time == 2
    assert sim.trains[1].arrival_time == 3 and sim.trains[1].waiting == 1


def test_hand_built_mutual_blocking_is_detected_as_deadlock():
    # t=0: T0 enters s0 (P->X), T1 enters s2 (Q->Y). t=1: both arrive.
    # T0 needs Y (full, holds T1); T1 needs X (full, holds T0): no progress ever.
    infra = deadlock_infra((None, 1, 1, None))
    sim = make_sim([((0, 1, 2, 3), 0), ((3, 2, 1, 0), 0)], infra)
    assert sim.done
    assert sim.termination == "deadlock"
    assert sim.t == 1
    assert sim.is_deadlocked()


def test_deadlock_is_reported_only_once_no_train_can_progress_anymore():
    # Same blocking pair as above plus T2 (P->X), which departs only at t=5.
    # While T2 is still pending the state can change, so it is not yet a deadlock;
    # at t=5 T2 is ready but blocked too (X full) -> fixed point -> deadlock.
    infra = deadlock_infra((None, 1, 1, None))
    sim = make_sim([((0, 1, 2, 3), 0), ((3, 2, 1, 0), 0), ((0, 1), 5)], infra)
    # T2 becomes ready at 5, so before t=5 the state is not a fixed point.
    assert sim.done and sim.termination == "deadlock"
    assert sim.t == 5


def test_no_deadlock_flag_for_unlimited_node_capacity():
    infra = deadlock_infra(None)
    sim = make_sim([((0, 1, 2, 3), 0), ((3, 2, 1, 0), 0)], infra)
    run_to_end(sim)
    assert sim.termination == "success"
    assert not sim.is_deadlocked()
