"""Cycle 1: simulation clock, time step (1 min) and deterministic event ordering."""

from conftest import make_sim, run_to_end

A, B, C = 0, 1, 2


def test_reset_starts_at_time_zero_and_runs_until_the_first_decision():
    # Two trains depart A at t=3 and compete for s0: the first decision is at t=3.
    sim = make_sim([((A, B), 3), ((A, B), 3)])
    assert sim.step_times == [0, 1, 2]       # completed steps before the decision
    assert sim.t == 3
    assert sim.decision is not None and sim.decision.t == 3
    assert sim.total_waiting == 0            # trains before departure never wait
    assert [e for e in sim.events if e.t < 3] == []


def test_clock_advances_exactly_one_minute_per_step_and_train_arrives_on_time():
    # Departs at 1, s0 takes 2, s1 takes 3 -> arrival 1 + 2 + 3 = 6, no waiting.
    sim = make_sim([((A, B, C), 1)])
    seen = run_to_end(sim)
    assert seen == []                      # one train never needs a decision
    assert sim.step_times == list(range(0, 7))
    assert sim.trains[0].arrival_time == 6
    assert sim.total_waiting == 0
    assert sim.termination == "success"


def test_train_becomes_ready_exactly_at_departure_time():
    sim = make_sim([((A, B, C), 3)])
    # No decisions possible with a single train: peek at the event log instead.
    run_to_end(sim)
    ready = [e for e in sim.events if e.kind == "ready"]
    assert [(e.t, e.train) for e in ready] == [(3, 0)]
    enter = [e for e in sim.events if e.kind == "enter"]
    assert [(e.t, e.train, e.section) for e in enter] == [(3, 0, 0), (5, 0, 1)]


def test_section_released_in_phase_2_is_available_in_phase_8_of_same_step():
    # Hand calculation (single track s0 between A and B, 2 min):
    #   t=0: T0 departs A, enters s0 (A->B), will arrive at B at t=2.
    #   t=1: T1 departs B wanting s0 (B->A); s0 occupied by T0 -> T1 waits 1 train-minute.
    #   t=2: phase 2 T0 releases s0 and arrives at B; phase 8 T1 enters s0, T0 enters s1.
    #   T1 arrives at A at 2 + 2 = 4; T0 arrives at C at 2 + 3 = 5.  Total waiting W = 1.
    sim = make_sim([((A, B, C), 0), ((B, A), 1)])
    seen = run_to_end(sim)
    assert seen == []
    assert sim.trains[0].arrival_time == 5
    assert sim.trains[1].arrival_time == 4
    assert sim.trains[1].waiting == 1
    assert sim.trains[0].waiting == 0
    assert sim.total_waiting == 1
    release_t2 = [e for e in sim.events if e.t == 2 and e.kind in ("release", "enter")]
    assert [(e.phase, e.kind, e.train) for e in release_t2] == [
        (2, "release", 0), (8, "enter", 0), (8, "enter", 1)]


def test_events_are_ordered_by_phase_then_ascending_train_id_within_each_step():
    sim = make_sim([((C, B), 0), ((A, B), 0), ((B, C), 4), ((B, A), 4)])
    run_to_end(sim)
    assert len(sim.events) > 0
    by_t = {}
    for e in sim.events:
        by_t.setdefault(e.t, []).append((e.phase, e.train))
    for t, keys in by_t.items():
        assert keys == sorted(keys), f"events at t={t} not ordered: {keys}"


def test_competing_requests_expose_a_decision_with_ascending_candidates():
    # Both trains depart A at t=0 and want single-track s0: a genuine conflict.
    sim = make_sim([((A, B), 0), ((A, B), 0)])
    assert not sim.done
    d = sim.decision
    assert d is not None
    assert d.t == 0 and d.section == 0 and d.candidates == (0, 1)
    sim.apply(1)          # give s0 to T1: T1 arrives at 2, T0 enters at 2 and arrives at 4
    run_to_end(sim)
    assert sim.trains[1].arrival_time == 2
    assert sim.trains[0].arrival_time == 4
    assert sim.trains[0].waiting == 2
    assert sim.total_waiting == 2
