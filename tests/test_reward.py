"""Cycle 4: reward calculation and episode termination."""

import random

from conftest import line_infra, make_sim, run_to_end
from railway_gnn_rl.sim import RailwaySim, Scenario, TrainSpec
from test_sim_occupancy import deadlock_infra

A, B, C = 0, 1, 2


def test_step_rewards_are_negative_waiting_increments_hand_calculated():
    # Both depart A at 0 for single-track s0 (2 min). Give s0 to T1:
    # t=0: T0 waits (W=1, r=-1); t=1: T0 waits (W=2, r=-1); t=2: T1 arrives, T0 enters (r=0);
    # t=3: r=0; t=4: T0 arrives -> success.
    sim = make_sim([((A, B), 0), ((A, B), 0)])
    assert sim.reward_before_first_decision == 0.0   # the decision is taken inside step t=0
    r = sim.apply(1)
    assert sim.step_rewards == [-1.0, -1.0, 0.0, 0.0, 0.0]
    assert r == -2.0
    assert sim.episode_return == -2.0 == -sim.total_waiting
    assert sim.termination == "success"


def test_rewards_between_decisions_sum_to_episode_return_without_double_counting():
    rng = random.Random(1)
    infra = line_infra()
    for _ in range(40):
        trains = [((A, B, C), rng.randint(0, 6)) for _ in range(3)] + \
                 [((C, B, A), rng.randint(0, 6)) for _ in range(3)]
        sim = make_sim(trains, infra)
        total = sim.reward_before_first_decision
        n_dec = 0
        while not sim.done:
            total += sim.apply(rng.choice(sim.decision.candidates))
            n_dec += 1
        assert len(sim.step_rewards) == len(sim.step_times)
        assert total == sum(sim.step_rewards) == sim.episode_return
        assert sim.episode_return == -sim.total_waiting          # gamma=1, no penalties
        ws = [s.total_waiting for s in sim.history]
        assert all(b >= a for a, b in zip(ws, ws[1:])) and ws[0] >= 0
        assert sim.step_rewards == [-(b - a) for a, b in zip([0] + ws, ws)]


def test_arrival_delay_is_positive_part_of_lateness_against_schedule():
    # Scheduled arrival = departure + running time + slack(2).
    # T0 gets s0 first: arrival 2 = sched 0+2+2=4 -> delay 0.
    # T1 waits 2 -> arrival 4, sched 4 -> delay 0. T2 waits 4 -> arrival 6, sched 4 -> delay 2.
    sim = make_sim([((A, B), 0), ((A, B), 0), ((A, B), 0)])
    run_to_end(sim)
    assert [sim.trains[i].arrival_time for i in range(3)] == [2, 4, 6]
    assert sim.scheduled_arrival == [4, 4, 4]
    assert [sim.trains[i].delay for i in range(3)] == [0, 0, 2]
    assert sim.total_delay == 2


def test_optional_terminal_delay_penalty_enters_the_return_at_arrival():
    specs = tuple(TrainSpec(i, (A, B), 0) for i in range(3))
    sim = RailwaySim(Scenario(line_infra(), specs), delay_weight=0.5)
    sim.reset()
    run_to_end(sim)
    assert sim.total_waiting == 6 and sim.total_delay == 2
    assert sim.episode_return == -6 - 0.5 * 2
    # the penalty is booked in the step of arrival (t=6)
    assert sim.step_rewards[6] == -0.5 * 2 - 0  # nobody waits at t=6


def test_truncation_at_max_time_penalises_unfinished_trains():
    # Route A->B->C needs 5 minutes; with max_time=3 the train cannot finish.
    sim = make_sim([((A, B, C), 0)], max_time=3)
    assert sim.done and sim.termination == "truncated"
    assert sim.t == 3
    assert sim.unfinished_penalty == 3                    # penalty per unfinished train = max_time
    assert sim.step_rewards[-1] == -3.0
    assert sim.episode_return == -sim.total_waiting - 3


def test_deadlock_terminates_with_documented_penalty():
    infra = deadlock_infra((None, 1, 1, None))
    specs = (TrainSpec(0, (0, 1, 2, 3), 0), TrainSpec(1, (3, 2, 1, 0), 0))
    sim = RailwaySim(Scenario(infra, specs))
    sim.reset()
    assert sim.termination == "deadlock" and sim.t == 1
    # both wait at t=1 (W=2) and both are unfinished -> 2 * max_time penalty
    assert sim.total_waiting == 2
    assert sim.episode_return == -2 - 2 * sim.max_time


def test_default_max_time_is_the_proven_completion_bound():
    sim = make_sim([((A, B, C), 4), ((C, B, A), 1), ((A, B), 0)])
    assert sim.max_time == 4 + (5 + 5 + 2)
