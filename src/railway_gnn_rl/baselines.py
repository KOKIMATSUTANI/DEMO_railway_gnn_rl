"""Rule-based dispatchers and evaluation.

A dispatcher is a function `sim -> action` called only at decision points
(action = a candidate train id or HOLD).  The three rules are work-conserving:
they never choose HOLD.

* FCFS:            the candidate that reached its current node (or became ready)
                   earliest; ties -> lowest train id.
* fixed priority:  the lowest train id.
* longest waiting: the candidate with the largest accumulated waiting counter;
                   ties -> lowest train id.
* random:          uniformly random legal action, HOLD included (seeded).
"""

from __future__ import annotations

import random

from .agent import EpisodeRecord, record_of
from .sim import HOLD, RailwaySim, Scenario


def fcfs(sim: RailwaySim) -> int:
    return min(sim.decision.candidates, key=lambda i: (sim.trains[i].at_node_since, i))


def fixed_priority(sim: RailwaySim) -> int:
    return min(sim.decision.candidates)


def longest_waiting(sim: RailwaySim) -> int:
    return min(sim.decision.candidates, key=lambda i: (-sim.trains[i].waiting, i))


def make_random(seed: int):
    rng = random.Random(seed)

    def dispatch(sim: RailwaySim) -> int:
        return rng.choice(sim.legal_actions())
    return dispatch


def run_dispatcher(dispatcher, scenario: Scenario, delay_weight: float = 0.0) -> EpisodeRecord:
    sim = RailwaySim(scenario, delay_weight=delay_weight)
    sim.reset()
    decisions = holds = 0
    while not sim.done:
        a = dispatcher(sim)
        decisions += 1
        holds += a == HOLD
        sim.apply(a)
    return record_of(sim, decisions, holds)


def evaluate(dispatcher, scenarios, delay_weight: float = 0.0) -> list[EpisodeRecord]:
    return [run_dispatcher(dispatcher, sc, delay_weight) for sc in scenarios]
