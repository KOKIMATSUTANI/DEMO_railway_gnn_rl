"""Exact optimum benchmark by memoised depth-first search over decisions.

The simulator is deterministic, so a scenario defines a finite decision tree
(every episode ends by t <= max_time).  The solver explores all legal actions
(candidate trains and HOLD) at every decision point, using the *same*
simulator, and memoises the best achievable future return per Markov state
(`RailwaySim.state_key`):

    V(s) = max_a [ r(s, a) + V(s') ],   V(terminal) = 0

where r(s, a) is the reward returned by `sim.apply(a)` (sum of step rewards
until the next decision).  With the default reward this maximises -W, i.e.
minimises total waiting time.  Complexity: O(#distinct decision states x
branching factor) simulator calls; memoisation merges action orders leading to
the same state.  If more than `node_limit` distinct states would be expanded the
search stops and reports status "not_computed" instead of guessing.  Ties are
broken by the first action in `legal_actions()` order (ascending ids, HOLD last).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from .sim import RailwaySim, Scenario


@dataclass
class OptResult:
    status: str                     # "optimal" | "not_computed"
    total_waiting: int | None
    episode_return: float | None
    actions: list | None            # an optimal action sequence (replayable)
    states_expanded: int


class _LimitReached(Exception):
    pass


def solve(scenario: Scenario, node_limit: int = 2_000_000, delay_weight: float = 0.0) -> OptResult:
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20_000))
    root = RailwaySim(scenario, delay_weight=delay_weight, record=False)
    root.reset()
    memo: dict = {}

    def value(sim: RailwaySim) -> float:
        if sim.done:
            return 0.0
        key = sim.state_key()
        if key in memo:
            return memo[key][0]
        if len(memo) >= node_limit:
            raise _LimitReached
        best, best_a = float("-inf"), None
        for a in sim.legal_actions():
            nxt = sim.clone()
            v = nxt.apply(a) + value(nxt)
            if v > best:
                best, best_a = v, a
        memo[key] = (best, best_a)
        return best

    try:
        future = value(root)
    except _LimitReached:
        return OptResult("not_computed", None, None, None, len(memo))
    actions, sim = [], root.clone()
    while not sim.done:
        a = memo[sim.state_key()][1]
        actions.append(a)
        sim.apply(a)
    assert sim.episode_return == root.reward_before_first_decision + future
    return OptResult("optimal", sim.total_waiting, sim.episode_return, actions, len(memo))


def replay(scenario: Scenario, actions, delay_weight: float = 0.0) -> RailwaySim:
    """Run the simulator with a fixed action sequence (one action per decision)."""
    sim = RailwaySim(scenario, delay_weight=delay_weight)
    sim.reset()
    for a in actions:
        sim.apply(a)
    assert sim.done, "action sequence ended before the episode"
    return sim
