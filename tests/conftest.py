"""Shared tiny, hand-checkable fixtures."""

from __future__ import annotations

import pytest

from railway_gnn_rl.sim import Infrastructure, RailwaySim, Scenario, Section, TrainSpec


def line_infra(kind0: str = "single", kind1: str = "single", cap0: int = 1, cap1: int = 1,
               node_capacity=None) -> Infrastructure:
    """A --(s0, 2 min)-- B --(s1, 3 min)-- C"""
    return Infrastructure(
        node_names=("A", "B", "C"),
        sections=(
            Section(0, 0, 1, travel_time=2, capacity=cap0, kind=kind0),
            Section(1, 1, 2, travel_time=3, capacity=cap1, kind=kind1),
        ),
        node_capacity=node_capacity,
    )


def make_sim(trains, infra=None, max_time=None) -> RailwaySim:
    infra = infra or line_infra()
    specs = tuple(TrainSpec(i, tuple(r), d) for i, (r, d) in enumerate(trains))
    sim = RailwaySim(Scenario(infra, specs, max_time))
    sim.reset()
    return sim


def run_to_end(sim: RailwaySim, choose=lambda d: d.candidates[0]) -> list:
    """Run until termination; `choose` maps a Decision to a train id. Returns decisions seen."""
    seen = []
    while not sim.done:
        assert sim.decision is not None
        seen.append(sim.decision)
        sim.apply(choose(sim.decision))
    return seen


@pytest.fixture
def line():
    return line_infra()
