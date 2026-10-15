"""Demo infrastructure (7 nodes, 8 sections) and synthetic scenario generation.

Layout (travel times in minutes; "1-track" = single track, capacity 1 total;
"1-track x2" = single track admitting 2 following trains in the same direction;
"2-track" = double track, 1 train per direction):

            N --------s3 (1-track, 4)-------.
          / |                                \\
  s1 (1-track, 4)  s6 (1-track, 3)            J2 --s5 (2-track, 3)-- E
        /   |                                / \\
 W --s0-- J1  |                             /   s7 (1-track, 6)
 (2-track,3)\\ |                            /     \\
            s2 (2-track, 5)   s4 (1-track x2, 5)    H
              \\ |           /
                S ---------'

All stations have unlimited capacity, so the network is deadlock-free by
construction (see sim.py).  The data are synthetic; no GTFS/OSM is used.
"""

from __future__ import annotations

import random

from .sim import Infrastructure, Scenario, Section, TrainSpec

W, J1, N, S, J2, E, H = range(7)
NODE_NAMES = ("W", "J1", "N", "S", "J2", "E", "H")

N_TRAINS = 5
DEP_LOW, DEP_HIGH = 0, 8       # training/validation/evaluation departure window (minutes)
HELDOUT_DEP_HIGH = 4           # held-out set: denser departures in [0, 4]
SLACK = 2                      # schedule buffer per train (minutes)


def demo_infrastructure() -> Infrastructure:
    return Infrastructure(
        node_names=NODE_NAMES,
        sections=(
            Section(0, W, J1, travel_time=3, capacity=1, kind="double"),
            Section(1, J1, N, travel_time=4, capacity=1, kind="single"),
            Section(2, J1, S, travel_time=5, capacity=1, kind="double"),
            Section(3, N, J2, travel_time=4, capacity=1, kind="single"),
            Section(4, S, J2, travel_time=5, capacity=2, kind="single"),
            Section(5, J2, E, travel_time=3, capacity=1, kind="double"),
            Section(6, N, S, travel_time=3, capacity=1, kind="single"),
            Section(7, J2, H, travel_time=6, capacity=1, kind="single"),
        ),
        node_capacity=None,
    )


# Predefined route templates (node sequences).  The agent never routes; it only
# decides the order in which competing trains use a contested section.
ROUTES: tuple[tuple[int, ...], ...] = (
    (W, J1, N, J2, E),      # 0 eastbound via north
    (W, J1, S, J2, E),      # 1 eastbound via south
    (E, J2, N, J1, W),      # 2 westbound via north
    (E, J2, S, J1, W),      # 3 westbound via south
    (W, J1, N, J2, H),      # 4 west -> harbour
    (H, J2, S, J1, W),      # 5 harbour -> west
    (E, J2, H),             # 6 east -> harbour
    (H, J2, E),             # 7 harbour -> east
    (N, S, J2, H),          # 8 local freight north -> harbour (uses crossover s6)
    (H, J2, S, N),          # 9 local freight harbour -> north
)


def generate_scenario(seed: int, dep_low: int = DEP_LOW, dep_high: int = DEP_HIGH) -> Scenario:
    """5 trains; each picks a route template uniformly and an integer departure
    time uniformly in [dep_low, dep_high], using only `random.Random(seed)`."""
    rng = random.Random(seed)
    trains = tuple(
        TrainSpec(i, ROUTES[rng.randrange(len(ROUTES))], rng.randint(dep_low, dep_high), SLACK)
        for i in range(N_TRAINS)
    )
    return Scenario(demo_infrastructure(), trains)
