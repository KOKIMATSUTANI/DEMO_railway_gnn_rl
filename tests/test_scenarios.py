"""Cycle 5a: demo infrastructure and scenario generator."""

from railway_gnn_rl.scenarios import ROUTES, demo_infrastructure, generate_scenario
from railway_gnn_rl.sim import RailwaySim


def test_demo_infrastructure_has_fixed_scale_and_constrained_sections():
    infra = demo_infrastructure()
    assert len(infra.node_names) == 7
    assert len(infra.sections) == 8
    assert infra.node_capacity is None            # unlimited stations -> deadlock-free
    kinds = [(s.kind, s.capacity) for s in infra.sections]
    assert ("single", 1) in kinds                 # classic single track
    assert any(k != ("single", 1) for k in kinds)  # further constrained section types
    assert all(isinstance(s.travel_time, int) and s.travel_time > 0 for s in infra.sections)
    pairs = {frozenset((s.u, s.v)) for s in infra.sections}
    assert len(pairs) == 8                        # no parallel duplicates


def test_routes_are_connected_simple_paths_and_use_every_section():
    infra = demo_infrastructure()
    used = set()
    for r in ROUTES:
        assert len(set(r)) == len(r)
        for a, b in zip(r, r[1:]):
            used.add(infra.section_between(a, b).id)
    assert used == set(range(8))


def test_generator_is_deterministic_and_respects_documented_limits():
    a, b = generate_scenario(123), generate_scenario(123)
    assert a == b
    assert len(a.trains) == 5
    assert all(0 <= tr.departure <= 8 and tr.route in ROUTES for tr in a.trains)
    assert any(generate_scenario(s).trains[0].departure == 8 for s in range(100))
    assert generate_scenario(124) != a
    shifted = generate_scenario(7, dep_low=0, dep_high=5)
    assert all(tr.departure <= 5 for tr in shifted.trains)


def test_generated_scenarios_produce_sequential_decisions():
    n_dec = []
    for seed in range(50):
        sim = RailwaySim(generate_scenario(seed))
        sim.reset()
        k = 0
        while not sim.done:
            sim.apply(sim.decision.candidates[0])
            k += 1
        assert sim.termination == "success"
        n_dec.append(k)
    assert sum(n_dec) / len(n_dec) >= 2.0         # on average several decisions per episode
    assert max(n_dec) >= 4
