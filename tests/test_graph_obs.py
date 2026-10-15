"""Cycle 5b: graph observation construction (tiny deterministic graph)."""

import pytest
import torch

from conftest import make_sim
from railway_gnn_rl.graph_obs import EDGE_FEATURES, NODE_FEATURES, TRAIN_FEATURES, build_observation
from railway_gnn_rl.scenarios import generate_scenario
from railway_gnn_rl.sim import HOLD, RailwaySim

A, B, C = 0, 1, 2


@pytest.fixture
def conflict_sim():
    # T0: A->B->C departs 0 and enters s0 at 0 (arrives B at 2).
    # T1, T2: C->B depart 0 and compete for single-track s1 (3 min).
    # HOLD is offered: T0 will request s1 in 2 < 3 minutes.
    return make_sim([((A, B, C), 0), ((C, B), 0), ((C, B), 0)])


def f(name, names):
    return names.index(name)


def test_topology_counts_shapes_and_dtypes(conflict_sim):
    obs = build_observation(conflict_sim)
    assert obs.x.shape == (3, len(NODE_FEATURES)) and obs.x.dtype == torch.float32
    assert obs.edge_index.shape == (2, 4) and obs.edge_index.dtype == torch.long
    assert obs.edge_attr.shape == (4, len(EDGE_FEATURES)) and obs.edge_attr.dtype == torch.float32
    # directed edge 2s is u->v, 2s+1 is v->u
    assert obs.edge_index.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]
    assert obs.cand_feat.shape == (3, len(TRAIN_FEATURES)) and obs.cand_feat.dtype == torch.float32
    for t in (obs.x, obs.edge_attr, obs.cand_feat):
        assert torch.isfinite(t).all()


def test_actions_and_candidate_indexing(conflict_sim):
    obs = build_observation(conflict_sim)
    assert obs.actions == [1, 2, HOLD]
    assert obs.cand_node.tolist() == [C, C, B]        # HOLD uses the section endpoints (u, v)
    assert obs.cand_next_node.tolist() == [B, B, C]
    assert obs.cand_edge.tolist() == [3, 3, 2]        # s1 traversed v->u; HOLD -> edge 2s


def test_hand_calculated_node_and_edge_features(conflict_sim):
    obs = build_observation(conflict_sim)
    n = 3
    x, e = obs.x, obs.edge_attr
    assert x[C, f("waiting_trains", NODE_FEATURES)] == pytest.approx(2 / n)
    assert x[C, f("candidate_here", NODE_FEATURES)] == 1.0
    assert x[A, f("candidate_here", NODE_FEATURES)] == 0.0
    assert x[B, f("inbound_trains", NODE_FEATURES)] == pytest.approx(1 / n)   # T0 heading to B
    assert x[B, f("destination_demand", NODE_FEATURES)] == pytest.approx(2 / n)
    assert x[A, f("is_terminal", NODE_FEATURES)] == 1.0 and x[B, f("is_terminal", NODE_FEATURES)] == 0.0
    # edge 0 = A->B carries T0 (remaining 2); edge 1 = B->A is the opposing direction
    assert e[0, f("occupancy_same_dir", EDGE_FEATURES)] == 1.0
    assert e[1, f("occupancy_opposite_dir", EDGE_FEATURES)] == 1.0
    assert e[0, f("min_remaining_time", EDGE_FEATURES)] == pytest.approx(2 / 10)
    assert e[0, f("travel_time", EDGE_FEATURES)] == pytest.approx(2 / 10)
    assert e[3, f("requests", EDGE_FEATURES)] == pytest.approx(2 / n)
    assert e[3, f("contested", EDGE_FEATURES)] == 1.0 and e[2, f("contested", EDGE_FEATURES)] == 1.0
    assert e[0, f("contested", EDGE_FEATURES)] == 0.0
    # T0 still has to traverse B->C and is approaching it (next directed edge)
    assert e[2, f("future_demand", EDGE_FEATURES)] == pytest.approx(1 / n)
    assert e[2, f("approaching", EDGE_FEATURES)] == pytest.approx(1 / n)


def test_hold_action_features(conflict_sim):
    obs = build_observation(conflict_sim)
    hold = obs.cand_feat[2]
    assert hold[f("is_hold", TRAIN_FEATURES)] == 1.0
    assert hold[f("waiting", TRAIN_FEATURES)] == 0.0 and hold[f("direction", TRAIN_FEATURES)] == 0.0
    assert hold[f("n_competitors", TRAIN_FEATURES)] == pytest.approx(1 / 3)
    assert hold[f("min_competitor_eta", TRAIN_FEATURES)] == pytest.approx(2 / 10)
    assert obs.cand_feat[0, f("is_hold", TRAIN_FEATURES)] == 0.0


def test_graph_updates_after_train_movements(conflict_sim):
    sim = conflict_sim
    before = build_observation(sim)
    sim.apply(1)
    # t=3: T1 has arrived at B; T0 (at B since 2) and T2 (at C since 0) compete for s1.
    assert sim.decision.t == 3 and sim.decision.candidates == (0, 2)
    after = build_observation(sim)
    assert after.actions == [0, 2]
    x = after.x
    assert x[B, f("waiting_trains", NODE_FEATURES)] == pytest.approx(1 / 3)
    assert x[C, f("waiting_trains", NODE_FEATURES)] == pytest.approx(1 / 3)
    assert x[C, f("waiting_minutes", NODE_FEATURES)] == pytest.approx(3 / 30)
    assert after.edge_attr[0, f("occupancy_same_dir", EDGE_FEATURES)] == 0.0
    assert not torch.equal(before.x, after.x)
    # candidate T2 features at t=3 (hand-calculated)
    tf = after.cand_feat[after.actions.index(2)]
    assert tf[f("direction", TRAIN_FEATURES)] == -1.0
    assert tf[f("route_progress", TRAIN_FEATURES)] == 0.0
    assert tf[f("remaining_run_time", TRAIN_FEATURES)] == pytest.approx(3 / 30)
    assert tf[f("waiting", TRAIN_FEATURES)] == pytest.approx(3 / 30)
    assert tf[f("projected_lateness", TRAIN_FEATURES)] == pytest.approx((3 + 3 - 5) / 30)
    assert tf[f("contested_travel_time", TRAIN_FEATURES)] == pytest.approx(3 / 10)


def test_demo_scenario_observation_has_seven_nodes_and_sixteen_directed_edges():
    for seed in range(20):
        sim = RailwaySim(generate_scenario(seed))
        sim.reset()
        if sim.decision is not None:
            break
    assert sim.decision is not None
    obs = build_observation(sim)
    assert obs.x.shape[0] == 7 and obs.edge_index.shape[1] == 16
    assert obs.cand_feat.shape[0] == len(sim.legal_actions())
