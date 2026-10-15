"""Cycle 6: GNN forward pass, output shapes and GNN-to-policy integration."""

import pytest
import torch
from torch_geometric.data import Batch

from railway_gnn_rl.graph_obs import NODE_FEATURES, build_observation
from railway_gnn_rl.models import make_policy
from railway_gnn_rl.scenarios import generate_scenario
from railway_gnn_rl.sim import RailwaySim


def demo_obs(n=3):
    out = []
    for seed in range(200):
        sim = RailwaySim(generate_scenario(seed))
        sim.reset()
        if sim.decision is not None and len(sim.legal_actions()) >= 2:
            out.append(build_observation(sim))
        if len(out) == n:
            return out
    raise AssertionError("not enough decision states")


@pytest.mark.parametrize("kind", ["gnn", "mlp"])
def test_encoder_and_policy_output_shapes_and_finiteness(kind):
    torch.manual_seed(0)
    pol = make_policy(kind, hidden=16, layers=2)
    obs = demo_obs(3)
    batch = Batch.from_data_list(obs)
    h, g = pol.encoder(batch)
    assert h.shape == (3 * 7, 16) and g.shape == (3, 16)
    scores = pol(batch)
    assert scores.shape == (sum(o.num_cands for o in obs),)
    assert torch.isfinite(scores).all()


@pytest.mark.parametrize("kind", ["gnn", "mlp"])
def test_batched_scores_equal_individual_scores(kind):
    torch.manual_seed(0)
    pol = make_policy(kind, hidden=16, layers=2)
    obs = demo_obs(3)
    batched = pol(Batch.from_data_list(obs))
    single = torch.cat([pol(Batch.from_data_list([o])) for o in obs])
    assert torch.allclose(batched, single, atol=1e-6)


def two_hop_node(obs):
    """A node at graph distance exactly 2 from the first candidate's waiting node."""
    ei = obs.edge_index
    start = int(obs.cand_node[0])
    nbr = lambda v: set(ei[1][ei[0] == v].tolist())
    one = nbr(start)
    two = set().union(*(nbr(v) for v in one)) - one - {start}
    assert two
    return start, min(two)


def test_message_passing_propagates_remote_state_into_candidate_embedding():
    torch.manual_seed(0)
    obs = demo_obs(1)[0]
    start, far = two_hop_node(obs)
    changed = obs.clone()
    changed.x = obs.x.clone()
    changed.x[far, NODE_FEATURES.index("waiting_trains")] += 0.6
    for kind, expect_change in (("gnn", True), ("mlp", False)):
        pol = make_policy(kind, hidden=16, layers=2)
        h0, _ = pol.encoder(Batch.from_data_list([obs]))
        h1, _ = pol.encoder(Batch.from_data_list([changed]))
        diff = (h0[start] - h1[start]).abs().max().item()
        assert (diff > 1e-6) == expect_change, (kind, diff)


def test_changing_graph_state_changes_action_scores():
    torch.manual_seed(0)
    pol = make_policy("gnn", hidden=16, layers=2)
    obs = demo_obs(1)[0]
    changed = obs.clone()
    changed.edge_attr = obs.edge_attr.clone()
    changed.edge_attr[:, 3] += 0.5          # occupancy everywhere
    s0 = pol(Batch.from_data_list([obs]))
    s1 = pol(Batch.from_data_list([changed]))
    assert not torch.allclose(s0, s1)


def test_gradients_reach_all_gnn_parameters():
    torch.manual_seed(0)
    pol = make_policy("gnn", hidden=16, layers=2)
    batch = Batch.from_data_list(demo_obs(3))
    pol(batch).pow(2).sum().backward()
    for name, p in pol.encoder.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_gnn_and_mlp_parameter_counts_are_comparable():
    count = lambda m: sum(p.numel() for p in m.parameters())
    g, m = count(make_policy("gnn", 32, 2)), count(make_policy("mlp", 32, 2))
    assert 0.75 <= m / g <= 1.33, (g, m)
