"""Cycles 7-8: action selection, masking, transition storage and training updates."""

import copy

import torch

from railway_gnn_rl.agent import (ReinforceTrainer, action_distribution, action_mask,
                                  run_episode, select_action)
from railway_gnn_rl.graph_obs import build_observation
from railway_gnn_rl.models import make_policy
from railway_gnn_rl.scenarios import generate_scenario
from railway_gnn_rl.sim import HOLD, RailwaySim, Scenario, TrainSpec
from test_sim_hold import TRAINS, crossing_infra


def crossing_scenario():
    return Scenario(crossing_infra(), tuple(TrainSpec(i, r, d) for i, (r, d) in enumerate(TRAINS)))


def decision_sim():
    sim = RailwaySim(crossing_scenario())
    sim.reset()
    return sim          # decision: candidates (0,), HOLD allowed


def test_action_mask_marks_exactly_the_legal_actions():
    sim = decision_sim()
    mask = action_mask(sim)
    assert mask.dtype == torch.bool and mask.shape == (4,)      # 3 trains + HOLD
    assert mask.tolist() == [True, False, False, True]


def test_distribution_gives_zero_probability_to_illegal_actions():
    torch.manual_seed(0)
    pol = make_policy("gnn", 16, 2)
    sim = decision_sim()
    probs = action_distribution(pol, build_observation(sim), action_mask(sim)).probs
    assert probs.shape == (4,)
    assert probs[1] == 0 and probs[2] == 0
    assert torch.isfinite(probs).all() and abs(probs.sum().item() - 1) < 1e-6


def test_greedy_selection_is_deterministic_argmax():
    torch.manual_seed(0)
    pol = make_policy("gnn", 16, 2)
    sim = decision_sim()
    probs = action_distribution(pol, build_observation(sim), action_mask(sim)).probs
    a1, _, _ = select_action(pol, sim, "greedy")
    a2, _, _ = select_action(pol, sim, "greedy")
    expected = int(torch.argmax(probs))
    assert a1 == a2 == (HOLD if expected == 3 else expected)


def test_sampling_only_returns_legal_actions_and_is_reproducible():
    torch.manual_seed(0)
    pol = make_policy("gnn", 16, 2)
    sim = decision_sim()
    g =torch.Generator().manual_seed(5)
    acts = [select_action(pol, sim, "sample", g)[0] for _ in range(200)]
    assert set(acts) <= {0, HOLD}
    g1, g2 = torch.Generator().manual_seed(9), torch.Generator().manual_seed(9)
    assert [select_action(pol, sim, "sample", g1)[0] for _ in range(20)] == \
           [select_action(pol, sim, "sample", g2)[0] for _ in range(20)]


def test_run_episode_stores_transitions_and_return_equals_sum_of_rewards():
    torch.manual_seed(0)
    pol = make_policy("gnn", 16, 2)
    for seed in range(5):
        rec, trans = run_episode(generate_scenario(seed), pol, "greedy")
        assert rec.termination == "success"
        assert rec.episode_return == sum(rec.step_rewards) == -rec.total_waiting
        assert rec.decisions == len(trans) and rec.steps == len(rec.step_rewards)
        assert rec.reward_before_first_decision + sum(t.reward for t in trans) == rec.episode_return
        for t in trans:
            assert t.mask[t.action_index] and t.obs.num_cands == int(t.mask.sum())


def test_episode_without_decisions_gives_no_update():
    torch.manual_seed(0)
    pol = make_policy("gnn", 16, 2)
    before = copy.deepcopy(pol.state_dict())
    tr = ReinforceTrainer(pol, lr=0.1, seed=0)
    single = Scenario(crossing_infra(), (TrainSpec(0, (0, 1, 2), 0),))
    m = tr.train_episode(single)
    assert m["decisions"] == 0 and m["updated"] is False
    assert all(torch.equal(before[k], v) for k, v in pol.state_dict().items())


def test_valid_update_changes_parameters_and_stays_finite():
    torch.manual_seed(0)
    pol = make_policy("gnn", 16, 2)
    before = copy.deepcopy(pol.state_dict())
    tr = ReinforceTrainer(pol, lr=1e-2, entropy_coef=0.01, seed=0)
    m = tr.train_episode(crossing_scenario())
    assert m["updated"] is True and m["decisions"] >= 1
    assert torch.isfinite(torch.tensor(m["loss"]))
    assert m["grad_norm_clipped"] <= 1.0 + 1e-6
    changed = [k for k, v in pol.state_dict().items() if not torch.equal(before[k], v)]
    assert any(k.startswith("encoder.") for k in changed)      # gradients reached the GNN


def test_training_learns_the_hand_calculated_crossing_decision():
    # Greedy-optimal: HOLD T0 so the two followers pass together (W = 6 instead of 8).
    torch.manual_seed(0)
    pol = make_policy("gnn", 16, 2)
    tr = ReinforceTrainer(pol, lr=1e-2, entropy_coef=0.0, seed=0)
    for _ in range(150):
        tr.train_episode(crossing_scenario())
    rec, _ = run_episode(crossing_scenario(), pol, "greedy")
    assert rec.total_waiting == 6
