"""REINFORCE (Monte-Carlo policy gradient) agent with explicit action masking.

Fixed action indexing per scenario with n trains: index i < n = "grant the
contested section to train i", index n = HOLD.  The mask marks the legal
actions of the current decision (`RailwaySim.legal_actions`).  The policy
network scores only the legal actions; every illegal index gets logit -inf and
therefore probability exactly 0, so an illegal action can never be sampled.

Training update (one per training episode that contains >= 1 decision):
    G      = undiscounted return of an episode sampled from pi_theta  (gamma = 1)
    G_base = return of the greedy (argmax) rollout of pi_theta on the same
             scenario, computed without gradient (self-critical baseline)
    loss   = -((G - G_base) / adv_scale) * sum_k log pi_theta(a_k | s_k)
             - entropy_coef * sum_k H(pi_theta(. | s_k))
The baseline does not depend on the sampled actions, so the gradient estimate
stays unbiased.  Gradients flow through the action logits into the policy head
and the GNN encoder.  Rewards before the first decision do not depend on any
action; including them in G only shifts G and G_base equally.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Categorical
from torch_geometric.data import Batch

from .graph_obs import DispatchGraph, build_observation
from .sim import HOLD, RailwaySim, Scenario


def action_index(action: int, n_trains: int) -> int:
    return n_trains if action == HOLD else action


def index_action(index: int, n_trains: int) -> int:
    return HOLD if index == n_trains else index


def action_mask(sim: RailwaySim) -> torch.Tensor:
    n = len(sim.trains)
    mask = torch.zeros(n + 1, dtype=torch.bool)
    for a in sim.legal_actions():
        mask[action_index(a, n)] = True
    return mask


def action_distribution(policy, obs: DispatchGraph, mask: torch.Tensor) -> Categorical:
    n = mask.numel() - 1
    logits = policy(Batch.from_data_list([obs]))
    full = torch.full((n + 1,), float("-inf"))
    idx = torch.tensor([action_index(a, n) for a in obs.actions], dtype=torch.long)
    assert bool(mask[idx].all()) and int(mask.sum()) == len(obs.actions)
    full = full.index_put((idx,), logits)
    return Categorical(logits=full)


def _choose(dist: Categorical, mode: str, generator=None) -> int:
    if mode == "greedy":
        return int(torch.argmax(dist.probs))          # ties -> lowest index
    if mode == "sample":
        return int(torch.multinomial(dist.probs, 1, generator=generator))
    raise ValueError(mode)


def select_action(policy, sim: RailwaySim, mode: str = "greedy", generator=None):
    """Returns (action, log_prob, entropy) for the pending decision."""
    dist = action_distribution(policy, build_observation(sim), action_mask(sim))
    idx = _choose(dist, mode, generator)
    return index_action(idx, len(sim.trains)), dist.log_prob(torch.tensor(idx)), dist.entropy()


@dataclass
class Transition:
    obs: DispatchGraph
    mask: torch.Tensor
    action_index: int
    log_prob: torch.Tensor
    entropy: torch.Tensor
    reward: float          # sum of step rewards until the next decision / episode end


@dataclass
class EpisodeRecord:
    episode_return: float
    total_waiting: int
    total_delay: int
    decisions: int
    holds: int
    steps: int
    termination: str
    step_rewards: list
    reward_before_first_decision: float


def record_of(sim: RailwaySim, decisions: int, holds: int) -> EpisodeRecord:
    return EpisodeRecord(sim.episode_return, sim.total_waiting, sim.total_delay, decisions, holds,
                         len(sim.step_rewards), sim.termination, list(sim.step_rewards),
                         sim.reward_before_first_decision)


def run_episode(scenario: Scenario, policy, mode: str = "greedy", generator=None,
                delay_weight: float = 0.0, track_grad: bool = False):
    sim = RailwaySim(scenario, delay_weight=delay_weight)
    sim.reset()
    trans: list[Transition] = []
    holds = 0
    with torch.set_grad_enabled(track_grad):
        while not sim.done:
            obs, mask = build_observation(sim), action_mask(sim)
            dist = action_distribution(policy, obs, mask)
            idx = _choose(dist, mode, generator)
            a = index_action(idx, len(sim.trains))
            holds += a == HOLD
            r = sim.apply(a)
            trans.append(Transition(obs, mask, idx, dist.log_prob(torch.tensor(idx)),
                                    dist.entropy(), r))
    return record_of(sim, len(trans), holds), trans


class ReinforceTrainer:
    def __init__(self, policy, lr=1e-3, entropy_coef=0.01, adv_scale=10.0, grad_clip=1.0,
                 delay_weight=0.0, seed=0):
        self.policy = policy
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.entropy_coef = entropy_coef
        self.adv_scale = adv_scale
        self.grad_clip = grad_clip
        self.delay_weight = delay_weight
        self.generator = torch.Generator().manual_seed(seed)

    def train_episode(self, scenario: Scenario) -> dict:
        self.policy.train()
        rec, trans = run_episode(scenario, self.policy, "sample", self.generator,
                                 self.delay_weight, track_grad=True)
        m = {"episode_return": rec.episode_return, "total_waiting": rec.total_waiting,
             "total_delay": rec.total_delay, "decisions": rec.decisions, "holds": rec.holds,
             "steps": rec.steps, "termination": rec.termination, "updated": False,
             "loss": float("nan"), "entropy": float("nan"), "baseline_return": float("nan"),
             "advantage": float("nan"), "grad_norm": float("nan"),
             "grad_norm_clipped": float("nan"), "step_rewards": rec.step_rewards}
        if not trans:
            return m
        base, _ = run_episode(scenario, self.policy, "greedy", delay_weight=self.delay_weight)
        adv = rec.episode_return - base.episode_return
        logp = torch.stack([t.log_prob for t in trans])
        ent = torch.stack([t.entropy for t in trans])
        loss = -(adv / self.adv_scale) * logp.sum() - self.entropy_coef * ent.sum()
        self.opt.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        clipped = torch.sqrt(sum((p.grad.detach() ** 2).sum()
                                 for p in self.policy.parameters() if p.grad is not None))
        self.opt.step()
        m.update(updated=True, loss=float(loss.detach()), entropy=float(ent.detach().mean()),
                 baseline_return=base.episode_return, advantage=adv,
                 grad_norm=float(norm), grad_norm_clipped=float(clipped))
        return m
