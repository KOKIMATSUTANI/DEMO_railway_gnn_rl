"""Graph observation construction.

The railway infrastructure becomes a directed graph:
  * one graph node per station/junction (N nodes, features `x` [N, F_node]);
  * two directed edges per physical section s: edge 2s = u->v, edge 2s+1 = v->u
    (`edge_index` [2, 2S] long, features `edge_attr` [2S, F_edge]).
Train-level information for the decision enters through one feature row per
legal action (`cand_feat` [K, F_train]); K = number of candidate trains plus 1
if HOLD is allowed.  For each action, `cand_node` / `cand_next_node` give the
graph nodes whose embeddings the policy combines (waiting node and next node
of the train; for HOLD the endpoints u, v of the contested section) and
`cand_edge` the directed edge the train would enter (2s for HOLD).

All features are float32 and scaled by fixed constants (no data-dependent
normalisation): counts of trains by the number of trains n, times by
TIME_NORM = 30 min, travel/remaining times by TRAVEL_NORM = 10 min.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from .sim import HOLD, RailwaySim, Section

TIME_NORM = 30.0
TRAVEL_NORM = 10.0
SECTIONS_NORM = 5.0

NODE_FEATURES = (
    "is_terminal",         # static: node has exactly one section
    "is_junction",         # static: node has >= 3 sections
    "waiting_trains",      # trains ready and waiting at the node / n
    "waiting_minutes",     # sum of waiting counters of those trains / TIME_NORM
    "inbound_trains",      # trains on a section heading to the node / n
    "destination_demand",  # unfinished trains whose destination is the node / n
    "pending_departures",  # not-yet-departed trains starting at the node / n
    "candidate_here",      # 1 if a candidate of the current decision waits here
)

EDGE_FEATURES = (
    "travel_time",            # static: section travel time / TRAVEL_NORM
    "capacity",               # static: capacity / 2
    "is_single",              # static: 1 for single track (shared by both directions)
    "occupancy_same_dir",     # trains on the section in this direction / capacity
    "occupancy_opposite_dir", # trains on the section in the opposite direction / capacity
    "min_remaining_time",     # min remaining traversal time on the section / TRAVEL_NORM
    "future_demand",          # unfinished trains that still have to use this directed edge / n
    "requests",               # trains waiting at the tail node for this edge / n
    "approaching",            # trains on a section whose next directed edge is this one / n
    "contested",              # 1 if the section is the one of the current decision
)

TRAIN_FEATURES = (
    "is_hold",                # 1 for the HOLD action, else 0
    "direction",              # +1 / -1 on the contested section (0 for HOLD)
    "route_progress",         # sections completed / sections on route
    "remaining_run_time",     # pure running time still ahead / TIME_NORM
    "waiting",                # waiting train-minutes so far / TIME_NORM
    "projected_lateness",     # (t + remaining running time - scheduled arrival) / TIME_NORM
    "remaining_sections",     # sections still ahead / SECTIONS_NORM
    "time_since_departure",   # (t - departure) / TIME_NORM
    "contested_travel_time",  # travel time of the contested section / TRAVEL_NORM (all actions)
    "n_competitors",          # trains a grant would delay (see RailwaySim.competitors) / n
    "min_competitor_eta",     # minutes until the first competitor requests / TRAVEL_NORM
)


class DispatchGraph(Data):
    """PyG Data with candidate index fields that are offset correctly when batched."""

    def __inc__(self, key, value, *args, **kwargs):
        if key in ("cand_node", "cand_next_node"):
            return self.num_nodes
        if key == "cand_edge":
            return self.edge_index.size(1)
        return super().__inc__(key, value, *args, **kwargs)


def directed_edge(s: Section, direction: int) -> int:
    return 2 * s.id + (0 if direction == 1 else 1)


def _remaining_edges(sim: RailwaySim, i: int) -> list[int]:
    """Directed edges train i still has to enter."""
    st = sim.trains[i]
    route = sim.scenario.trains[i].route
    start = {"pending": 0, "at_node": st.route_idx, "on_section": st.route_idx + 1}.get(st.status)
    if start is None:
        return []
    out = []
    for a, b in zip(route[start:], route[start + 1:]):
        s = sim.infra.section_between(a, b)
        out.append(directed_edge(s, 1 if (a, b) == (s.u, s.v) else -1))
    return out


def build_observation(sim: RailwaySim) -> DispatchGraph:
    infra = sim.infra
    n_nodes, n_sec, n = len(infra.node_names), len(infra.sections), len(sim.trains)
    specs = sim.scenario.trains
    degree = [0] * n_nodes
    for s in infra.sections:
        degree[s.u] += 1
        degree[s.v] += 1

    d = sim.decision
    actions = sim.legal_actions()
    cand_trains = list(d.candidates) if d is not None else []

    x = torch.zeros(n_nodes, len(NODE_FEATURES))
    e = torch.zeros(2 * n_sec, len(EDGE_FEATURES))
    NF, EF = NODE_FEATURES.index, EDGE_FEATURES.index
    for v in range(n_nodes):
        x[v, NF("is_terminal")] = float(degree[v] == 1)
        x[v, NF("is_junction")] = float(degree[v] >= 3)

    for i, st in enumerate(sim.trains):
        route = specs[i].route
        if st.status == "finished":
            continue
        x[route[-1], NF("destination_demand")] += 1 / n
        for k in _remaining_edges(sim, i):
            e[k, EF("future_demand")] += 1 / n
        if st.status == "pending":
            x[route[0], NF("pending_departures")] += 1 / n
        elif st.status == "at_node":
            node = route[st.route_idx]
            x[node, NF("waiting_trains")] += 1 / n
            x[node, NF("waiting_minutes")] += st.waiting / TIME_NORM
            s = sim.next_section(i)
            e[directed_edge(s, sim.direction_on(i, s)), EF("requests")] += 1 / n
            if i in cand_trains:
                x[node, NF("candidate_here")] = 1.0
        elif st.status == "on_section":
            x[route[st.route_idx + 1], NF("inbound_trains")] += 1 / n
            nxt = _remaining_edges(sim, i)
            if nxt:
                e[nxt[0], EF("approaching")] += 1 / n

    src, dst = [], []
    for s in infra.sections:
        for direction, (a, b) in ((1, (s.u, s.v)), (-1, (s.v, s.u))):
            k = directed_edge(s, direction)
            src.append(a)
            dst.append(b)
            dirs = [sim.trains[j].direction for j in sim.occupants[s.id]]
            e[k, EF("travel_time")] = s.travel_time / TRAVEL_NORM
            e[k, EF("capacity")] = s.capacity / 2
            e[k, EF("is_single")] = float(s.kind == "single")
            e[k, EF("occupancy_same_dir")] = dirs.count(direction) / s.capacity
            e[k, EF("occupancy_opposite_dir")] = dirs.count(-direction) / s.capacity
            if sim.occupants[s.id]:
                rem = min(sim.trains[j].remaining for j in sim.occupants[s.id])
                e[k, EF("min_remaining_time")] = rem / TRAVEL_NORM
            if d is not None and d.section == s.id:
                e[k, EF("contested")] = 1.0

    cand_node, cand_next, cand_edge = [], [], []
    feats = torch.zeros(len(actions), len(TRAIN_FEATURES))
    if d is not None:
        s = infra.sections[d.section]
        comps = sim.competitors(s, d.group, cand_trains)
        TF = TRAIN_FEATURES.index
        for row, a in enumerate(actions):
            feats[row, TF("contested_travel_time")] = s.travel_time / TRAVEL_NORM
            feats[row, TF("n_competitors")] = len(comps) / n
            feats[row, TF("min_competitor_eta")] = (min(c[1] for c in comps) / TRAVEL_NORM
                                                     if comps else 0.0)
            if a == HOLD:
                feats[row, TF("is_hold")] = 1.0
                cand_node.append(s.u)
                cand_next.append(s.v)
                cand_edge.append(directed_edge(s, 1))
                continue
            st, spec = sim.trains[a], specs[a]
            route = spec.route
            direction = sim.direction_on(a, s)
            remaining_run = sum(infra.section_between(p, q).travel_time
                                for p, q in zip(route[st.route_idx:], route[st.route_idx + 1:]))
            n_sections = len(route) - 1
            feats[row, TF("direction")] = float(direction)
            feats[row, TF("route_progress")] = st.route_idx / n_sections
            feats[row, TF("remaining_run_time")] = remaining_run / TIME_NORM
            feats[row, TF("waiting")] = st.waiting / TIME_NORM
            feats[row, TF("projected_lateness")] = (
                sim.t + remaining_run - sim.scheduled_arrival[a]) / TIME_NORM
            feats[row, TF("remaining_sections")] = (n_sections - st.route_idx) / SECTIONS_NORM
            feats[row, TF("time_since_departure")] = (sim.t - spec.departure) / TIME_NORM
            cand_node.append(route[st.route_idx])
            cand_next.append(route[st.route_idx + 1])
            cand_edge.append(directed_edge(s, direction))

    return DispatchGraph(
        x=x,
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=e,
        cand_feat=feats,
        cand_node=torch.tensor(cand_node, dtype=torch.long),
        cand_next_node=torch.tensor(cand_next, dtype=torch.long),
        cand_edge=torch.tensor(cand_edge, dtype=torch.long),
        num_cands=len(actions),
        actions=actions,
        num_nodes=n_nodes,
    )
