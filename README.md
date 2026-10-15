# GNN + Reinforcement Learning for a Small Railway Dispatching Problem

An educational, fully reproducible prototype: a discrete-time railway simulator
(7 stations/junctions, 8 sections, 5 trains), a Graph Neural Network (GNN) that
reads the network state as a graph, and a policy-gradient Reinforcement Learning
(RL) agent that decides which train may use a contested track section. The
learned policy is compared with simple dispatching rules, with an **exact
optimum** computed by exhaustive search, and with an **MLP ablation** that sees
the same inputs but has no message passing.

This is not a production railway simulator, and it does not claim that RL beats
conventional methods. Section 9 reports what was measured.

Contents
1. [Problem overview](#1-problem-overview)
2. [Simulation specification and MDP](#2-simulation-specification-and-mdp)
3. [Input data and data flow](#3-input-data-and-data-flow)
4. [Graph structure](#4-graph-structure)
5. [Features and tensor shapes](#5-node-features-edge-features-and-tensor-shapes)
6. [GNN architecture](#6-gnn-architecture)
7. [RL agent and the MLP ablation](#7-rl-agent-architecture-and-the-mlp-ablation)
8. [Reward function](#8-reward-function)
9. [Baselines, exact optimum, statistics, and results](#9-baselines-exact-optimum-statistical-method-and-results)
10. [Figures: how to read them](#10-evaluation-figures-how-to-read-them)
11. [Quick start and reproducibility](#11-quick-start-and-reproducibility)
12. [Limitations](#12-limitations-and-future-extensions)
13. [Development process (TDD) and deviations from the task specification](#13-development-process-and-deviations-from-the-specification)

Glossary: **GNN** graph neural network · **RL** reinforcement learning · **MDP**
Markov decision process · **MLP** multi-layer perceptron (plain feed-forward
network) · **FCFS** first come, first served · **CI** confidence interval ·
**SD** standard deviation · **train-minute** one train waiting for one minute.

---

## 1. Problem overview

Trains share track. On a **single-track** section only one direction can be
used at a time. Whenever several trains want the same section at the same
time, someone has to decide who goes first. That person is the dispatcher.
The losers wait at their station. **The agent plays the dispatcher.** It is
asked only when a real choice exists, and it chooses either

* which competing train enters the contested section now, or
* **HOLD**: let nobody enter this minute, because a train that is about to
  arrive would otherwise be blocked (a "crossing decision").

**What is optimised:** total waiting time `W` in train-minutes, summed over all
trains and the whole episode.

**What is simplified compared with a real railway:** no signals, blocks, or
braking curves; a section is either free or occupied; travel times are fixed
integers; stations have unlimited platforms; routes are fixed (no re-routing);
no rolling-stock or crew constraints; synthetic timetables.

### Worked example: two trains, one single-track section (with HOLD)

Network `A --s0-- B --s1-- C`. `s0` is double track (2 trains per direction,
1 min). `s1` is single track, 5 min, and admits 2 *following* trains in the
same direction. T0 starts at C towards B at t=0. T1 and T2 start at A at t=0
towards C, so they reach B at t=1.

| | grant T0 immediately | HOLD T0 at t=0 |
|---|---|---|
| s1 timeline | T0 on s1 during t=0..4 | T1+T2 on s1 together during t=1..5 |
| T0 | arrives B at 5, waits 0 | waits t=0..5 (6 min), arrives B at 11 |
| T1, T2 | wait at B t=1..4 (4 min each), arrive C at 10 | arrive C at 6, wait 0 |
| **total waiting W** | **8 train-min** | **6 train-min** |

Holding one train costs 6 minutes and saves 8. This exact case is a unit test
(`tests/test_sim_hold.py`). The exact solver finds it, and a GNN policy trained
only on this scenario learns it (`tests/test_agent.py`).

---

## 2. Simulation specification and MDP

### 2.1 Time, units, rounding

* **Time step Δt = 1 minute.** All times (departures, travel times, arrivals) are
  integer minutes. Waiting time is counted in train-minutes.
* **Assumption:** travel times are positive integers. Every event therefore falls
  on the 1-minute grid. There are no sub-step events, so no rounding and no
  discretisation error occur. This is tested (`test_vv.py::test_time_step_consistency_no_sub_step_events`):
  every release happens exactly `travel_time` minutes after the entry.

### 2.2 Event order within one step (`src/railway_gnn_rl/sim.py`)

1. Advance clock t → t+1.
2. Trains that complete a section release it and arrive at the next node.
3. Trains at their destination are marked finished. Arrival time and delay are recorded.
4. Trains whose departure time equals t become ready at their origin.
5. Every ready, unfinished train at a node requests its next section.
6. Conflict detection per section (ascending section id). A request is
   *legal* if granting it respects capacity, direction, and node rules.
7. Decision. Requests are grouped per *resource*: the whole section for single
   track, each direction separately for double track. If all legal requests of
   a group fit together and HOLD is not allowed, they are granted
   automatically. Otherwise the simulation pauses and the agent is asked. Steps
   6–7 repeat in passes over the sections until a pass grants nothing. This only
   matters for finite station capacities, which the demo network does not use.
8. Granted trains enter their section (remaining time = travel time). Entry
   events are committed in ascending train id.
9. Waiting accounting: every ready, unfinished train that stands at a node and
   did not enter a section in this step accrues 1 train-minute.
10. Reward of the step, then the termination check (success, deadlock, truncation).

`reset()` runs steps 2–10 for t = 0 (no clock advance) and continues until the
first decision.

**Tie-breaking:** simultaneous events are processed in ascending train id.
Candidates of a decision are listed in ascending id (HOLD last). **A section
released in step 2 is available in step 8 of the same minute**, which is tested
with a hand-calculated example
(`test_sim_clock.py::test_section_released_in_phase_2_is_available_in_phase_8_of_same_step`).

**Occupancy rules:** `single`: at most `capacity` trains, all in the same
direction (capacity 1 = classic single track; capacity 2 = following moves
allowed). `double`: at most `capacity` trains per direction.

**When HOLD is allowed** (resource = section for single track, one direction for
double track): there is at least one legal requester and at least one
*competitor*. A competitor is a train that a grant now would delay:
(a) a requester of the same resource that is currently blocked by occupancy;
(b) a train on another section whose next section is this one and which
arrives in fewer than `travel_time(section)` minutes; or (c) a not-yet-departed
train starting with this section whose departure is fewer than
`travel_time(section)` minutes away. Because a competitor is always moving or
about to depart, holding is never pointless forever. Every HOLD costs at least
one train-minute.

**Definitions.** *Waiting* is described in step 9. Trains on a section and
trains before their departure never wait. *Arrival delay* per train is
`max(0, arrival − scheduled_arrival)` minutes, with
`scheduled_arrival = departure + pure running time + slack`, where slack = 2 min.
So a train's delay equals `max(0, its waiting − 2)`.

**Decision epochs vs simulation steps.** Most minutes need no decision. The
agent is queried only at decision points. One minute can contain several
decisions (for example, two sections in conflict). Many minutes contain none.

### 2.3 Termination, truncation, deadlock

* **Success:** all trains have arrived.
* **Truncation:** at `max_time = max departure + Σ running times of all trains`
  (for example 71 min for evaluation scenario 0). **Proof that it never triggers on the demo
  network:** after the last departure, in every minute with unfinished trains at
  least one train is traversing a section. If no train were on a section, every
  requested section would be empty, so every request would be legal. HOLD
  would not be allowed either, because a competitor must be moving or pending.
  So some train would enter. The makespan is therefore at most the last
  departure plus the total traversal minutes. On truncation each unfinished
  train would cost a penalty of `max_time` (documented, never observed).
* **Deadlock** (precise definition): unfinished trains exist and no train can
  ever progress again. The detector flags the fixed point "no train on a
  section, no train still to depart, no train entered in this step". Then the
  state can never change again (proof by induction over steps 2–8). On deadlock
  the episode terminates with the same penalty as truncation. With unlimited
  station capacity, which is the demo network, deadlock is impossible: if no
  train occupies a section, every request is legal. The detector is still
  implemented and tested on a hand-built mutual-blocking case with finite
  station capacities (`test_sim_occupancy.py`). Observed deadlocks: see §9.

Invalid actions cannot violate occupancy: the simulator rejects them
(`ValueError`), and the agent never proposes them (masking, §7).

### 2.4 Formal MDP

| Element | Definition in this code |
|---|---|
| State S | Clock t; per train status (pending / at node / on section / finished), route index, section, direction, remaining traversal minutes, waiting counter, delay; section occupancy; position inside the current step (which sections are still to be resolved, which trains already entered, which resources are held, pending decision). `RailwaySim.state_key()` is exactly the part that determines the future. |
| Observation | Graph tensors plus one feature row per legal action (§5). **Fully observable** with respect to future dynamics: every quantity that influences future transitions is either encoded directly or determined by the encoded features plus static routes. Caveat: the policy sees per-node and per-edge *aggregates* plus the candidates' own features, not every non-candidate train's full route, so the representation is a compression of S. The only randomness is the scenario. |
| Action A | At a decision: one of the candidate trains (ids, ascending) or HOLD if allowed. Fixed indexing for masking: index i < 5 = train i, index 5 = HOLD. |
| Transition P | **Deterministic**, given by the event order in §2.2. Randomness comes only from scenario generation and the policy's own sampling during training. |
| Reward R | `r_t = −(W_t − W_{t−1})` per simulation step (§8). |
| Discount γ | **γ = 1.** The episode is finite and the objective is total waiting, so the undiscounted return is exactly −W. γ < 1 would change the objective and is not used. |
| Horizon | Episode ends on success (always, on the demo network). Truncation at `max_time`; deadlock detection (§2.3). |
| Reward between decisions | `sim.apply(a)` returns the sum of the step rewards of all steps completed after decision k and up to decision k+1 (or the end). The step in which a decision is taken completes after the decision and is counted once. Rewards completed before the first decision are reported separately (`reward_before_first_decision`) and depend on no action. Tested: `test_reward.py::test_rewards_between_decisions_sum_to_episode_return_without_double_counting`. |

---

## 3. Input data and data flow

All data are **synthetic**. No GTFS, OSM, or other external data or services are needed.

| Field | Meaning | Example | Unit / data type | Used by |
|---|---|---|---|---|
| `node_names` | stations and junctions | `("W","J1","N","S","J2","E","H")` | tuple[str], static | simulator, graph |
| `Section.u, v` | endpoints of a physical section | s3: N–J2 | node ids (int), static | simulator, `edge_index` |
| `Section.travel_time` | running time over the section | 4 | minutes (int > 0), static | simulator, edge feature |
| `Section.capacity` | trains allowed (total for single, per direction for double) | 1 | int, static | simulator, edge feature |
| `Section.kind` | `"single"` (shared by both directions) or `"double"` | `"single"` | str, static | simulator, edge feature |
| `node_capacity` | station capacity (`None` = unlimited) | `None` | tuple / None, static | simulator (deadlock tests only) |
| `TrainSpec.route` | node sequence origin → destination (origin, destination and direction follow from it) | `(E, J2, N, J1, W)` | tuple[int], initial config | simulator, train features |
| `TrainSpec.departure` | minute the train becomes ready | 2 | minutes (int), initial config | simulator |
| `TrainSpec.slack` | schedule buffer | 2 | minutes (int) | scheduled arrival |
| scheduled arrival | `departure + running time + slack` | 18 | minutes, derived | delay, train feature |
| status / position | pending, at node n, on section s, finished | `("section", 3)` | dynamic state | observation |
| remaining time | minutes until the train leaves its section | 3 | int, dynamic | edge feature |
| waiting counter | train-minutes waited so far | 4 | int, dynamic | reward, features |
| scenario seed | determines routes and departures | 900 000 | int | `generate_scenario` |

Each of the 5 trains draws one of 10 route templates uniformly and an integer
departure time uniformly in [0, 8] min (held-out set: [0, 4]) from
`random.Random(seed)` (`src/railway_gnn_rl/scenarios.py`).

**Example input → simulation state** (evaluation scenario 0, from `outputs/scenarios/evaluation.json`):

```
train 0: W→J1→N→J2→E  dep 5   train 1: W→J1→N→J2→H  dep 0   train 2: H→J2→E  dep 8
train 3: E→J2→N→J1→W  dep 2   train 4: E→J2→H        dep 0
scheduled arrivals = [21, 19, 19, 18, 11] min, max_time = 71 min
```

`reset()` → t = 0. Trains 1 and 4 become ready. Train 1 enters s0 (double track)
automatically. Train 4 wants s5 (J2–E, double track, direction E→J2), but train
3 will depart from E in the same direction in 2 < 3 minutes. The first
decision is therefore `Decision(t=0, section=5, candidates=(4,), can_hold=True)`
with legal actions `[train 4, HOLD]`.

Data flow:

```
static infrastructure + scenario (seeded)          ← synthetic input
        │
        ▼
RailwaySim  (deterministic 10-phase step, pauses at decisions)     ← simulation logic
        │  build_observation(sim)
        ▼
DispatchGraph: x[7,8], edge_index[2,16], edge_attr[16,10],
               cand_feat[K,11], cand_node[K], cand_next_node[K], cand_edge[K]   ← model-ready tensors
        │
        ▼
GNN encoder → node embeddings h[7,32], graph embedding g[1,32]    ← neural network
        │
        ▼
policy head → one logit per legal action → masked softmax → action ← neural network + RL
        │  sim.apply(action) → reward until next decision
        └──────────────────────────────────────────────► next observation
```

---

## 4. Graph structure

```
                 N ─────── s3 (single, 4) ───────┐
               ╱ │                               │
     s1 (single,4) s6 (single, 3)               J2 ── s5 (double, 3) ── E
             ╱   │                             ╱  ╲
  W ── s0 ── J1  │            s4 (single×2, 5)     s7 (single, 6)
 (double, 3) ╲   │                         ╱          ╲
     s2 (double, 5)                       ╱            H
               ╲ │                       ╱
                 S ─────────────────────┘
  single   = single track, 1 train total (opposite directions exclude each other)
  single×2 = single track, up to 2 trains *in the same direction* (following moves)
  double   = double track, 1 train per direction
  numbers  = travel time in minutes
```

| Element | Representation | Meaning | Example |
|---|---|---|---|
| graph node | index 0..6 | a station or junction (W=0, J1=1, N=2, S=3, J2=4, E=5, H=6) | node 4 = J2 |
| directed edge 2s | `edge_index[:, 2s] = (u, v)` | section s traversed u→v | edge 6 = N→J2 |
| directed edge 2s+1 | `edge_index[:, 2s+1] = (v, u)` | section s traversed v→u | edge 7 = J2→N |
| bidirectional travel | two directed edges per section | both directions carry their own occupancy features | – |
| single-track constraint | `is_single = 1`, plus `occupancy_opposite_dir` on both directed edges | opposite traffic blocks the section | s3 |
| capacity | `capacity` feature (capacity/2) | trains admitted | s4: 1.0 (= 2 trains) |
| train positions | not graph nodes; they enter node/edge features (waiting, inbound, occupancy, remaining time, demand) and the per-action train features | – | – |

**The topology alone does not capture the state.** Train-specific information
is aggregated into node and edge features, and for the trains in the current
decision it is passed separately as `cand_feat`.

---

## 5. Node features, edge features, and tensor shapes

All features are `float32`, scaled by fixed constants with no data-dependent
normalisation: train counts are divided by n = 5, times by 30 min, and
travel/remaining times by 10 min (`src/railway_gnn_rl/graph_obs.py`).

**Node features `x` [7, 8]**

| # | name | calculation | unit / encoding | static? |
|---|---|---|---|---|
| 0 | is_terminal | node has exactly one section | 0/1 | static |
| 1 | is_junction | node has ≥ 3 sections | 0/1 | static |
| 2 | waiting_trains | ready trains standing at the node / 5 | fraction | dynamic |
| 3 | waiting_minutes | Σ waiting counters of those trains / 30 | min / 30 | dynamic |
| 4 | inbound_trains | trains on a section heading to the node / 5 | fraction | dynamic |
| 5 | destination_demand | unfinished trains whose destination is the node / 5 | fraction | dynamic |
| 6 | pending_departures | not-yet-departed trains starting here / 5 | fraction | dynamic |
| 7 | candidate_here | 1 if a candidate of the current decision waits here | 0/1 | dynamic |

**Edge features `edge_attr` [16, 10]** (per directed edge)

| # | name | calculation | unit / encoding | static? |
|---|---|---|---|---|
| 0 | travel_time | travel time / 10 | min / 10 | static |
| 1 | capacity | capacity / 2 | – | static |
| 2 | is_single | single track | 0/1 | static |
| 3 | occupancy_same_dir | trains on the section in this direction / capacity | fraction | dynamic |
| 4 | occupancy_opposite_dir | trains in the opposite direction / capacity | fraction | dynamic |
| 5 | min_remaining_time | smallest remaining traversal time on the section / 10 | min / 10 | dynamic |
| 6 | future_demand | unfinished trains that still have to enter this directed edge / 5 | fraction | dynamic |
| 7 | requests | trains waiting at the tail node for this edge / 5 | fraction | dynamic |
| 8 | approaching | trains on a section whose next directed edge is this one / 5 | fraction | dynamic |
| 9 | contested | the section of the current decision (both directions) | 0/1 | dynamic |

**Per-action features `cand_feat` [K, 11]**, one row per legal action. K =
number of candidate trains + 1 if HOLD is allowed.

| # | name | for a train candidate | for HOLD |
|---|---|---|---|
| 0 | is_hold | 0 | 1 |
| 1 | direction | +1 / −1 on the contested section | 0 |
| 2 | route_progress | sections completed / sections on route | 0 |
| 3 | remaining_run_time | pure running time still ahead / 30 | 0 |
| 4 | waiting | waiting so far / 30 | 0 |
| 5 | projected_lateness | (t + remaining running time − scheduled arrival) / 30 | 0 |
| 6 | remaining_sections | sections still ahead / 5 | 0 |
| 7 | time_since_departure | (t − departure) / 30 | 0 |
| 8 | contested_travel_time | travel time of the contested section / 10 | same |
| 9 | n_competitors | competitors (§2.2) / 5 | same |
| 10 | min_competitor_eta | minutes until the first competitor requests / 10 | same |

Index tensors (`int64`, length K): `cand_node` = the node where the train waits
(for HOLD: endpoint u of the section), `cand_next_node` = the node it would
travel to (HOLD: endpoint v), `cand_edge` = the directed edge it would enter
(HOLD: edge 2s).

**Concrete example** (evaluation scenario 0, first decision, measured):

| tensor | shape | dtype |
|---|---|---|
| `x` | (7, 8) | float32 |
| `edge_index` | (2, 16) | int64 |
| `edge_attr` | (16, 10) | float32 |
| `cand_feat` | (2, 11) — train 4 and HOLD | float32 |
| `cand_node`, `cand_next_node`, `cand_edge` | (2,) each | int64 |
| GNN node embeddings `h` | (7, 32) | float32 |
| graph embedding `g` | (1, 32) | float32 |
| action logits | (2,) | float32 |

Row for E (node 5) in `x`: `[1, 0, 0.2, 0, 0, 0.4, 0.2, 1]` (terminal, 1 train
waiting, 2 trains heading to E, 1 pending departure, a candidate waits here).
Row of train 4: `[0, −1, 0, 0.3, 0, −0.067, 0.4, 0, 0.3, 0.2, 0.2]`.

The railway graph is distinct from the network's hidden units: the 7 graph
nodes and 16 directed edges are railway objects, while the 32-dimensional
embeddings and all weight matrices are learned parameters.

---

## 6. GNN architecture

**Message passing in plain language.** Each station starts with a description
of itself (8 numbers). In one *layer*, every station collects messages from its
neighbours. A message combines the neighbour's current description with the
features of the connecting track, for example "occupied in the opposite
direction, 3 minutes left". The station then updates its own description from
those messages. After two layers a station "knows" about everything within two
sections, for example a train approaching the junction behind the next
single-track section.

**Implementation (`src/railway_gnn_rl/models.py`, `GNNEncoder`):**

* input projection `h⁰ = ReLU(W_in x)`, 8 → 32;
* L = 2 layers of `GINEConv` (PyTorch Geometric) with residual connections:
  `h ← h + ReLU(GINEConv(h, edge_index, edge_attr))`, where
  `GINEConv: h_v' = MLP( h_v + Σ_{u→v} ReLU(h_u + W_e e_uv) )`, MLP = Linear(32,32)-ReLU-Linear(32,32), aggregation = sum;
* graph embedding `g` = mean of the node embeddings.
* Learned parameters: `W_in`, per layer the edge projection `W_e` (10 → 32) and the
  MLP. Encoder total: 5 216 parameters. Whole policy: 10 081.

```
railway state ─► graph tensors (x, edge_index, edge_attr) ─► GINEConv ×2 ─► h [7,32], g [1,32]
                                                                         └► policy head (§7)
```

How embeddings become action scores, for each legal action k:

```
z_k = [ h[cand_node_k] | h[cand_next_node_k] | edge_attr[cand_edge_k] | cand_feat_k | g ]   (117 numbers)
logit_k = MLP(z_k)     MLP = Linear(117,32)-ReLU-Linear(32,32)-ReLU-Linear(32,1)
```

The logits enter a softmax, giving π(a | s), and the policy-gradient loss
(§7) is backpropagated through the head into `h` and `g`, and so into every
GNN weight. The tests confirm that all GNN parameters receive non-zero
gradients, that a change in remote graph state changes the candidate's
embedding (two hops away) and the action scores, and that batched and single
evaluation agree.

---

## 7. RL agent architecture and the MLP ablation

```
 Observation ──► GNN encoder ──► policy head ──► masked softmax ──► legal action ──► RailwaySim ──► reward + next observation
 [simulation]    [neural net]    [neural net]    [RL: masking]      [RL: sample /     [simulation      │
                                                                     argmax]          logic]           │
      ▲                                                                                               │
      └──────────────── REINFORCE update once per training episode (Adam) ◄───────────────────────────┘
                        [RL algorithm]
```

* **Algorithm:** REINFORCE (Monte-Carlo policy gradient) with a *self-critical*
  baseline. For each training episode: (1) sample an episode from π_θ, giving
  return G; (2) run the greedy (argmax) policy on the same scenario without
  gradient, giving G_base; (3) loss
  `−((G − G_base)/10) · Σ_k log π_θ(a_k|s_k) − 0.01 · Σ_k H(π_θ(·|s_k))`;
  (4) clip the gradient to norm 1, then take one Adam step (lr = 1e-3). The
  baseline does not depend on the sampled actions, so the gradient stays
  unbiased.
* **Why REINFORCE:** with γ = 1 it optimises exactly −W. Episodes contain only a
  few decisions (≈ 4), so full Monte-Carlo returns are cheap and no value
  function needs to be learned.
* **Legal actions and masking:** the policy scores only the legal actions. In
  the fixed 6-slot action space (5 trains + HOLD) every illegal slot receives
  logit −∞, so its probability is exactly 0. Tests check the mask, the
  zero-probability property, and that 200 samples are always legal.
* **Exploration:** sampling from the softmax during training, plus the entropy
  bonus. Evaluation uses argmax (deterministic).
* **No training on decisions without effect:** single legal requests without
  HOLD are granted automatically. An episode with zero decisions produces no
  update (tested).
* **Counting:** *environment step* = 1 simulated minute; *decision step* = one
  agent query; *episode* = one scenario from t = 0 to the end; *gradient update*
  = one Adam step per training episode with ≥ 1 decision; *evaluation run* =
  one greedy episode on a fixed scenario. Each training episode also runs one
  greedy baseline episode, which is not counted as a training episode.
* **Hyperparameters (main configuration, fixed before evaluation):** hidden 32,
  2 GNN layers, lr 1e-3, entropy 0.01, advantage scale 10, gradient clip 1.0,
  γ = 1, 500 training episodes per seed, validation every 50 episodes.

**MLP ablation (`MLPEncoder`).** Same inputs, same head, same RL algorithm, same
budget. The only difference is the absence of message passing:
`h = node-wise MLP(x)` uses each station's own features only, at the same depth,
and `g = MLP(flatten(x) ‖ flatten(edge_attr))` receives the complete global
state as one vector, without relational structure. Policy parameters: 11 281
(MLP) vs 10 081 (GNN), ratio 1.12. A test proves that the MLP's node embedding
does not react to a remote node's state while the GNN's does.

---

## 8. Reward function

Per simulation step: **`r_t = −(W_t − W_{t−1})`**, where `W_t` is the accumulated
waiting time (train-minutes) after step t. A step in which k trains wait gives
`r_t = −k`. The reward is never positive. It is 0 when nobody waits.

**Worked example** (`test_reward.py`): two trains depart A at 0 and want the
same single track (2 min). Grant it to T1. T0 waits at t = 0 and t = 1:
step rewards `[−1, −1, 0, 0, 0]`, return `−2 = −W`.

**Return vs waiting time:** with γ = 1 the undiscounted episode return is
`Σ_t r_t = −W_T`. Higher return (closer to 0) means less waiting. This holds
only under this reward convention. Optional terminal terms, both reported
separately from W:

* truncation or deadlock: `−max_time` per unfinished train (never occurred);
* sensitivity variant "wait+delay": `−1 · Σ delay` at the arrival steps, so the
  return becomes `−W − Σ delay` for that variant only.

Reward improvement alone does not prove better railway performance. Training
returns come from *sampled* actions on *changing* scenarios, so the operational
claim is made only from the fixed evaluation scenarios with paired statistics
(§9).

---

## 9. Baselines, exact optimum, statistical method, and results

### 9.1 Dispatchers

| dispatcher | decision rule |
|---|---|
| FCFS | the candidate that reached its current node (or became ready) earliest; ties → lowest id; never HOLD |
| fixed priority | lowest train id; never HOLD |
| longest waiting | largest accumulated waiting counter; ties → lowest id; never HOLD |
| random | uniformly random legal action, HOLD included (seed 0); a sanity lower bound |
| exact optimum | see 9.2 |
| GNN-RL | trained policy, greedy (argmax) at evaluation |
| MLP-RL (ablation) | same, with the MLP encoder (§7) |

### 9.2 Exact optimum (`src/railway_gnn_rl/optimum.py`)

The simulator is deterministic and every episode ends by `max_time`, so each
scenario defines a finite decision tree. The solver runs a depth-first search
over **all** legal actions (candidates and HOLD) using the same simulator, and
memoises the best future return per Markov state:
`V(s) = max_a [ r(s,a) + V(s') ]`, with `V(terminal) = 0`. Complexity:
O(number of distinct decision states × branching factor) simulator calls.
Memoisation merges different action orders that reach the same state. If more
than 10⁶ states would be expanded, the scenario is reported as
"not_computed" instead of guessed. Ties go to the first action in id order. The
optimal action sequence is stored (`optimum.csv`) and replayed through the
simulator to confirm the value (test `test_optimum.py`). The solver also
matches a plain brute force without memoisation on 25 tiny scenarios.
**Measured:** optimum computed for 50/50 evaluation and 20/20 held-out
scenarios; at most 11 726 states per scenario; 11.9 s wall time for all 70.

### 9.3 Scenario sets and seeds

| set | seeds | size | departures | purpose |
|---|---|---|---|---|
| training | 1 000 000 + 100 000·seed + episode | 500 per seed | [0, 8] min | training only |
| validation | 800 000–800 019 | 20 | [0, 8] | learning-curve monitoring, convergence check, sensitivity study |
| evaluation | 900 000–900 049 | 50 | [0, 8] | final comparison only (saved to `outputs/scenarios/evaluation.json`) |
| held-out | 700 000–700 019 | 20 | [0, 4] (denser) | generalisation probe |

Training seeds: 0–4 for the GNN and for the MLP. Sensitivity study: seeds 0–2.
The main configuration (§7) was fixed before any evaluation, and no
hyperparameter was selected using the evaluation set.

### 9.4 Statistical method

* **Over scenarios** (one dispatcher, n = 50): mean, SD, and a 95 % percentile
  bootstrap CI (10 000 resamples, seed 0). For the learned policies, each
  scenario's value is first averaged over the 5 training seeds.
* **Over training seeds** (n = 5): 95 % Student-t interval of the 5 per-seed means.
* **Paired comparisons** (same scenarios for every dispatcher): per-scenario
  difference `W_A − W_B`; 95 % paired percentile bootstrap CI of the mean
  difference; two-sided Wilcoxon signed-rank test (zero differences dropped);
  effect size Cohen's `d_z = mean(d)/SD(d)`.
* **Optimality gap:** `(mean W_policy − mean W_opt) / mean W_opt`, a ratio of
  means over the scenarios where the optimum was computed (all of them here).
* **Convergence criterion** (fixed in advance): the last 3 validation means
  (every 50 episodes) lie within a band of width 5 % of their mean.

### 9.5 Results (measured; full tables in `outputs/summary.md`)

**Evaluation set, 50 scenarios** (W = total waiting per scenario, train-minutes)

| dispatcher | mean W | 95 % CI | gap vs optimum | difference vs FCFS [95 % CI] | Wilcoxon p | d_z | mean delay (min) |
|---|---|---|---|---|---|---|---|
| FCFS | 13.84 | [11.84, 15.98] | 12.2 % | – | – | – | 8.04 |
| fixed priority | 13.78 | [11.84, 15.88] | 11.7 % | −0.06 [−0.40, 0.26] | 0.76 | −0.05 | 8.38 |
| longest waiting | 13.88 | [11.86, 16.04] | 12.5 % | +0.04 [−0.12, 0.24] | 0.65 | 0.06 | 8.02 |
| random | 18.80 | [16.14, 21.58] | 52.4 % | +4.96 [3.48, 6.48] | < 0.0001 | 0.91 | 12.22 |
| MLP-RL (5 seeds) | 13.76 | [11.75, 15.93] | 11.5 % | −0.08 [−0.56, 0.41] | 0.83 | −0.04 | 8.35 |
| **GNN-RL (5 seeds)** | **13.71** | [11.70, 15.88] | **11.1 %** | −0.13 [−0.63, 0.36] | 0.75 | −0.07 | 8.34 |
| exact optimum | 12.34 | [10.52, 14.32] | 0 % | −1.50 [−2.00, −1.02] | < 0.0001 | −0.82 | – |

Seed level (mean of per-seed means, 95 % t-interval, n = 5): GNN 13.71
[13.49, 13.93] (per seed 13.46, 13.80, 13.68, 13.94, 13.66); MLP 13.76
[13.58, 13.95] (per seed 13.72, 13.94, 13.84, 13.54, 13.78). Per-seed
optimality gaps: GNN 9.1–13.0 %, MLP 9.7–13.0 %.

**Ablation GNN − MLP (evaluation set):** paired per-scenario difference −0.06
[−0.22, 0.10] train-min, Wilcoxon p = 0.40, d_z = −0.09; seed level −0.06
[−0.38, 0.27].

**Held-out set (20 denser scenarios):** FCFS 18.75, fixed priority 18.20,
longest waiting 18.95, random 22.75, MLP 18.06, GNN 17.96, optimum 16.05
(gaps: 16.8 / 13.4 / 18.1 / 41.7 / 12.5 / 11.9 / 0 %). GNN − FCFS = −0.79
[−1.62, −0.10] (bootstrap), but Wilcoxon p = 0.063 with only 7 non-zero
differences. GNN − MLP = −0.10 [−0.32, 0.08], p = 0.29.

**Deadlocks / truncations:** 0 / 0 in all 980 evaluation episodes (every
dispatcher and seed, both sets), in all 5 000 logged main training episodes
(all terminated with "success"), and in 200 random-policy V&V runs. The
training episodes of the sensitivity runs are not logged individually. **Capacity violations: 0** (checked by the simulator's rules and by
`test_vv.py`).

**Reproducibility:** re-evaluating the 10 saved checkpoints reproduced the mean
total waiting **exactly** (max |difference| = 0.0, tolerance 1e-9). The three
"base" sensitivity runs, trained in separate processes, are bit-identical to
the main GNN seeds 0–2.

**Training behaviour and convergence.** The convergence criterion was met by
all 10 main runs. That is a statement about *stability*, not quality: the
greedy validation mean was nearly flat from the first check onwards (GNN
≈ 16.6–17.1 train-min, apart from short excursions such as seed 4 at
episode 100: 22.2; MLP seed 1 started at 36.2). What clearly changed during
training is the **sampled** behaviour:

| episodes | GNN: HOLD / ep. | decisions / ep. | policy entropy | sampled W | MLP: HOLD / ep. | sampled W |
|---|---|---|---|---|---|---|
| 0–49 | 2.02 | 6.22 | 0.745 | 20.36 | 2.30 | 21.38 |
| 200–249 | 1.14 | 5.49 | 0.609 | 17.63 | 1.20 | 17.84 |
| 450–499 | 0.10 | 4.36 | 0.313 | 14.73 | 0.24 | 15.20 |

**Interpretation (only what the statistics support).**

1. On the evaluation set, GNN-RL, MLP-RL and the three rules are **statistically
   indistinguishable** (all paired CIs contain 0; p ≥ 0.65; |d_z| ≤ 0.07). All
   are 11–12.5 % above the exact optimum. The optimum is significantly better
   than every other dispatcher (p < 0.0001).
2. **The ablation shows no measurable benefit of message passing** in this
   instance. The claim "the GNN helps" is therefore **not supported** here.
3. Both learned policies use **HOLD 0 times** at evaluation. In effect they
   learned "never hold, and pick a reasonable order". The optimum holds in
   11/50 scenarios (17 of 198 actions) and beats FCFS in 25/50 scenarios
   (11 of those involve a hold). The remaining gap is exactly the
   non-myopic decisions the policies did not find.
4. The rising training return (Figure A) mostly reflects reduced random
   exploration, not a better greedy policy. This is the case where reward
   improvement does not mean better railway performance.
5. The held-out set hints at a small advantage of the learned policies over
   FCFS (bootstrap CI excludes 0) but not over fixed priority, and the Wilcoxon
   test is not significant. With 20 scenarios this is **inconclusive**.

**Likely causes and what would be tried next** (not done here, to avoid tuning
on the evaluation set): HOLD helps only in specific states, and a sampled HOLD
usually costs waiting, so the self-critical REINFORCE signal pushes HOLD
probability down early (entropy 0.75 → 0.31). The signal per decision is weak:
one advantage per episode is shared by all ≈ 4–6 decisions. Options:
per-decision credit assignment (for example a learned value baseline or
actor-critic), imitation of the exact optimum's actions as a warm start, more
episodes, or entropy scheduling. Note also that `lr = 3e-3` made one of three
seeds collapse (see §9.6).

### 9.6 Sensitivity study (3 seeds per setting; validation set used for interpretation)

| setting | validation mean W [95 % t-CI] | evaluation mean W [95 % t-CI] | eval. mean delay |
|---|---|---|---|
| base (L=2, H=32, lr=1e-3, wait reward) | 16.85 [16.63, 17.07] | 13.65 [13.22, 14.08] | 8.31 |
| GNN layers = 1 | 16.80 [16.55, 17.05] | 13.83 [13.69, 13.98] | 8.43 |
| GNN layers = 3 | 16.83 [16.76, 16.91] | 13.83 [13.45, 14.20] | 8.35 |
| hidden = 64 | 16.70 [16.27, 17.13] | 13.73 [13.35, 14.10] | 8.34 |
| lr = 3e-3 | 18.47 [10.86, 26.08] | 16.21 [5.29, 27.12] | 10.65 |
| reward = wait + delay (λ = 1) | 16.83 [16.69, 16.98] | 13.89 [13.70, 14.07] | 8.51 |

Depth (1/2/3 layers), width (32/64), and the delay-penalty reward variant make
no discernible difference. With lr = 3e-3, seed 1 collapsed to a poor policy
(validation 22.0, evaluation 21.28), while seeds 0 and 2 were normal. That run
still satisfied the convergence criterion, so the criterion detects
stagnation, not success. Per-run values: `outputs/sensitivity_runs.csv`.

### 9.7 Runtime (measured, 18-core CPU, 10 worker processes)

Full pipeline `railway-demo all`: **2 min 42 s** wall clock (training 10 runs:
47 s; optimum: 12 s; evaluation total: 23 s; sensitivity 18 runs: 84 s; figures:
2 s). One 500-episode training run takes ≈ 35–43 s on one core. The full
training budget of the specification was used unchanged (500 episodes, 5 seeds).

---

## 10. Evaluation figures: how to read them

General rules: **lower waiting time is better. Higher return is better only
under this reward convention (return = −waiting).** Single episodes fluctuate
strongly because every training episode is a *different* random scenario, so
the return does not improve monotonically. The immediate reward (one minute) is
not the cumulative return (one episode), and neither is the operational
evaluation metric (fixed evaluation scenarios, greedy policy). Error bars are
95 % CIs computed by the method stated in each figure. Figures are in
`outputs/figures/`.

**A. Training episode return** (`A_return_curve.png`), *training data*.
x: training episode (0–499), one point per sampled training episode. y: episode
return = −total waiting (train-minutes), unitless sign convention.
Faint dots: raw values of all 5 seeds. Lines: mean over seeds of the per-seed
trailing 25-episode moving average. Band: ±1 SD across seeds of those moving
averages. *Reading:* the return rises from about −21 to −15 over 500
episodes. GNN and MLP lie on top of each other. *Caveat:* sampled actions plus
changing scenarios; the rise mainly reflects less random exploration (§9.5).

**B. Training total waiting** (`B_waiting_curve.png`), *training data with
evaluation references*. x: training episode; y: total waiting per episode
(train-minutes), aggregated like A. The horizontal lines are **evaluation-set**
means (FCFS 13.84, exact optimum 12.34). They are not training values, and the
training scenarios are different draws from the same distribution. *Reading:*
the smoothed sampled waiting falls from ≈ 21 to ≈ 15 but stays above both
reference levels, because sampling still explores.

**C1. Immediate reward per step** (`C_step_reward.png`) and **C2. cumulative
return** (`C_cumulative_return.png`), *evaluation scenario 0, GNN seed 0,
greedy*. x: simulation time t (minutes). y (C1): `r_t = −(W_t − W_{t−1})`, i.e.
minus the number of trains waiting in that minute. y (C2): running sum of C1
(train-minutes). One bar or step = one simulated minute; raw values with no
smoothing. *Reading:* waiting happens in bursts around conflicts (here t = 2
and t = 7–14). The final value of C2 is −W for this episode.

**D. Baseline comparison** (`D_comparison.png`), *evaluation set*. x:
dispatcher; y: mean total waiting per scenario (train-minutes) over the 50
scenarios (learned policies: per-scenario mean over 5 seeds). Error bars:
95 % percentile bootstrap CI over scenarios. *Reading:* all bars except random
and the optimum are almost equal. *Caveat:* these CIs are wide because
scenarios differ greatly in difficulty (SD ≈ 7.5). They are **not** the right
tool for comparing dispatchers; use the paired statistics (§9.5, Figure E).

**E. Paired differences** (`E_paired_difference.png`), *evaluation set*. x:
per-scenario difference W(GNN, mean over seeds) − W(FCFS) in train-minutes;
negative = GNN waits less. y: number of scenarios (bin width 1 min). One
observation = one evaluation scenario. Dashed line at 0, solid line at the mean
(−0.13). *Reading:* in 28 of 50 scenarios the seed-averaged difference lies within ±0.5 train-min. Gains and
losses of a few minutes roughly cancel out. Paired 95 % CI [−0.63, 0.36],
Wilcoxon p = 0.75.

**F. Sensitivity study** (`F_sensitivity.png`), *validation set*. x: mean total
waiting on the 20 validation scenarios (train-minutes); y: setting. Dot = mean
over 3 training seeds; bar = 95 % t-interval over the seeds (n = 3, hence
wide). *Reading:* all settings overlap except lr = 3e-3, whose huge interval
comes from one collapsed seed.

**Summary: measured vs expected**

| statement | status |
|---|---|
| training return increases | measured (Figure A), mainly exploration reduction |
| learned greedy policy better than FCFS on evaluation set | **not supported** (p = 0.75) |
| GNN better than MLP (effect of message passing) | **not supported** (p = 0.40) |
| exact optimum better than all heuristics and learned policies | measured, p < 0.0001 |
| learned policies far better than random | measured (+5.1 train-min for random vs GNN) |
| zero deadlocks / truncations / capacity violations | measured |
| exact reproducibility of evaluation from checkpoints | measured (difference 0.0) |

---

## 11. Quick start and reproducibility

Requirements: Linux/WSL or macOS, [uv](https://docs.astral.sh/uv/), CPU only.
PyTorch is installed from the CPU wheel index (see `pyproject.toml`).

```bash
uv sync                                   # install dependencies (Python ≥3.11; uv picked 3.13 here)
uv run pytest -q                          # all unit, V&V and integration tests (~40 s)
uv run pytest -q tests/test_vv.py         # simulator verification & validation only
uv run pytest -q -m integration           # end-to-end tests only

uv run railway-demo all                   # full experiment: train + evaluate + sensitivity + figures
# or step by step (all write to outputs/):
uv run railway-demo train       --seeds 0 1 2 3 4 --episodes 500   # GNN and MLP (ablation), 5 seeds each
uv run railway-demo evaluate                                       # optimum, baselines, checkpoints, summary
uv run railway-demo sensitivity --sens-seeds 0 1 2
uv run railway-demo figures
```

`--workers N` sets the number of parallel processes (default 8). Each process
uses one torch thread.

**Outputs (`outputs/`)**

| path | content |
|---|---|
| `config.json` | every hyperparameter, scenario-set seeds, library versions, git commit, wall-clock times, reproducibility check |
| `scenarios/{validation,evaluation,heldout}.json` | the fixed scenario sets, as evaluated |
| `checkpoints/{gnn,mlp}_seed{s}.pt` | trained weights + their configuration |
| `train_metrics.csv` | one row per training episode per kind and seed: return, waiting, delay, decisions, holds, steps, termination, loss, entropy, baseline return, advantage, gradient norm |
| `validation_curve.csv` | validation mean waiting every 50 episodes |
| `train_runs.json` | per run: wall time, validation curve, convergence flag, evaluation mean right after training |
| `optimum.csv` | exact optimum per evaluation / held-out scenario: status, W, states expanded, optimal action sequence |
| `eval_per_scenario.csv` | every dispatcher (and seed) on every scenario |
| `summary.json`, `summary.md` | the comparison tables of §9 |
| `sensitivity_runs.csv`, `sensitivity.csv` | sensitivity study, per run and aggregated |
| `step_trace.csv` | step rewards for Figure C |
| `figures/*.png` | Figures A–F |
| `run.log` | console log of the full run, including `time` |

**Reproducing the numbers.** Seeds: Python `random`, NumPy, and PyTorch are
seeded with the training seed. Scenarios come from fixed seeds (§9). Torch
runs with one thread per process, and all model operations used here are
deterministic on CPU. `railway-demo evaluate` reloads every checkpoint and
re-evaluates it. The maximum absolute difference to the evaluation performed
right after training is stored in `config.json → reproducibility` (tolerance
1e-9). Rerunning `railway-demo all` on the same library versions reproduces the
training curves bit-for-bit, because the training run is deterministic per seed
(tested in `test_experiment.py`). On different CPUs or library versions,
floating-point summation order may differ. That has not been quantified here.

---

## 12. Limitations and future extensions

* **Infrastructure and movement:** 7 nodes, 8 sections, fixed integer travel
  times, no acceleration or braking, no block sections or signals, no
  interlocking. Stations have unlimited capacity, so deadlock is excluded by
  construction rather than handled. Real stations have limited tracks.
* **Dispatching actions:** only the order of entry into a contested section and
  HOLD. No re-routing, no platform assignment, no speed control, no
  cancellations. HOLD is offered only under the competitor rule of §2.2.
* **Safety and signalling:** not modelled. The simulator guarantees capacity and
  direction rules, nothing else.
* **Synthetic scenarios:** 5 trains, uniform random routes from 10 templates and
  uniform departures in an 8-minute window. No real timetable, no stochastic
  disturbances (delays are caused only by conflicts), no passenger weighting.
* **Training stability:** REINFORCE with an episode-level baseline. One of three
  seeds diverged at lr = 3e-3. The learned policies converged (by the stated
  criterion) to "never HOLD" and did not reach the optimum.
* **Baselines:** three simple rules. Stronger conventional methods, such as
  MILP/CP scheduling or rolling-horizon optimisation, are not included. On
  this instance the exhaustive optimum is the strong benchmark.
* **Exact optimum:** valid only for this instance size and these rules. The
  state space grows exponentially with the number of trains and decisions, so
  the method does not scale to realistic networks.
* **Generalisation:** tested only on the same network with a shifted departure
  window (20 scenarios, inconclusive). Not tested: other topologies, more
  trains, or longer horizons. Generalisation across topologies is the main
  argument for GNNs, and this demo does not exercise it.
* **Determinism:** exact on this machine and library versions. Other CPUs or
  versions may change floating-point summation order, which was not quantified.
* **What this is:** a successful *educational demonstration* of how graph-structured
  railway state can be fed through a GNN into an RL policy, with an honest,
  verified evaluation pipeline. It is **not** evidence of a useful real-world
  dispatching method. In this instance the learned policies match simple rules
  and stay ≈ 11 % above the optimum.

Possible extensions: per-decision credit assignment (actor-critic), imitation
of the exact solver, finite station capacities (making deadlock avoidance a
learning problem), variable topologies for generalisation tests, stochastic
running-time disturbances, and comparison with a MILP rolling-horizon dispatcher.

---

## 13. Development process and deviations from the specification

**Test-driven development.** Each component was built as RED → GREEN →
REFACTOR. The test was written first, run and seen failing for missing
behaviour (an API skeleton raising `NotImplementedError`, never an import
error), then implemented. The git history has one commit per cycle group.
Cycles: clock and event order; state transitions; occupancy, node capacity
and deadlock; HOLD action; reward and termination; scenarios and graph
observation; GNN/MLP forward pass; masking, action selection and training
update; baselines, evaluation and exact optimum; statistics; experiment
pipeline; figures; CLI and integration. The V&V tests (`tests/test_vv.py`)
verify behaviour that already existed and were not written as RED tests. Test
suite: **95 tests, all passing, ≈ 40 s.**

**Deviations from the original task specification (all intentional and documented):**

1. **HOLD action (user decision).** Measurements showed that with decisions only
   for *simultaneous* conflicts there were ≈ 1.1–1.3 decisions per episode and
   the optimum was only ≈ 3–4 % better than the simplest rule. After
   consultation, a HOLD action was added (§2.2). Consequence: step 7 also
   exposes a decision when exactly one train is legal but a competitor would be
   delayed.
2. **Grouping by resource** (direction groups on double track) and **repeated
   passes of steps 6–7** were added for unambiguous decisions. The passes only
   matter with finite station capacities.
3. **Departure window [0, 8] min** (held-out: [0, 4]), chosen during scenario
   design (before any training) because wider windows produced too few
   conflicts. The decision is visible in `tests/test_scenarios.py`.
4. **Deadlock option (a)** was chosen for the demo network (provably
   deadlock-free). The detector and the penalty (option b) are implemented and
   tested with finite station capacities.
5. Two existing tests were updated when the HOLD rule changed the decision
   sequence of their scenarios. Their hand-calculated times were unchanged.

Before finalising, the README was checked as if by a numerical-simulation reader:
Δt, units, event order, tie-breaking, MDP, definitions of every metric and
figure axis, and agreement with the code (feature names and shapes are taken
from `graph_obs.py`; all numbers come from `outputs/`).
