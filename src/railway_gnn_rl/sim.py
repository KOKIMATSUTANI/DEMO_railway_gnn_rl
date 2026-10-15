"""Discrete-time railway dispatching simulator.

Time step dt = 1 minute; all times are integer minutes.  One simulation step at
(new) time t runs these phases in this exact order:

  1. advance clock t -> t+1
  2. trains completing a section release it and arrive at the next node
  3. trains at their destination are marked finished
  4. trains whose departure time equals t become ready (at their origin)
  5. ready, unfinished trains at a node request their next section
  6. conflict detection per section (ascending section id): a request is
     *legal* if granting it respects capacity, direction and node rules
  7. decision: requests are grouped per resource (the whole section for single
     track, each direction for double track).  If all legal requests of a group
     can be granted together they are granted automatically; otherwise the
     simulator pauses and exposes a `Decision` (the contested section and the
     legal candidate trains of that group).
     Phases 6-7 repeat in passes over all sections until a pass grants
     nothing (only matters for finite node capacities, see below).
  8. granted trains enter the section (remaining time = travel time); the
     entry events are committed in ascending train id
  9. waiting-time accounting: every ready, unfinished train at a node that did
     not enter a section in this step accrues 1 train-minute
 10. reward bookkeeping and termination (success / truncation / deadlock)

Within a phase, trains are processed in ascending train id.  A section
released in phase 2 is available for entry in phase 8 of the same step.
`reset()` runs phases 2-10 for t = 0 (no clock advance) and then continues
until the first decision or the end of the episode.

Occupancy rules
---------------
* kind "single": at most `capacity` trains in total, all in the same direction
  (capacity 1 = classic single track; capacity 2 = following moves allowed).
* kind "double": at most `capacity` trains per direction, directions independent.
* node capacity (optional, None = unlimited): trains at the node plus trains on
  a section heading to it.  Entry into a section therefore reserves a slot at
  the target node.  A train leaves its node when it enters a section; a train
  at its destination leaves the network (finished).

Deadlock
--------
A deadlock is a state in which some trains are unfinished and no train can
ever progress again.  The detector (phase 10) flags the *fixed point*: there
are unfinished trains, no train is traversing a section, no train is waiting
for a future departure, and no train entered a section in this step.  Then
phases 2-4 of every later step change nothing, the same requests remain
illegal, and by induction the state never changes again.  (A partial deadlock
is reported once all other trains have finished.)  With unlimited node
capacities a deadlock is impossible: if no train occupies any section, every
request is legal (all sections empty), so some train enters.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Section:
    id: int
    u: int
    v: int
    travel_time: int  # minutes, positive integer
    capacity: int
    kind: str  # "single" or "double"


@dataclass(frozen=True)
class Infrastructure:
    node_names: tuple[str, ...]
    sections: tuple[Section, ...]
    node_capacity: tuple[int | None, ...] | None = None

    def section_between(self, a: int, b: int) -> Section:
        for s in self.sections:
            if (s.u, s.v) in ((a, b), (b, a)):
                return s
        raise KeyError(f"no section between nodes {a} and {b}")


@dataclass(frozen=True)
class TrainSpec:
    id: int
    route: tuple[int, ...]  # node sequence origin ... destination
    departure: int          # minute at which the train becomes ready
    slack: int = 2          # schedule buffer (minutes) on top of pure running time


@dataclass(frozen=True)
class Scenario:
    infra: Infrastructure
    trains: tuple[TrainSpec, ...]
    max_time: int | None = None  # None -> provable completion bound (see RailwaySim)


@dataclass(frozen=True)
class Decision:
    t: int
    section: int
    candidates: tuple[int, ...]  # legal competing train ids, ascending


@dataclass(frozen=True)
class Event:
    t: int
    phase: int
    train: int
    kind: str  # release | finish | ready | enter
    section: int | None = None


@dataclass
class TrainState:
    status: str = "pending"  # pending | at_node | on_section | finished
    route_idx: int = 0       # index into route of the current / last node
    section: int | None = None
    direction: int = 0       # +1 if traversing u->v, -1 if v->u, 0 if not on a section
    remaining: int = 0       # remaining traversal minutes
    waiting: int = 0         # accumulated waiting train-minutes
    arrival_time: int | None = None


@dataclass(frozen=True)
class TrainSnapshot:
    status: str
    position: tuple[str, int] | None  # ("node", n) | ("section", s) | None
    route_idx: int
    remaining: int
    waiting: int
    direction: int


@dataclass(frozen=True)
class StepSnapshot:
    """State at the end of the step at time t (after phase 10)."""
    t: int
    trains: tuple[TrainSnapshot, ...]
    occupants: tuple[tuple[int, ...], ...]
    total_waiting: int


def running_time(infra: Infrastructure, route: tuple[int, ...]) -> int:
    return sum(infra.section_between(a, b).travel_time for a, b in zip(route, route[1:]))


class RailwaySim:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.infra = scenario.infra
        for i, tr in enumerate(scenario.trains):
            assert tr.id == i, "train ids must be 0..n-1 in order"
            assert len(tr.route) >= 2
        self.run_times = [running_time(self.infra, tr.route) for tr in scenario.trains]
        self.scheduled_arrival = [tr.departure + r + tr.slack
                                  for tr, r in zip(scenario.trains, self.run_times)]
        # Completion bound for unlimited node capacity: after the last departure at
        # least one train is traversing in every minute until all have finished,
        # so the makespan is <= max departure + sum of all running times.
        self.max_time = scenario.max_time if scenario.max_time is not None else (
            max(tr.departure for tr in scenario.trains) + sum(self.run_times))

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        self.t = 0
        self.trains = [TrainState() for _ in self.scenario.trains]
        self.occupants: list[list[int]] = [[] for _ in self.infra.sections]
        self.total_waiting = 0
        self.decision: Decision | None = None
        self.done = False
        self.termination: str | None = None
        self.events: list[Event] = []
        self.step_times: list[int] = []
        self.history: list[StepSnapshot] = []
        self._begin_step(advance=False)
        self._resolve()

    # ------------------------------------------------------------ public API
    def apply(self, train_id: int) -> None:
        """Grant the contested section of the pending decision to `train_id`."""
        d = self.decision
        if d is None:
            raise RuntimeError("no decision pending")
        if train_id not in d.candidates:
            raise ValueError(f"train {train_id} is not a legal candidate {d.candidates}")
        self.decision = None
        self._grant(train_id, d.section)
        self._resolve()

    def position(self, i: int) -> tuple[str, int] | None:
        st = self.trains[i]
        if st.status == "at_node":
            return ("node", self.scenario.trains[i].route[st.route_idx])
        if st.status == "on_section":
            return ("section", st.section)
        return None

    def next_section(self, i: int) -> Section:
        route = self.scenario.trains[i].route
        k = self.trains[i].route_idx
        return self.infra.section_between(route[k], route[k + 1])

    def direction_on(self, i: int, s: Section) -> int:
        route = self.scenario.trains[i].route
        k = self.trains[i].route_idx
        return 1 if (route[k], route[k + 1]) == (s.u, s.v) else -1

    def node_load(self, n: int) -> int:
        """Trains at node n plus trains on a section heading to n."""
        load = 0
        for i, st in enumerate(self.trains):
            route = self.scenario.trains[i].route
            if st.status == "at_node" and route[st.route_idx] == n:
                load += 1
            elif st.status == "on_section" and route[st.route_idx + 1] == n:
                load += 1
        return load

    def can_enter(self, i: int, s: Section) -> bool:
        d = self.direction_on(i, s)
        dirs = [self.trains[j].direction for j in self.occupants[s.id]]
        if s.kind == "single":
            if len(dirs) >= s.capacity or any(x != d for x in dirs):
                return False
        elif dirs.count(d) >= s.capacity:
            return False
        caps = self.infra.node_capacity
        if caps is not None:
            target = self.scenario.trains[i].route[self.trains[i].route_idx + 1]
            if caps[target] is not None and self.node_load(target) >= caps[target]:
                return False
        return True

    def is_deadlocked(self) -> bool:
        """Fixed-point deadlock test (see module docstring)."""
        unfinished = [st for st in self.trains if st.status != "finished"]
        if not unfinished:
            return False
        if any(st.status in ("on_section", "pending") for st in unfinished):
            return False
        return not any(self.can_enter(i, self.next_section(i))
                       for i, st in enumerate(self.trains) if st.status == "at_node")

    # ------------------------------------------------------------ the phases
    def _log(self, phase: int, train: int, kind: str, section: int | None = None):
        self.events.append(Event(self.t, phase, train, kind, section))

    def _begin_step(self, advance: bool) -> None:
        """Phases 1-5 (phase 1 only if `advance`)."""
        if advance:
            self.t += 1                                            # phase 1
        specs = self.scenario.trains
        for i, st in enumerate(self.trains):                       # phase 2
            if st.status == "on_section":
                st.remaining -= 1
                if st.remaining == 0:
                    self.occupants[st.section].remove(i)
                    self._log(2, i, "release", st.section)
                    st.section, st.direction = None, 0
                    st.route_idx += 1
                    st.status = "at_node"
        for i, st in enumerate(self.trains):                       # phase 3
            if st.status == "at_node" and st.route_idx == len(specs[i].route) - 1:
                st.status = "finished"
                st.arrival_time = self.t
                self._log(3, i, "finish")
        for i, st in enumerate(self.trains):                       # phase 4
            if st.status == "pending" and specs[i].departure == self.t:
                st.status = "at_node"
                self._log(4, i, "ready")
        self._entered: list[int] = []                              # phase 5
        self._granted_in_pass = False
        self._todo_sections = self._requested_sections()

    def _requested_sections(self) -> list[int]:
        return sorted({self.next_section(i).id for i, st in enumerate(self.trains)
                       if st.status == "at_node"})

    def _requesters(self, sid: int) -> list[int]:
        return [i for i, st in enumerate(self.trains)
                if st.status == "at_node" and self.next_section(i).id == sid]

    def _resolve(self) -> None:
        """Phases 6-8 for the remaining sections, then 9-10; repeat until a
        decision is needed or the episode ends."""
        while not self.done:
            while self._todo_sections:
                sid = self._todo_sections[0]
                s = self.infra.sections[sid]
                legal = [i for i in self._requesters(sid) if self.can_enter(i, s)]  # 6
                contested = None
                for group in self._resource_groups(legal, s):                   # 7
                    if self._all_fit(group, s):
                        for i in group:
                            self._grant(i, sid)
                    elif contested is None:
                        contested = group
                if contested is not None:
                    self.decision = Decision(self.t, sid, tuple(contested))
                    return
                self._todo_sections.pop(0)
                if not self._todo_sections and self._granted_in_pass:
                    # another pass: a grant may have freed a node slot elsewhere
                    self._granted_in_pass = False
                    self._todo_sections = self._requested_sections()
            for i in sorted(self._entered):                                     # 8
                self._log(8, i, "enter", self.trains[i].section)
            self._end_step()
            if not self.done:
                self._begin_step(advance=True)

    def _resource_groups(self, legal: list[int], s: Section) -> list[list[int]]:
        """Requests competing for the same resource: the whole section for single
        track, each direction separately for double track."""
        if s.kind == "single":
            return [legal] if legal else []
        groups = [[i for i in legal if self.direction_on(i, s) == d] for d in (1, -1)]
        return [g for g in groups if g]

    def _all_fit(self, legal: list[int], s: Section) -> bool:
        """True if all legal requesters can enter together (granted in id order)."""
        saved_occ = list(self.occupants[s.id])
        saved = {i: (self.trains[i].status, self.trains[i].direction) for i in legal}
        ok = True
        for i in legal:
            if not self.can_enter(i, s):
                ok = False
                break
            self.trains[i].direction = self.direction_on(i, s)
            self.trains[i].status = "on_section"   # counts towards the target node
            self.occupants[s.id].append(i)
        self.occupants[s.id] = saved_occ
        for i, (status, direction) in saved.items():
            self.trains[i].status, self.trains[i].direction = status, direction
        return ok

    def _grant(self, i: int, sid: int) -> None:
        s = self.infra.sections[sid]
        st = self.trains[i]
        st.direction = self.direction_on(i, s)
        st.status = "on_section"
        st.section = sid
        st.remaining = s.travel_time
        self.occupants[sid].append(i)
        self._entered.append(i)
        self._granted_in_pass = True

    def _end_step(self) -> None:
        """Phases 9-10."""
        for i, st in enumerate(self.trains):                       # phase 9
            if st.status == "at_node":
                st.waiting += 1
                self.total_waiting += 1
        self.step_times.append(self.t)                             # phase 10
        self.history.append(StepSnapshot(
            self.t,
            tuple(TrainSnapshot(st.status, self.position(i), st.route_idx, st.remaining,
                                st.waiting, st.direction)
                  for i, st in enumerate(self.trains)),
            tuple(tuple(o) for o in self.occupants),
            self.total_waiting))
        if all(st.status == "finished" for st in self.trains):
            self.done, self.termination = True, "success"
        elif not self._entered and self.is_deadlocked():
            self.done, self.termination = True, "deadlock"
