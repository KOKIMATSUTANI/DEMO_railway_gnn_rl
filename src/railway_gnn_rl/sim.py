"""Discrete-time railway dispatching simulator.

Time step dt = 1 minute; all times are integer minutes.  One simulation step at
(new) time t runs these phases in this exact order:

  1. advance clock t -> t+1
  2. trains completing a section release it and arrive at the next node
  3. trains at their destination are marked finished
  4. trains whose departure time equals t become ready (at their origin)
  5. ready, unfinished trains at a node request their next section
  6. conflict detection per section (ascending section id)
  7. decision: grant automatically if all legal requests fit, else pause and
     expose a Decision to the agent
  8. granted trains enter the section (remaining time = travel time)
  9. waiting-time accounting
 10. reward and termination

Within a phase, trains are processed in ascending train id.  A section
released in phase 2 is available for entry in phase 8 of the same step.
`reset()` runs phases 2-10 for t = 0 (no clock advance).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Section:
    id: int
    u: int
    v: int
    travel_time: int
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
    route: tuple[int, ...]
    departure: int
    slack: int = 2


@dataclass(frozen=True)
class Scenario:
    infra: Infrastructure
    trains: tuple[TrainSpec, ...]
    max_time: int | None = None


@dataclass(frozen=True)
class Decision:
    t: int
    section: int
    candidates: tuple[int, ...]


@dataclass(frozen=True)
class Event:
    t: int
    phase: int
    train: int
    kind: str  # release | arrive | finish | ready | enter
    section: int | None = None


@dataclass
class TrainState:
    status: str = "pending"  # pending | at_node | on_section | finished
    route_idx: int = 0       # index into route of the current / last node
    section: int | None = None
    direction: int = 0       # +1 if traversing u->v, -1 if v->u
    remaining: int = 0
    waiting: int = 0
    arrival_time: int | None = None
    delay: int | None = None
    scheduled_arrival: int = 0


class RailwaySim:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.infra = scenario.infra
        for i, tr in enumerate(scenario.trains):
            assert tr.id == i, "train ids must be 0..n-1 in order"

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
        self._entered: set[int] = set()
        self._todo_sections: list[int] = []
        self._begin_step(advance=False)
        self._resolve()

    # ------------------------------------------------------------ public API
    def apply(self, train_id: int) -> None:
        d = self.decision
        if d is None:
            raise RuntimeError("no decision pending")
        if train_id not in d.candidates:
            raise ValueError(f"train {train_id} is not a legal candidate {d.candidates}")
        self.decision = None
        self._grant(train_id, d.section)
        self._resolve()

    # --------------------------------------------------------------- helpers
    def next_section(self, i: int) -> Section:
        route = self.scenario.trains[i].route
        k = self.trains[i].route_idx
        return self.infra.section_between(route[k], route[k + 1])

    def _direction(self, i: int, s: Section) -> int:
        route = self.scenario.trains[i].route
        k = self.trains[i].route_idx
        return 1 if (route[k], route[k + 1]) == (s.u, s.v) else -1

    def _log(self, phase: int, train: int, kind: str, section: int | None = None):
        self.events.append(Event(self.t, phase, train, kind, section))

    def can_enter(self, i: int, s: Section) -> bool:
        return len(self.occupants[s.id]) < s.capacity

    # ------------------------------------------------------------ the phases
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
        self._entered = set()                                      # phase 5
        self._granted: list[tuple[int, int]] = []
        requested = {self.next_section(i).id for i, st in enumerate(self.trains)
                     if st.status == "at_node"}
        self._todo_sections = sorted(requested)

    def _requesters(self, sid: int) -> list[int]:
        return [i for i, st in enumerate(self.trains)
                if st.status == "at_node" and i not in self._entered
                and self.next_section(i).id == sid]

    def _resolve(self) -> None:
        """Phases 6-8 for the remaining sections, then 9-10; repeat until a
        decision is needed or the episode ends."""
        while not self.done:
            while self._todo_sections:
                sid = self._todo_sections[0]
                s = self.infra.sections[sid]
                legal = [i for i in self._requesters(sid) if self.can_enter(i, s)]  # phase 6
                if not legal:
                    self._todo_sections.pop(0)
                    continue
                if self._all_fit(legal, s):                                     # phase 7
                    for i in legal:
                        self._grant(i, sid)
                    self._todo_sections.pop(0)
                    continue
                self.decision = Decision(self.t, sid, tuple(legal))
                return
            for i, sid in sorted(self._granted):                                # phase 8
                self._enter(i, sid)
            self._end_step()
            if not self.done:
                self._begin_step(advance=True)

    def _all_fit(self, legal: list[int], s: Section) -> bool:
        saved = list(self.occupants[s.id])
        ok = True
        for i in legal:
            if not self.can_enter(i, s):
                ok = False
                break
            self.occupants[s.id].append(i)
        self.occupants[s.id] = saved
        return ok

    def _grant(self, i: int, sid: int) -> None:
        """Reserve section capacity for train i (committed in phase 8)."""
        self.occupants[sid].append(i)
        self._entered.add(i)
        self._granted.append((i, sid))

    def _enter(self, i: int, sid: int) -> None:
        s = self.infra.sections[sid]
        st = self.trains[i]
        st.direction = self._direction(i, s)
        st.status = "on_section"
        st.section = sid
        st.remaining = s.travel_time
        self._log(8, i, "enter", sid)

    def _end_step(self) -> None:
        """Phases 9-10."""
        for i, st in enumerate(self.trains):                       # phase 9
            if st.status == "at_node" and i not in self._entered:
                st.waiting += 1
                self.total_waiting += 1
        self.step_times.append(self.t)                             # phase 10
        if all(st.status == "finished" for st in self.trains):
            self.done = True
            self.termination = "success"
