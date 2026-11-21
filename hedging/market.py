"""One booking season: travellers book, cancel and get promoted until the trains leave.

The clock runs in hours from the moment bookings open. Each train has its own
departure and its own chart, `chart_hours` before it leaves. Before the chart a
vacated berth goes to the head of that train's waitlist; after it the waitlist
no longer exists and the berth is either resold or departs empty.
"""

import heapq
import random
from collections import Counter, deque
from dataclasses import dataclass, field

CONFIRMED, WAITLISTED, CLOSED = "confirmed", "waitlisted", "closed"

# events at the same instant are handled in this order
CHART, DEPARTURE, FEE_STEP, PLAN_CHANGE, ARRIVAL = range(5)



@dataclass(eq=False)
class Ticket:
    holder: "Traveller"
    train: "Train"
    status: str
    booked_at: float
    position: int = 0               # place in the waitlist when booked, 0 if confirmed at once
    confirmed_at: float = None
    at_chart: bool = False          # confirmed by the quota release at charting
    redundant_since: float = None
    closed_at: float = None
    ending: str = ""                # travelled, released, lapsed, plan_change, dropped

    @property
    def live(self):
        return self.status != CLOSED


@dataclass(eq=False)
class Traveller:
    id: int
    type: str
    values: tuple
    arrives: float
    plan_change_at: float = None    # None if her plans hold
    strategy: object = None

    tickets: list = field(default_factory=list)
    boarded: "Train" = None
    changed_plans: bool = False
    turned_away: bool = False
    paid: float = 0.0               # fares, cancellation charges and forfeited fares
    refunds: float = 0.0
    releases: int = 0               # cancellations she had to make herself
    decisions: list = field(default_factory=list)   # filled by learning strategies
    notes: dict = field(default_factory=dict)       # filled by probing strategies

    def held(self, status=None):
        return [t for t in self.tickets if t.live and status in (None, t.status)]

    @property
    def hedged(self):
        return len(self.tickets) > 1


class Train:
    def __init__(self, index, spec, cfg):
        self.index = index
        self.name = spec.name
        self.fare = spec.fare
        self.capacity = spec.capacity
        self.departure = cfg.advance_days * 24 + spec.departs
        self.chart_time = self.departure - cfg.chart_hours
        self.slots = int(cfg.waitlist_cap * spec.capacity)
        self.quota = min(cfg.quota_berths, spec.capacity)
        self.confirmed = []
        self.queue = []
        self.charted = False
        self.resold = 0
        self.empty = Counter()      # berths that left empty, by cause

    @property
    def free_berths(self):
        if self.charted:
            return 0
        return self.capacity - self.quota - len(self.confirmed)

    @property
    def waitlist_open(self):
        return not self.charted and len(self.queue) < self.slots


class Season:
    def __init__(self, cfg, travellers, seed=0):
        self.cfg = cfg
        self.travellers = travellers
        self.trains = [Train(i, spec, cfg) for i, spec in enumerate(cfg.trains)]
        self.rng = random.Random(f"market-{seed}")
        self.now = 0.0
        self.refunds_paid = 0.0
        self.charges_collected = 0.0
        self._to_review = deque()

    def run(self):
        events = []
        for me in self.travellers:
            events.append((me.arrives, ARRIVAL, me.id, me))
            if me.plan_change_at is not None:
                events.append((me.plan_change_at, PLAN_CHANGE, me.id, me))
        for train in self.trains:
            events.append((train.chart_time, CHART, train.index, train))
            events.append((train.departure, DEPARTURE, train.index, train))
            for hours in (72, 24, self.cfg.no_refund_hours):
                events.append((train.departure - hours, FEE_STEP, train.index, train))
        heapq.heapify(events)

        handlers = {
            ARRIVAL: self._arrive,
            PLAN_CHANGE: self._change_plans,
            CHART: self._chart,
            DEPARTURE: self._depart,
            FEE_STEP: self._recheck_holders,
        }
        while events:
            self.now, kind, _, subject = heapq.heappop(events)
            handlers[kind](subject)
            self._settle()
        return self

    # --- what a traveller can see and do -------------------------------------

    def surplus(self, me, train):
        return me.values[train.index] - train.fare

    def ranking(self, me):
        """Trains still on sale that beat her outside option, best first."""
        worth_it = [t for t in self.trains
                    if not t.charted and self.surplus(me, t) > self.cfg.outside_option]
        return sorted(worth_it, key=lambda t: self.surplus(me, t), reverse=True)

    def cancel_charge(self, train, at=None):
        """What the railway keeps if a confirmed ticket is cancelled now (or at time `at`)."""
        hours_left = train.departure - (self.now if at is None else at)
        if hours_left > 72:
            return self.cfg.confirmed_flat
        if hours_left > 24:
            return self.cfg.confirmed_72h * train.fare
        if hours_left > self.cfg.no_refund_hours:
            return self.cfg.confirmed_24h * train.fare
        return train.fare

    def keep_value(self, me, train):
        """What a confirmed ticket on `train` is worth to its holder right now.

        Travelling on it yields the surplus; giving it up instead costs the
        cancellation charge. So another train is only worth switching to if
        its surplus beats this sum. In the last hours nothing is refunded,
        the fare is sunk, and the ticket is worth its full value.
        """
        return self.surplus(me, train) + self.cancel_charge(train)

    def request(self, me, train):
        if train.free_berths > 0:
            ticket = Ticket(me, train, CONFIRMED, self.now, confirmed_at=self.now)
            train.confirmed.append(ticket)
        elif train.waitlist_open:
            ticket = Ticket(me, train, WAITLISTED, self.now, position=len(train.queue) + 1)
            train.queue.append(ticket)
        else:
            return None
        me.tickets.append(ticket)
        return ticket

    def release(self, ticket):
        """Voluntary release of a redundant ticket by its holder."""
        me = ticket.holder
        if ticket.status == WAITLISTED and self._refund_due(ticket):
            me.refunds += self.cfg.release_refund
            self.refunds_paid += self.cfg.release_refund
        if ticket.status == WAITLISTED or self.cancel_charge(ticket.train) < ticket.train.fare:
            me.releases += 1    # with nothing to get back she does not bother: she just stays away
        self._close(ticket, "released")

    # --- events ------------------------------------------------------------

    def _arrive(self, me):
        me.strategy.book(me, self)
        me.turned_away = not me.tickets

    def _change_plans(self, me):
        if me.boarded:
            return
        me.changed_plans = True
        for ticket in me.held():
            self._close(ticket, "plan_change")

    def _chart(self, train):
        promoted = train.queue[: train.quota]
        for ticket in promoted:
            self._confirm(ticket, at_chart=True)
        train.charted = True
        for ticket in list(train.queue):
            self._close(ticket, "lapsed")
        for _ in range(train.capacity - len(train.confirmed)):
            self._offer_late(train, "unsold")
        self._recheck_holders(train)

    def _recheck_holders(self, train):
        """Giving up a berth on `train` just got dearer, so a pending upgrade may no longer pay."""
        self._to_review.extend(t.holder for t in train.confirmed)

    def _depart(self, train):
        for ticket in list(train.confirmed):
            me = ticket.holder
            me.boarded = train
            ticket.status, ticket.ending, ticket.closed_at = CLOSED, "travelled", self.now
            me.paid += train.fare
            for other in me.held():
                self._close(other, "dropped")

    # --- bookkeeping -------------------------------------------------------

    def _settle(self):
        """Let every traveller whose holdings changed drop what she no longer needs."""
        while self._to_review:
            self._review(self._to_review.popleft())

    def _review(self, me):
        confirmed = me.held(CONFIRMED)
        if not confirmed:
            return
        keep = max(confirmed, key=lambda t: self.keep_value(me, t.train))
        for ticket in confirmed:
            if ticket is not keep:
                ticket.redundant_since = ticket.redundant_since or self.now
                self.release(ticket)
        # a waitlisted ticket is redundant once she would not switch to it even if it confirmed
        bar = self.keep_value(me, keep.train)
        for ticket in me.held(WAITLISTED):
            if ticket.redundant_since is None and self.surplus(me, ticket.train) <= bar:
                ticket.redundant_since = self.now
                if me.strategy.release(me, self, ticket):
                    self.release(ticket)

    def _refund_due(self, ticket):
        return self.cfg.release_refund > 0 and not ticket.train.charted

    def _confirm(self, ticket, at_chart=False):
        train = ticket.train
        train.queue.remove(ticket)
        train.confirmed.append(ticket)
        ticket.status, ticket.confirmed_at, ticket.at_chart = CONFIRMED, self.now, at_chart
        self._to_review.append(ticket.holder)

    def _close(self, ticket, ending):
        me, train = ticket.holder, ticket.train
        was_confirmed = ticket.status == CONFIRMED
        ticket.status, ticket.ending, ticket.closed_at = CLOSED, ending, self.now

        if not was_confirmed:
            train.queue.remove(ticket)
            self._charge(me, self.cfg.waitlist_cancel)
            return

        train.confirmed.remove(ticket)
        self._charge(me, self.cancel_charge(train))
        if not train.charted:
            if train.queue:
                self._confirm(train.queue[0])
        elif ending == "plan_change":
            self._offer_late(train, "late cancellation")
        elif ticket.at_chart:
            self._offer_late(train, "redundant ticket held")
        else:
            self._offer_late(train, "switched after chart")

    def _charge(self, me, amount):
        me.paid += amount
        self.charges_collected += amount

    def _offer_late(self, train, cause):
        """A berth nobody on the chart needs: resold at the counter, or it leaves empty."""
        if self.rng.random() < self.cfg.resale_prob:
            train.resold += 1
        else:
            train.empty[cause] += 1

    # --- results -----------------------------------------------------------

    def utility(self, me):
        value = me.values[me.boarded.index] if me.boarded else self.cfg.outside_option
        return value - me.paid + me.refunds - self.cfg.hassle * me.releases


def make_travellers(cfg, seed):
    """Draw the people of one season. Strategies are assigned separately."""
    rng = random.Random(f"people-{seed}")
    counts = [round(cfg.count * t.share) for t in cfg.types]
    counts[0] += cfg.count - sum(counts)
    kinds = [t for t, n in zip(cfg.types, counts) for _ in range(n)]
    rng.shuffle(kinds)

    travel_day = cfg.advance_days * 24
    first = travel_day + min(t.departs for t in cfg.trains)
    travellers = []
    for i, kind in enumerate(kinds):
        values = tuple(max(0.0, v + rng.gauss(0, cfg.value_noise)) for v in kind.values)
        arrives = rng.uniform(0, first - cfg.last_booking_hours)
        change_at = None
        if rng.random() < cfg.plan_change_prob:
            # plans mostly fall through close to the trip, so skew towards her favourite's departure
            favourite = max(range(len(values)), key=lambda k: values[k] - cfg.trains[k].fare)
            leaves = travel_day + cfg.trains[favourite].departs
            change_at = leaves - (leaves - arrives) * rng.random() ** 2
        travellers.append(Traveller(i, kind.name, values, arrives, change_at))
    return travellers
