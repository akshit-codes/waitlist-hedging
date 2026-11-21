"""The proposal's closed-form rules for hedging (Eq. 2) and releasing (Eq. 3).

Both rules need the chance that a waitlisted ticket confirms. `Beliefs` holds
those chances at several levels of detail (one flat number, per train, per
train and queue position, and with the trains no longer treated as
independent), so we can see how much the rules improve as the chances they are
fed get closer to what the market produces.
"""

from collections import defaultdict

from .market import WAITLISTED
from .strategies import Strategy

BUCKET_EDGES = (6, 12, 24)


def bucket(position):
    """Coarse place in the queue: 0 is the front, 3 is deep."""
    return sum(position > edge for edge in BUCKET_EDGES)


class Beliefs:
    """Chance that a waitlisted ticket confirms before the chart (q1) and at it (q2)."""

    def __init__(self, default=(0.20, 0.05), table=None, joint=None):
        self.default = default
        self.table = table or {}    # (train index, bucket) or (train index, None) -> (q1, q2)
        # (train, other train) -> chance the ticket on the other train never confirms,
        # given that the one on `train` does. Empty means: treat the trains as independent.
        self.joint = joint or {}

    def chances(self, train, position):
        for key in ((train.index, bucket(position)), (train.index, None)):
            if key in self.table:
                return self.table[key]
        return self.default

    def p(self, train, position):
        return sum(self.chances(train, position))

    def falls_through(self, other, train):
        """Chance that the waitlisted ticket `other` fails, given that `train` confirms."""
        independent = 1 - self.p(other.train, other.position)
        return self.joint.get((train.index, other.train.index), independent)


def observe(season, only=None):
    """(train, queue position when booked, outcome) for waitlisted tickets that ran their course.

    Tickets cancelled by their holder while still waiting are left out: we
    never learn whether they would have confirmed. That biases the chances
    when travellers release redundant tickets, so `only` can restrict the
    count to a group of travellers who were made to hold everything.
    """
    seen = []
    for me in season.travellers:
        if only is not None and me.id not in only:
            continue
        for t in me.tickets:
            if t.position == 0:
                continue
            if t.confirmed_at is not None:
                when = "at chart" if t.at_chart else "before chart"
                seen.append((t.train.index, t.position, when))
            elif t.ending == "lapsed":
                seen.append((t.train.index, t.position, "never"))
    return seen


def observe_pairs(season, only):
    """For travellers waitlisted on two trains: (type, first train, second train, outcomes).

    The first ticket is on the train she likes better. An outcome is True if
    the ticket confirmed. Only pairs where both tickets ran their course.
    """
    pairs = []
    for me in season.travellers:
        if me.id not in only:
            continue
        done = [t for t in me.tickets
                if t.position > 0 and (t.confirmed_at is not None or t.ending == "lapsed")]
        if len(done) == 2:
            first, second = done
            pairs.append((me.type, first.train.index, second.train.index,
                          first.confirmed_at is not None, second.confirmed_at is not None))
    return pairs


def fit(observations, by_position=True, pairs=None):
    """Beliefs that match the observed confirmation frequencies.

    With `pairs`, the beliefs also carry how often the better ticket fails
    when the backup confirms, in place of the independence assumption.
    """
    counts = defaultdict(lambda: [0, 0, 0])     # before chart, at chart, total
    for train, position, outcome in observations:
        keys = [(train, None)] + ([(train, bucket(position))] if by_position else [])
        for key in keys:
            counts[key][0] += outcome == "before chart"
            counts[key][1] += outcome == "at chart"
            counts[key][2] += 1
    table = {k: (a / n, b / n) for k, (a, b, n) in counts.items() if n >= 30}

    backup_confirmed = defaultdict(lambda: [0, 0])      # better ticket failed, total
    for _, first, second, first_ok, second_ok in pairs or []:
        if second_ok:
            backup_confirmed[second, first][0] += not first_ok
            backup_confirmed[second, first][1] += 1
    joint = {k: failed / n for k, (failed, n) in backup_confirmed.items() if n >= 30}
    return Beliefs(table=table, joint=joint)


def hedge_margin(me, market, train, base, beliefs):
    """Eq. 2: expected gain from one more waitlisted ticket, minus its expected cost.

    The ticket only matters if every train she likes better and is waiting
    for falls through. What it then buys is the step up from her fallback:
    the confirmed berth she already has, or her outside option.
    """
    cfg = market.cfg
    p = beliefs.p(train, len(train.queue) + 1)
    all_better_fail = 1.0
    for t in me.held(WAITLISTED):
        all_better_fail *= beliefs.falls_through(t, train)

    surplus = market.surplus(me, train)
    if base:
        fallback = market.surplus(me, base)
        cost_if_confirmed = market.cancel_charge(base)      # she then gives the base berth up
    else:
        fallback = cfg.outside_option
        cost_if_confirmed = 0.0
    expected_cost = (1 - p) * cfg.waitlist_cancel + p * cost_if_confirmed
    return p * all_better_fail * (surplus - fallback) - expected_cost


def release_margin(me, market, ticket, beliefs):
    """Eq. 3: what releasing a redundant waitlisted ticket now saves, minus the hassle.

    If she keeps the ticket and it confirms, she has to cancel a confirmed
    ticket instead: at today's charge if that happens before the chart, at
    the charge that applies at chart time if it happens then.
    """
    cfg, train = market.cfg, ticket.train
    q1, q2 = beliefs.chances(train, train.queue.index(ticket) + 1)
    saved = q1 * (market.cancel_charge(train) - cfg.waitlist_cancel)
    saved += q2 * (market.cancel_charge(train, at=train.chart_time) - cfg.waitlist_cancel)
    return saved + cfg.release_refund - cfg.hassle


class RuleBased(Strategy):
    """A traveller who follows Eqs. 2 and 3 with the given beliefs."""

    name = "rules"

    def __init__(self, beliefs):
        self.beliefs = beliefs

    def wants(self, me, market, train, base):
        if not me.tickets:
            return True     # the first ticket is not a hedge
        return hedge_margin(me, market, train, base, self.beliefs) > 0

    def release(self, me, market, ticket):
        return release_margin(me, market, ticket, self.beliefs) > 0


class Probe(Strategy):
    """Plays `inner`, and notes what Eqs. 2 and 3 predicted for each of its choices.

    `me.notes["hedge"][label]` adds up the Eq. 2 margins of the extra tickets
    she took; `me.notes["release"][label]` the Eq. 3 margins of the redundant
    tickets she had to decide about. One entry per set of beliefs.
    """

    def __init__(self, inner, beliefs_by_label):
        self.inner = inner
        self.beliefs = beliefs_by_label
        self.name = inner.name

    def wants(self, me, market, train, base):
        yes = self.inner.wants(me, market, train, base)
        if yes and me.tickets:
            for label, beliefs in self.beliefs.items():
                self._note(me, "hedge", label, hedge_margin(me, market, train, base, beliefs))
        return yes

    def release(self, me, market, ticket):
        for label, beliefs in self.beliefs.items():
            self._note(me, "release", label, release_margin(me, market, ticket, beliefs))
        me.notes["release decisions"] = me.notes.get("release decisions", 0) + 1
        return self.inner.release(me, market, ticket)

    @staticmethod
    def _note(me, kind, label, margin):
        tally = me.notes.setdefault(kind, {})
        tally[label] = tally.get(label, 0.0) + margin


def self_consistent_beliefs(cfg, seeds, pool=None, rounds=4):
    """Beliefs that are correct when everybody acts on them.

    Start from the flat 25% of the proposal, let rule-followers play, fit the
    chances their own behaviour produced, and repeat. These are "travellers
    who know the true chances".
    """
    from .simulate import seasons

    beliefs = Beliefs()
    for _ in range(rounds):
        seen = seasons(cfg, RuleBased(beliefs), seeds, observe, pool)
        beliefs = fit([obs for season in seen for obs in season])
    return beliefs
