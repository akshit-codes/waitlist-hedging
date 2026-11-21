"""Travellers who learn when to hedge and when to release, season after season.

Nobody is told the confirmation chances. A traveller faces up to two kinds of
decision in a season:

  booking   her first choice is full and one more ticket is on offer:
            how many tickets will she hold (1, 2 or 3)?
  release   a waitlisted ticket of hers has become redundant:
            cancel it now, or leave it to the chart?

She only sees what the booking site shows: whether she already has a berth,
how long the queue is, how close the chart is. Each decision is scored with
the utility she ends the season with (Monte Carlo control, epsilon-greedy).
Travellers of one type pool what they learn in a shared table, but every entry
comes from one traveller's own choice and her own payoff.
"""

import random
from collections import defaultdict

from .market import Season, make_travellers
from .metrics import summarise
from .rules import bucket
from .strategies import Strategy

LIMITS = (1, 2, 3)
RELEASE, HOLD = "release", "hold"
MIN_STEP = 0.002


class QTable:
    def __init__(self):
        self.value = defaultdict(float)
        self.visits = defaultdict(int)

    def choose(self, state, actions, epsilon, rng):
        untried = [a for a in actions if self.visits[state, a] == 0]
        if untried:
            return rng.choice(untried)
        if rng.random() < epsilon:
            return rng.choice(actions)
        return max(actions, key=lambda a: self.value[state, a])

    def update(self, state, action, reward):
        key = (state, action)
        self.visits[key] += 1
        step = max(1 / self.visits[key], MIN_STEP)
        self.value[key] += step * (reward - self.value[key])

    def best(self, state, actions):
        return max(actions, key=lambda a: self.value[state, a])


class Learner(Strategy):
    name = "learner"

    def __init__(self, table, epsilon, rng, n_trains=3):
        self.table = table
        self.epsilon = epsilon
        self.rng = rng
        self.limits = [n for n in LIMITS if n <= n_trains]

    def wants(self, me, market, train, base):
        if not me.tickets:
            return True
        if "limit" not in me.notes:
            state = ("booking", base is not None, bucket(len(train.queue) + 1))
            me.notes["limit"] = self.table.choose(state, self.limits, self.epsilon, self.rng)
            me.decisions.append((state, me.notes["limit"]))
        return len(me.tickets) < me.notes["limit"]

    def release(self, me, market, ticket):
        train = ticket.train
        place = bucket(train.queue.index(ticket) + 1)
        soon = train.chart_time - market.now <= 24
        state = ("release", place, soon)
        action = self.table.choose(state, [RELEASE, HOLD], self.epsilon, self.rng)
        me.decisions.append((state, action))
        return action == RELEASE


def play(cfg, tables, epsilon, seed):
    """One season in which everybody is a learner."""
    rng = random.Random(f"explore-{seed}")
    travellers = make_travellers(cfg, seed)
    for me in travellers:
        me.strategy = Learner(tables[me.type], epsilon, rng, len(cfg.trains))
    return Season(cfg, travellers, seed).run()


def train(cfg, seasons=6000, explore=0.3, explore_floor=0.02, seed=0, log_every=100):
    """Let the population learn. Returns the tables and a log of behaviour over time."""
    tables = {name: QTable() for name in cfg.type_names}
    log, window = [], []
    for s in range(seasons):
        epsilon = max(explore_floor, explore * (1 - s / (0.6 * seasons)))
        season = play(cfg, tables, epsilon, seed + s)
        for me in season.travellers:
            reward = season.utility(me)
            for state, action in me.decisions:
                tables[me.type].update(state, action, reward)
        window.append(behaviour(season))
        if (s + 1) % log_every == 0:
            log.append({"season": s + 1, **_mean(window)})
            window = []
    return tables, log


def behaviour(season):
    """How much the travellers hedged and released, plus the usual metrics."""
    booking = [a for me in season.travellers for st, a in me.decisions if st[0] == "booking"]
    release = [a for me in season.travellers for st, a in me.decisions if st[0] == "release"]
    row = summarise(season)
    row["chose_to_hedge"] = sum(a > 1 for a in booking) / len(booking) if booking else 0.0
    row["released_on_time"] = sum(a == RELEASE for a in release) / len(release) if release else 0.0
    return row


def _mean(rows):
    return {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
