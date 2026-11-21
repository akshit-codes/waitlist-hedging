import pytest

from hedging.market import Season
from hedging.rules import Beliefs, bucket, fit, hedge_margin, release_margin
from hedging.strategies import Fixed

from .helpers import A, market, person, ticket


def half_played(people, until):
    """A season stopped at time `until`, to look at it from a traveller's seat."""
    cfg = market()
    season = Season(cfg, people)
    for me in sorted(people, key=lambda p: p.arrives):
        if me.arrives <= until:
            season.now = me.arrives
            me.strategy.book(me, season)
    season.now = until
    return season


def test_bucket():
    assert [bucket(p) for p in (1, 6, 7, 12, 13, 24, 25)] == [0, 0, 1, 1, 2, 2, 3]


def test_fit_recovers_observed_frequencies():
    seen = [(0, 1, "before chart")] * 30 + [(0, 1, "at chart")] * 10 + [(0, 1, "never")] * 60
    beliefs = fit(seen)
    assert beliefs.table[0, None] == (0.3, 0.1)
    assert beliefs.table[0, 0] == (0.3, 0.1)


def test_hedge_margin_with_a_confirmed_base():
    people = [person(0, (0, 2000), 1), person(1, (0, 2000), 2), person(2, (1500, 2600), 3, Fixed(1))]
    season = half_played(people, until=3)
    me = people[2]
    a, b = season.trains
    assert ticket(me, A).status == "confirmed"

    beliefs = Beliefs(default=(0.4, 0.1))               # confirms with chance 0.5
    # gain: 0.5 x (1800 - 700); cost: 0.5 x 60 if it lapses + 0.5 x 180 for giving a up
    assert hedge_margin(me, season, b, a, beliefs) == pytest.approx(550 - 30 - 90)


def test_hedge_margin_counts_only_cases_where_better_tickets_fail():
    full = [person(i, (2000, 0), i) for i in (1, 2)] + [person(i, (0, 2000), i) for i in (3, 4)]
    me = person(9, (2000, 1500), 5, Fixed(1))
    season = half_played(full + [me], until=5)
    a, b = season.trains
    assert ticket(me, A).status == "waitlisted"

    beliefs = Beliefs(default=(0.4, 0.1))
    # b only matters if a falls through: 0.5 x 0.5 x (700 - 400), minus 0.5 x 60
    assert hedge_margin(me, season, b, None, beliefs) == pytest.approx(75 - 30)


def test_release_margin():
    full = [person(i, (2000, 0), i) for i in (1, 2)]
    me = person(9, (2000, 0), 3)
    season = half_played(full + [me], until=3)
    season.now = 700                    # 34 h before a leaves: the charge is 25% of the fare
    waiting = ticket(me, A)

    beliefs = Beliefs(default=(0.2, 0.05))
    # at the chart, 8 h before departure, the whole fare is gone
    saved = 0.2 * (200 - 60) + 0.05 * (800 - 60)
    assert release_margin(me, season, waiting, beliefs) == pytest.approx(saved - 40)


def test_fit_measures_how_often_the_better_ticket_fails_when_the_backup_confirms():
    # train 1 is the better ticket, train 0 the backup; the two mostly confirm together
    pairs = ([("flexible", 1, 0, True, True)] * 30 + [("flexible", 1, 0, False, True)] * 10
             + [("flexible", 1, 0, False, False)] * 60)
    beliefs = fit([], pairs=pairs)
    assert beliefs.joint[0, 1] == 0.25      # far below what independence would give
