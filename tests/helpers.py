"""A two-train market small enough to work out by hand.

Train "a" leaves at 14:00 and train "b" at 22:00 on day 30, so on the model's
clock they depart at 734 and 742. With the chart 8 hours ahead, a's chart is
at 726 and b's at 734. Each train has two berths at Rs. 800. A confirmed
ticket costs Rs. 180 to cancel until 72 hours before departure, 25% of the
fare until 24 hours, 50% until 8 hours, and the whole fare after that.
"""

from dataclasses import replace

from hedging import config
from hedging.config import TrainSpec
from hedging.market import Season, Traveller
from hedging.strategies import Fixed

A, B = 0, 1


def market(**changes):
    cfg = replace(
        config.load(),
        trains=(TrainSpec("a", 14, 2, 800), TrainSpec("b", 22, 2, 800)),
        waitlist_cap=1.0,
        chart_hours=8,
        quota_berths=0,
        resale_prob=0.0,
        release_refund=0,
    )
    return replace(cfg, **changes)


def person(id, values, arrives, strategy=None, plan_change_at=None):
    return Traveller(id, "anyone", values, arrives, plan_change_at, strategy or Fixed(1))


def run(cfg, people):
    return Season(cfg, people).run()


def ticket(me, train_index):
    return next(t for t in me.tickets if t.train.index == train_index)
