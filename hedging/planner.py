"""Benchmark: a planner who knows everyone's values and hands out the berths.

She also knows whose plans will fall through, so no berth is wasted on them.
No booking rule can reach this; it marks the most welfare the trains can give.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .market import make_travellers


def planner_welfare(cfg, seed):
    people = [p for p in make_travellers(cfg, seed) if p.plan_change_at is None]
    # one column per berth; a traveller's gain is her value over the outside option
    train_of_berth = [k for k, t in enumerate(cfg.trains) for _ in range(t.capacity)]
    gain = np.array([[p.values[k] - cfg.outside_option for k in train_of_berth] for p in people])
    gain = np.maximum(gain, 0)      # leaving a traveller at home is always allowed
    rows, cols = linear_sum_assignment(gain, maximize=True)
    seated = int((gain[rows, cols] > 0).sum())
    empty = len(train_of_berth) - seated
    return (
        cfg.count * cfg.outside_option
        + gain[rows, cols].sum()
        - cfg.empty_berth_cost * empty
    )
