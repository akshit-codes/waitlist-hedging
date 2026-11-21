"""Run seasons, one at a time or many in parallel."""

import random
from concurrent.futures import ProcessPoolExecutor

from .market import Season, make_travellers
from .metrics import average, summarise
from .strategies import Strategy, menu


def assign(cfg, travellers, profile, seed):
    """Give every traveller a strategy.

    `profile` is a strategy (or its name) that everyone plays, or a dict from
    type name to either a strategy name or a mix {strategy name: share}.
    """
    options = menu(len(cfg.trains))
    rng = random.Random(f"strategies-{seed}")
    for me in travellers:
        choice = profile[me.type] if isinstance(profile, dict) else profile
        if isinstance(choice, dict):
            names = list(choice)
            choice = rng.choices(names, weights=[choice[n] for n in names])[0]
        me.strategy = choice if isinstance(choice, Strategy) else options[choice]


def play(cfg, profile, seed):
    travellers = make_travellers(cfg, seed)
    assign(cfg, travellers, profile, seed)
    return Season(cfg, travellers, seed).run()


def _measure_one(args):
    cfg, profile, seed, measure = args
    return measure(play(cfg, profile, seed))


def seasons(cfg, profile, seeds, measure=summarise, pool=None):
    """Play one season per seed and return `measure(season)` for each."""
    jobs = [(cfg, profile, s, measure) for s in seeds]
    if pool is None:
        return [_measure_one(j) for j in jobs]
    return list(pool.map(_measure_one, jobs, chunksize=8))


def evaluate(cfg, profile, seeds, pool=None):
    """Average metrics of a strategy profile, with 95% half-widths under `<metric>_ci`."""
    return average(seasons(cfg, profile, seeds, summarise, pool))


def workers():
    return ProcessPoolExecutor()
