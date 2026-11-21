"""Type-level equilibrium by best-response iteration on simulated payoffs.

Travellers of one type share a mix over the strategy menu. To find out what a
strategy is worth against the current mix we replay the same season several
times, each time switching a small random group of travellers (the "focal"
group) to one strategy. Everything else about the season is identical, so the
difference in the focal travellers' utilities is the gain from deviating.

A Nash deviation is one traveller's. We move a group at once (10% by default)
because one traveller per season gives very noisy payoffs;
experiments/equilibrium_check.py repeats the search with a far smaller group.
"""

import random
from statistics import mean, stdev

from .market import Season, make_travellers
from .simulate import assign
from .strategies import menu

FOCAL_SHARE = 0.1


def _focal_payoffs(args):
    cfg, profile, seed, focal_share = args
    options = menu(len(cfg.trains))
    picker = random.Random(f"focal-{seed}")
    n_focal = max(1, int(focal_share * cfg.count))
    focal = set(picker.sample(range(cfg.count), n_focal))

    payoffs = {t: {} for t in cfg.type_names}
    for name, strategy in options.items():
        travellers = make_travellers(cfg, seed)
        assign(cfg, travellers, profile, seed)
        for me in travellers:
            if me.id in focal:
                me.strategy = strategy
        season = Season(cfg, travellers, seed).run()
        for kind in cfg.type_names:
            mine = [season.utility(me) for me in travellers if me.id in focal and me.type == kind]
            payoffs[kind][name] = mean(mine) if mine else None
    return payoffs


def strategy_payoffs(cfg, profile, seeds, pool=None, focal_share=FOCAL_SHARE):
    """Expected utility of every strategy for every type, against `profile`.

    Returns {type: {strategy: (mean, standard error)}}. Standard errors are of
    the difference from the type's best strategy, since that is what decides
    a best response.
    """
    jobs = [(cfg, profile, s, focal_share) for s in seeds]
    runs = list(pool.map(_focal_payoffs, jobs, chunksize=4)) if pool else map(_focal_payoffs, jobs)
    runs = list(runs)
    out = {}
    for kind in cfg.type_names:
        rows = [r[kind] for r in runs if None not in r[kind].values()]
        means = {name: mean(r[name] for r in rows) for name in rows[0]}
        best = max(means, key=means.get)
        out[kind] = {}
        for name in means:
            gaps = [r[name] - r[best] for r in rows]
            se = stdev(gaps) / len(gaps) ** 0.5 if name != best else 0.0
            out[kind][name] = (means[name], se)
    return out


def uniform_profile(cfg):
    names = list(menu(len(cfg.trains)))
    return {t: {n: 1 / len(names) for n in names} for t in cfg.type_names}


def pure_profile(cfg, name):
    return {t: {n: float(n == name) for n in menu(len(cfg.trains))} for t in cfg.type_names}


def regret(mix, payoffs):
    """What a traveller of this type gains by switching to her best strategy."""
    current = sum(share * payoffs[name][0] for name, share in mix.items())
    return max(p[0] for p in payoffs.values()) - current


def solve(cfg, pool=None, start=None, rounds=20, seeds_per_round=120, seed=0,
          focal_share=FOCAL_SHARE):
    """Damped best-response iteration.

    Each round every type moves part of the way towards its best response to
    the current mix. The step shrinks over rounds, so a type whose best
    response keeps flipping settles on a mix instead of cycling.
    """
    profile = start or uniform_profile(cfg)
    history = []
    for r in range(rounds):
        seeds = range(seed + r * seeds_per_round, seed + (r + 1) * seeds_per_round)
        payoffs = strategy_payoffs(cfg, profile, seeds, pool, focal_share)
        history.append({
            "profile": {t: dict(m) for t, m in profile.items()},
            "regret": {t: regret(profile[t], payoffs[t]) for t in profile},
        })
        step = 2 / (r + 3)
        for kind, mix in profile.items():
            best = max(payoffs[kind], key=lambda n: payoffs[kind][n][0])
            for name in mix:
                mix[name] += step * ((name == best) - mix[name])

    # check the final mix on seeds it has not seen
    check = range(seed + 10**6, seed + 10**6 + 2 * seeds_per_round)
    payoffs = strategy_payoffs(cfg, profile, check, pool, focal_share)
    return {
        "profile": profile,
        "payoffs": payoffs,
        "regret": {t: regret(profile[t], payoffs[t]) for t in profile},
        "history": history,
    }
