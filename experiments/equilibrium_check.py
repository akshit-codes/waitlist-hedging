"""Is the equilibrium we report the only one, and how close is it to a Nash equilibrium?

What the solver finds is an approximate equilibrium between traveller types:
shares of each type on each strategy such that no type gains more than a few
rupees by moving. The table gives that regret for every type.

Two checks on the solver. First, best-response iteration is started from three
very different profiles to see whether it ends in the same place. Second, the
group of travellers who deviate together is shrunk from 10% of the population
to a single traveller, which is what the definition of Nash equilibrium asks
for. One traveller per season is a noisy measurement, so that run uses many
more seasons per round.
"""

from hedging import equilibrium
from hedging.simulate import evaluate, workers

from .common import describe, print_table, save_table, setup

ONE_TRAVELLER = 0.0     # the solver always moves at least one


def main():
    cfg, args = setup(__doc__)
    seeds = range(10**6, 10**6 + args.seasons)
    most = len(cfg.trains)
    runs = {
        "start: even mix": {},
        "start: nobody hedges": {"start": equilibrium.pure_profile(cfg, "single")},
        "start: all hedge late": {"start": equilibrium.pure_profile(cfg, f"hedge{most}-late")},
        "deviating group 3%": {"focal_share": 0.03, "seeds_per_round": 400},
        "one traveller deviates": {"focal_share": ONE_TRAVELLER, "seeds_per_round": 3000},
    }
    rows = []
    with workers() as pool:
        for label, options in runs.items():
            solved = equilibrium.solve(cfg, pool, **options)
            outcome = evaluate(cfg, solved["profile"], seeds, pool)
            print(f"{label:24s} {describe(solved['profile'])}")
            rows.append({
                "run": label,
                "hedgers": outcome["hedger_share"],
                "welfare": outcome["welfare"],
                "welfare_ci": outcome["welfare_ci"],
                **{f"{kind} hedging": 1 - solved["profile"][kind]["single"]
                   for kind in cfg.type_names},
                **{f"{kind} regret (Rs.)": solved["regret"][kind] for kind in cfg.type_names},
            })
    save_table(rows, "equilibrium_check")
    print()
    print_table(rows)


if __name__ == "__main__":
    main()
