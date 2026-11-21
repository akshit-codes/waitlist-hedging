"""Q1. Is hedging a prisoner's dilemma?

We vary the share of travellers who hedge and measure, against each mix, what
a traveller earns with a single ticket and with a hedge. Hedging is a dilemma
for a type if the hedge pays more whatever the others do, and yet the type is
worse off when everybody hedges than when nobody does.
"""

import matplotlib.pyplot as plt

from hedging.equilibrium import strategy_payoffs
from hedging.simulate import evaluate, workers

from .common import TYPE_COLOURS, print_table, save_figure, save_table, setup

SHARES = [0.0, 0.25, 0.5, 0.75, 1.0]
HEDGE = "hedge2-timely"


def main():
    cfg, args = setup(__doc__)
    seeds = range(args.seasons)
    rows = []
    with workers() as pool:
        for share in SHARES:
            mix = {"single": 1 - share, HEDGE: share}
            profile = {kind: mix for kind in cfg.type_names}
            payoffs = strategy_payoffs(cfg, profile, seeds, pool)
            market = evaluate(cfg, profile, seeds, pool)
            for kind in cfg.type_names:
                single, hedge = payoffs[kind]["single"][0], payoffs[kind][HEDGE][0]
                error = max(payoffs[kind]["single"][1], payoffs[kind][HEDGE][1])
                rows.append({
                    "hedging_share": share,
                    "type": kind,
                    "utility_single": single,
                    "utility_hedge": hedge,
                    "gain_from_hedging": hedge - single,
                    "gain_se": error,
                    "market_mean_utility": market["mean_utility"],
                    "market_welfare": market["welfare"],
                })

    save_table(rows, "q1_dilemma")
    print_table(rows, ["hedging_share", "type", "utility_single", "utility_hedge",
                       "gain_from_hedging", "gain_se"])
    print()
    for kind in cfg.type_names:
        mine = [r for r in rows if r["type"] == kind]
        always_pays = all(r["gain_from_hedging"] > 2 * r["gain_se"] for r in mine)
        nobody, everybody = mine[0]["utility_single"], mine[-1]["utility_hedge"]
        verdict = "a prisoner's dilemma" if always_pays and everybody < nobody else "not a dilemma"
        print(f"{kind}: {verdict} "
              f"(nobody hedges {nobody:.0f}, everybody hedges {everybody:.0f})")

    fig, axes = plt.subplots(1, len(cfg.type_names), figsize=(11, 3.4), sharex=True)
    for ax, kind in zip(axes, cfg.type_names):
        mine = [r for r in rows if r["type"] == kind]
        colour = TYPE_COLOURS.get(kind, "black")
        ax.plot(SHARES, [r["utility_hedge"] for r in mine], "o-", color=colour, label="hedges")
        ax.plot(SHARES, [r["utility_single"] for r in mine], "o--", color=colour, alpha=0.6,
                label="single ticket")
        ax.set_title(kind)
        ax.set_xlabel("share of travellers who hedge")
        ax.legend(frameon=False)
    axes[0].set_ylabel("expected utility (Rs.)")
    save_figure(fig, "q1_dilemma")


if __name__ == "__main__":
    main()
