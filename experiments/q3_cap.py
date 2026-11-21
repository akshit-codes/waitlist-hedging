"""Q3. Does a tighter waitlist cap reduce the loss, or set off a race for slots?

For each cap we recompute how travellers respond and compare the outcome with
the same market if nobody hedged.
"""

import matplotlib.pyplot as plt

from hedging import equilibrium
from hedging.simulate import evaluate, workers

from .common import describe, print_table, save_figure, save_table, setup

CAPS = [0.10, 0.25, 0.60, 1.00]      # 25% was the June 2025 rule, 60% is today's for AC classes


def main():
    cfg, args = setup(__doc__)
    seeds = range(args.seasons)
    rows = []
    with workers() as pool:
        for cap in CAPS:
            market = cfg.with_policy(cap=cap)
            solved = equilibrium.solve(market, pool)
            print(f"cap {cap:.2f}  {describe(solved['profile'])}")
            behaviours = {"nobody hedges": "single", "equilibrium": solved["profile"]}
            for behaviour, profile in behaviours.items():
                rows.append({"cap": cap, "behaviour": behaviour,
                             **evaluate(market, profile, seeds, pool)})

    save_table(rows, "q3_cap")
    print()
    print_table(rows, ["cap", "behaviour", "welfare", "welfare_ci", "hedger_share",
                       "tickets_per_traveller", "turned_away", "empty_berths", "waitlist_confirm_rate"])

    panels = [("welfare", "welfare (Rs.)", 1), ("turned_away", "turned away (%)", 100),
              ("hedger_share", "travellers who hedge (%)", 100)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    for ax, (key, label, scale) in zip(axes, panels):
        for behaviour, style in (("nobody hedges", "o--"), ("equilibrium", "o-")):
            mine = [r for r in rows if r["behaviour"] == behaviour]
            ax.errorbar([r["cap"] for r in mine], [scale * r[key] for r in mine],
                        yerr=[scale * r[key + "_ci"] for r in mine], fmt=style, label=behaviour)
        ax.set_xlabel("waitlist cap (share of capacity)")
        ax.set_ylabel(label)
    axes[0].legend(frameon=False)
    save_figure(fig, "q3_cap")


if __name__ == "__main__":
    main()
