"""Q5. Does preparing the chart earlier leave more berths empty when travellers hedge?

Charts at 4 hours (the rule until 2025), 8 hours (mid-2025) and 10 hours
(today) before departure, with and without hedging. Empty
berths are split by what vacated them.
"""

import matplotlib.pyplot as plt

from hedging import equilibrium
from hedging.metrics import EMPTY_CAUSES
from hedging.simulate import evaluate, workers

from .common import describe, print_table, save_figure, save_table, setup

CHART_HOURS = [4, 8, 10]


def main():
    cfg, args = setup(__doc__)
    seeds = range(args.seasons)
    rows = []
    with workers() as pool:
        for hours in CHART_HOURS:
            market = cfg.with_policy(chart=hours)
            solved = equilibrium.solve(market, pool)
            print(f"chart {hours:>2} h  {describe(solved['profile'])}")
            behaviours = {"nobody hedges": "single", "equilibrium": solved["profile"]}
            for behaviour, profile in behaviours.items():
                rows.append({"chart_hours": hours, "behaviour": behaviour,
                             **evaluate(market, profile, seeds, pool)})

    save_table(rows, "q5_chart")
    print()
    print_table(rows, ["chart_hours", "behaviour", "empty_berths", "empty_berths_ci",
                       *[f"empty: {c}" for c in EMPTY_CAUSES], "travel_rate", "welfare", "welfare_ci"])

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, behaviour in zip(axes, ("nobody hedges", "equilibrium")):
        mine = [r for r in rows if r["behaviour"] == behaviour]
        labels = [f"{r['chart_hours']} h" for r in mine]
        bottom = [0.0] * len(mine)
        for cause in EMPTY_CAUSES:
            heights = [100 * r[f"empty: {cause}"] for r in mine]
            ax.bar(labels, heights, bottom=bottom, label=cause)
            bottom = [b + h for b, h in zip(bottom, heights)]
        ax.set_title(behaviour)
        ax.set_xlabel("chart prepared before departure")
    axes[0].set_ylabel("berths departing empty (%)")
    axes[1].legend(frameon=False, fontsize=8)
    save_figure(fig, "q5_chart")


if __name__ == "__main__":
    main()
