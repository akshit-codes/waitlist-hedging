"""Q2. How much of the loss comes from hedging itself, and how much from late release?

Four reference behaviours and a planner, as in the proposal. Going from
"nobody hedges" to "everybody hedges and releases on time" isolates hedging;
switching on-time release to late release isolates the timing of release.
"""

from functools import partial

import matplotlib.pyplot as plt

from hedging import equilibrium
from hedging.metrics import EMPTY_CAUSES, average
from hedging.planner import planner_welfare
from hedging.simulate import seasons, workers

from .common import describe, print_table, save_figure, save_table, setup


def main():
    cfg, args = setup(__doc__)
    seeds = range(args.seasons)
    most = len(cfg.trains)
    with workers() as pool:
        solved = equilibrium.solve(cfg, pool)
        behaviours = {
            "nobody hedges": "single",
            "all hedge, release on time": f"hedge{most}-timely",
            "all hedge, release late": f"hedge{most}-late",
            "equilibrium": solved["profile"],
        }
        runs = {name: seasons(cfg, profile, seeds, pool=pool) for name, profile in behaviours.items()}
        planner = list(pool.map(partial(planner_welfare, cfg), seeds, chunksize=8))

    print("equilibrium:", describe(solved["profile"]))
    print(f"largest gain left from deviating: Rs. {max(solved['regret'].values()):.1f}\n")
    rows = [{"behaviour": name, **average(season_rows)} for name, season_rows in runs.items()]
    best = sum(planner) / len(planner)
    rows.append({"behaviour": "planner", **{k: float("nan") for k in rows[0] if k != "behaviour"}})
    rows[-1]["welfare"] = best
    save_table(rows, "q2_baselines")
    print_table(rows[:-1], ["behaviour", "welfare", "travel_rate", "empty_berths", "turned_away",
                            "redundant_waitlist_time", "mean_utility"])
    print(f"planner welfare: {best:,.0f}\n")

    # the same seeds are used throughout, so welfare differences can be taken season by season
    def gap(a, b):
        diffs = [x["welfare"] - y["welfare"] for x, y in zip(runs[a], runs[b])]
        stats = average([{"d": d} for d in diffs])
        return stats["d"], stats["d_ci"]

    hedging, hedging_ci = gap("nobody hedges", "all hedge, release on time")
    late, late_ci = gap("all hedge, release on time", "all hedge, release late")
    split = [
        {"source": "hedging itself", "welfare_lost": hedging, "ci": hedging_ci},
        {"source": "late release", "welfare_lost": late, "ci": late_ci},
    ]
    for row in split:
        row["share_of_loss"] = row["welfare_lost"] / (hedging + late)
    save_table(split, "q2_split")
    print_table(split)

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.6))
    names = list(runs)
    bottom = [0.0] * len(names)
    for cause in EMPTY_CAUSES:
        heights = [100 * r[f"empty: {cause}"] for r in rows[:-1]]
        left.bar(names, heights, bottom=bottom, label=cause)
        bottom = [b + h for b, h in zip(bottom, heights)]
    left.set_ylabel("berths departing empty (%)")
    left.legend(frameon=False, fontsize=8)
    left.tick_params(axis="x", labelrotation=15)

    right.bar(names, [100 * r["turned_away"] for r in rows[:-1]], color="#8a8a8a")
    right.set_ylabel("travellers turned away (%)")
    right.tick_params(axis="x", labelrotation=15)
    save_figure(fig, "q2_decomposition")


if __name__ == "__main__":
    main()
