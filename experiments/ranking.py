"""The railway's problem: which rules give the most welfare once travellers respond?

The railway moves first and travellers answer with their equilibrium, so we
solve that equilibrium for every combination of cap, refund and chart time on
a grid and pick the best. The winner, and each single change to today's rules,
is then scored on seasons that played no part in the search. Welfare is
compared with today's rules season by season, and the railway's own money is
reported next to it, because a refund is a transfer and not a welfare loss.
"""

from itertools import product

import matplotlib.pyplot as plt

from hedging import equilibrium
from hedging.metrics import average
from hedging.simulate import evaluate, seasons, workers

from .common import print_table, save_figure, save_table, setup

CAPS = [0.10, 0.25, 0.60, 1.00]
REFUNDS = [0, 30, 60]
CHARTS = [4, 8, 10]
SEARCH_SEASONS = 200


def name(rules, today):
    """Say only what differs from today's rules."""
    cap, refund, chart = rules
    parts = [f"cap {cap:.0%}" if cap < 1 else "no cap", f"refund Rs. {refund}", f"chart {chart} h"]
    changed = [part for part, new, old in zip(parts, rules, today) if new != old]
    return ", ".join(changed) or "today's rules"


def main():
    cfg, args = setup(__doc__)
    today = (cfg.waitlist_cap, cfg.release_refund, cfg.chart_hours)
    search = range(SEARCH_SEASONS)
    fresh = range(10**6, 10**6 + args.seasons)

    with workers() as pool:
        grid, profiles = [], {}
        for rules in product(CAPS, REFUNDS, CHARTS):
            cap, refund, chart = rules
            market = cfg.with_policy(cap=cap, refund=refund, chart=chart)
            profiles[rules] = equilibrium.solve(market, pool)["profile"]
            outcome = evaluate(market, profiles[rules], search, pool)
            grid.append({"cap": cap, "refund": refund, "chart_hours": chart,
                         "welfare": outcome["welfare"], "refunds_paid": outcome["refunds_paid"],
                         "turned_away": outcome["turned_away"]})
        save_table(grid, "policy_grid")
        best = max(grid, key=lambda r: r["welfare"])
        best = (best["cap"], best["refund"], best["chart_hours"])

        # today's rules, every single-lever change to them, and the best of the grid
        one_lever = [r for r in profiles if sum(a != b for a, b in zip(r, today)) == 1]
        shown = [today, *one_lever] + ([best] if best not in one_lever else [])
        runs = {}
        for rules in shown:
            market = cfg.with_policy(cap=rules[0], refund=rules[1], chart=rules[2])
            runs[rules] = seasons(market, profiles[rules], fresh, pool=pool)

    rows = []
    for rules, season_rows in runs.items():
        diffs = average([
            {"welfare_gain": r["welfare"] - t["welfare"],
             "railway_objective_gain": r["railway_objective"] - t["railway_objective"],
             "railway_income_change": r["railway_income"] - t["railway_income"]}
            for r, t in zip(season_rows, runs[today])
        ])
        mean = average(season_rows)
        rows.append({
            "rule": name(rules, today) + (" (best of grid)" if rules == best else ""),
            "welfare_gain": diffs["welfare_gain"],
            "welfare_gain_ci": diffs["welfare_gain_ci"],
            "railway_objective_gain": diffs["railway_objective_gain"],
            "railway_income_change": diffs["railway_income_change"],
            "refunds_paid": mean["refunds_paid"],
            "empty_berths": mean["empty_berths"],
            "turned_away": mean["turned_away"],
        })
    rows[1:] = sorted(rows[1:], key=lambda r: -r["welfare_gain"])
    save_table(rows, "ranking")
    print_table(rows)

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bars = rows[1:][::-1]
    ax.barh([r["rule"] for r in bars], [r["welfare_gain"] for r in bars],
            xerr=[r["welfare_gain_ci"] for r in bars], color="#1f4e79")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("welfare gain over today's rules (Rs. per season)")
    save_figure(fig, "ranking")


if __name__ == "__main__":
    main()
