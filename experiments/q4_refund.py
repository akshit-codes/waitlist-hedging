"""Q4. How large must an early-release refund be, and what does it cost?

The refund is paid when a traveller cancels a redundant waitlisted ticket
herself instead of leaving it to the chart. Holding such a ticket is only
tempting when it is unlikely to confirm, which depends on how long waitlists
are allowed to get, so the sweep is repeated under three caps: the 25% of
June 2025, today's 60%, and none. For each refund
we recompute the travellers' response.
"""

import matplotlib.pyplot as plt

from hedging import equilibrium
from hedging.simulate import evaluate, workers

from .common import print_table, save_figure, save_table, setup

REFUNDS = [0, 10, 20, 30, 40, 60]
CAPS = [0.25, 0.60, 1.00]


def late_share(profile, cfg):
    """Of the travellers who hedge, the share who leave redundant tickets to the chart."""
    hedging = late = 0.0
    for kind in cfg.types:
        for name, share in profile[kind.name].items():
            if name != "single":
                hedging += kind.share * share
                late += kind.share * share * name.endswith("late")
    return late / hedging if hedging else 0.0


def main():
    cfg, args = setup(__doc__)
    seeds = range(args.seasons)
    rows = []
    with workers() as pool:
        for cap in CAPS:
            for refund in REFUNDS:
                market = cfg.with_policy(cap=cap, refund=refund)
                solved = equilibrium.solve(market, pool)
                outcome = evaluate(market, solved["profile"], seeds, pool)
                rows.append({
                    "cap": cap,
                    "refund": refund,
                    "late_release_share": late_share(solved["profile"], cfg),
                    **outcome,
                })

    save_table(rows, "q4_refund")
    print_table(rows, ["cap", "refund", "late_release_share", "redundant_waitlist_time",
                       "empty_berths", "turned_away", "welfare", "welfare_ci", "refunds_paid",
                       "railway_objective", "railway_income"])

    panels = [("late_release_share", "hedgers who release late (%)", 100),
              ("welfare", "welfare (Rs.)", 1),
              ("refunds_paid", "refunds paid per season (Rs.)", 1)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    for ax, (key, label, scale) in zip(axes, panels):
        for cap in CAPS:
            mine = [r for r in rows if r["cap"] == cap]
            ax.plot(REFUNDS, [scale * r[key] for r in mine], "o-", label=f"cap {cap:.0%}")
        ax.set_xlabel("refund for on-time release (Rs.)")
        ax.set_ylabel(label)
    axes[0].legend(frameon=False)
    save_figure(fig, "q4_refund")


if __name__ == "__main__":
    main()
