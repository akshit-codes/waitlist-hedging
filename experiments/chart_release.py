"""How much of the harm from hedging depends on berths released at charting?

Unused quota berths reach the waitlist only when the chart is prepared. A
hedger whose preferred train confirms that late has often passed her backup
train's chart already, so the backup berth is lost. Here the number of berths
held back per train goes from none to six, with travellers responding in
equilibrium each time.
"""

from dataclasses import replace

import matplotlib.pyplot as plt

from hedging import equilibrium
from hedging.metrics import average
from hedging.simulate import seasons, workers

from .common import print_table, save_figure, save_table, setup

QUOTAS = [0, 1, 2, 3, 4, 6]


def main():
    cfg, args = setup(__doc__)
    seeds = range(args.seasons)
    rows = []
    with workers() as pool:
        for quota in QUOTAS:
            market = replace(cfg, quota_berths=quota)
            solved = equilibrium.solve(market, pool)
            nobody = seasons(market, "single", seeds, pool=pool)
            settled = seasons(market, solved["profile"], seeds, pool=pool)
            loss = average([{"d": a["welfare"] - b["welfare"]} for a, b in zip(nobody, settled)])
            outcome = average(settled)
            rows.append({
                "berths_released_at_chart": quota,
                "hedgers": outcome["hedger_share"],
                "welfare_loss_from_hedging": loss["d"],
                "loss_ci": loss["d_ci"],
                "switched_after_chart": outcome["empty: switched after chart"],
                "waitlist_confirm_rate": outcome["waitlist_confirm_rate"],
            })

    save_table(rows, "chart_release")
    print_table(rows)

    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.3))
    left.errorbar(QUOTAS, [r["welfare_loss_from_hedging"] for r in rows],
                  yerr=[r["loss_ci"] for r in rows], fmt="o-", color="#1f4e79")
    left.axhline(0, color="black", lw=0.8)
    left.set_ylabel("welfare lost to hedging (Rs. per season)")
    right.plot(QUOTAS, [100 * r["switched_after_chart"] for r in rows], "o-", color="#c0504d")
    right.set_ylabel("berths abandoned after their chart (%)")
    for ax in (left, right):
        ax.set_xlabel("berths per train released at charting")
    save_figure(fig, "chart_release")


if __name__ == "__main__":
    main()
