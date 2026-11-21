"""Do the main findings survive other values of the parameters we had to assume?

One parameter is moved at a time, down and up from its default. For each
market we report how many travellers hedge in equilibrium, what that costs in
welfare compared with nobody hedging, and how the loss in the all-hedge
baselines splits between hedging itself and late release.
"""

from dataclasses import replace

from hedging import equilibrium
from hedging.metrics import average
from hedging.simulate import seasons, workers

from .common import print_table, save_table, setup

VARIATIONS = {
    "count": (300, 360),
    "plan_change_prob": (0.08, 0.16),
    "quota_berths": (0, 6),
    "resale_prob": (0.0, 0.6),
    "hassle": (20, 80),
    "outside_option": (200, 600),
    "value_noise": (75, 300),
    "empty_berth_cost": (0, 1000),
}


def paired_loss(better, worse):
    """Mean welfare difference between two behaviours run on the same seeds."""
    stats = average([{"d": b["welfare"] - w["welfare"]} for b, w in zip(better, worse)])
    return stats["d"], stats["d_ci"]


def study(market, seeds, pool):
    most = len(market.trains)
    solved = equilibrium.solve(market, pool)
    nobody = seasons(market, "single", seeds, pool=pool)
    settled = seasons(market, solved["profile"], seeds, pool=pool)
    on_time = seasons(market, f"hedge{most}-timely", seeds, pool=pool)
    late = seasons(market, f"hedge{most}-late", seeds, pool=pool)

    loss, loss_ci = paired_loss(nobody, settled)
    from_hedging, _ = paired_loss(nobody, on_time)
    from_lateness, _ = paired_loss(on_time, late)
    return {
        "hedgers_in_equilibrium": average(settled)["hedger_share"],
        "welfare_loss_in_equilibrium": loss,
        "loss_ci": loss_ci,
        "late_release_share_of_all_hedge_loss": from_lateness / (from_hedging + from_lateness),
    }


def main():
    cfg, args = setup(__doc__)
    seeds = range(args.seasons)
    with workers() as pool:
        rows = [{"parameter": "defaults", "value": "", **study(cfg, seeds, pool)}]
        for name, values in VARIATIONS.items():
            for value in values:
                market = replace(cfg, **{name: value})
                rows.append({"parameter": name, "value": value, **study(market, seeds, pool)})
    save_table(rows, "sensitivity")
    print_table(rows)


if __name__ == "__main__":
    main()
