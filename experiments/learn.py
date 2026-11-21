"""Do travellers who learn from experience end up where the equilibrium says?

Three populations face the same market:

  equilibrium     the type-level mix found by best-response iteration
  rule-followers  apply Eqs. 2 and 3 with the true confirmation chances
  learners        know nothing at first and learn over repeated seasons

The fixed baselines (nobody hedges, everybody hedges and releases on time or
late) are played on the same seasons for comparison.

The learners' release rule is also printed by queue position, under today's cap
and under the 25% cap of June 2025, to see whether they discover Eq. 3's logic by themselves:
release what is likely to confirm, sit on what is not.
"""

import matplotlib.pyplot as plt

from hedging import equilibrium, learning
from hedging.metrics import average
from hedging.rules import BUCKET_EDGES, RuleBased, self_consistent_beliefs
from hedging.simulate import evaluate, workers

from .common import print_table, save_figure, save_table, setup

TRAINING_SEASONS = 8000
TRAINING_RUNS = 3
COLUMNS = ["hedger_share", "tickets_per_traveller", "redundant_waitlist_time", "empty_berths",
           "turned_away", "travel_rate", "mean_utility", "welfare", "welfare_ci"]


def queue_label(place):
    edges = (0, *BUCKET_EDGES)
    if place == len(BUCKET_EDGES):
        return f"{edges[-1] + 1}+"
    return f"{edges[place] + 1}-{edges[place + 1]}"


def release_rule(tables, cap):
    """What the learners do with a redundant ticket, by where it stands in the queue."""
    rows = []
    for kind, table in tables.items():
        for place in range(len(BUCKET_EDGES) + 1):
            for soon in (False, True):
                state = ("release", place, soon)
                visits = sum(table.visits[state, a] for a in (learning.RELEASE, learning.HOLD))
                if visits < 200:
                    continue    # too rarely met to have learned anything
                edge = table.value[state, learning.RELEASE] - table.value[state, learning.HOLD]
                rows.append({
                    "cap": cap,
                    "type": kind,
                    "queue position": queue_label(place),
                    "chart within 24 h": "yes" if soon else "no",
                    "times met": visits,
                    "learned": table.best(state, [learning.RELEASE, learning.HOLD]),
                    "value of releasing (Rs.)": edge,
                })
    return rows


def main():
    cfg, args = setup(__doc__)
    seeds = range(10**6, 10**6 + args.seasons)      # seasons the learners never trained on

    tight = cfg.with_policy(cap=0.25)
    with workers() as pool:
        # independent training runs, to see whether learning ends in the same place each time
        training = [pool.submit(learning.train, cfg, TRAINING_SEASONS, seed=run * 10**5)
                    for run in range(TRAINING_RUNS)]
        tight_training = pool.submit(learning.train, tight, TRAINING_SEASONS)

        solved = equilibrium.solve(cfg, pool)
        beliefs = self_consistent_beliefs(cfg, range(300), pool)
        rows = [
            {"population": "equilibrium", **evaluate(cfg, solved["profile"], seeds, pool)},
            {"population": "rule-followers", **evaluate(cfg, RuleBased(beliefs), seeds, pool)},
        ]
        trained = [job.result() for job in training]
        tight_tables, _ = tight_training.result()
        for run, (run_tables, _) in enumerate(trained, start=1):
            played = [learning.behaviour(learning.play(cfg, run_tables, 0.0, s)) for s in seeds]
            rows.append({"population": f"learners, run {run}", **average(played)})
        most = len(cfg.trains)
        for label, strategy in (("all hedge, release on time", f"hedge{most}-timely"),
                                ("all hedge, release late", f"hedge{most}-late"),
                                ("nobody hedges", "single")):
            rows.append({"population": label, **evaluate(cfg, strategy, seeds, pool)})
    tables, log = trained[0]
    save_table(rows, "learning_comparison")
    save_table(log, "learning_curve")
    print_table(rows, ["population", *COLUMNS])

    rule = release_rule(tables, cfg.waitlist_cap) + release_rule(tight_tables, 0.25)
    save_table(rule, "learning_release_rule")
    print("\nWhat the learners do with a redundant waitlisted ticket")
    print_table(rule)

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.4))
    season = [r["season"] for r in log]
    left.plot(season, [100 * r["chose_to_hedge"] for r in log],
              label="hedge when the first choice is full")
    left.plot(season, [100 * r["released_on_time"] for r in log],
              label="release a redundant ticket on time")
    left.set_xlabel("season")
    left.set_ylabel("share of decisions (%)")
    left.legend(frameon=False, fontsize=8)

    right.plot(season, [r["welfare"] for r in log], label="learners")
    right.axhline(rows[0]["welfare"], color="black", ls="--", lw=1, label="equilibrium")
    right.axhline(rows[-1]["welfare"], color="grey", ls=":", lw=1, label="nobody hedges")
    right.set_xlabel("season")
    right.set_ylabel("welfare (Rs.)")
    right.legend(frameon=False, fontsize=8)
    save_figure(fig, "learning")


if __name__ == "__main__":
    main()
