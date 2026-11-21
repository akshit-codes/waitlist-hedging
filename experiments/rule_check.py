"""Do the proposal's rules (Eqs. 2 and 3) match what the simulation shows?

The rules treat each train's confirmation chance as a fixed number, unrelated
to the other trains. In the market those chances come out of everybody's
bookings. Three checks, each at five levels of hedging:

  1. Are the chances fixed?       Confirmation rate per train as more people hedge.
  2. Are the trains independent?  For travellers waitlisted on two trains, how often
                                  both confirm, against the product of the two rates.
  3. Do the rules decide well?    The gain they predict from an extra ticket, and
                                  from releasing on time, against the simulated gain.

Chances counted over the whole market would be biased: a ticket its holder
cancels early never shows whether it would have confirmed, and the tickets of
people who do not hedge sit deeper in the queue than a hedger's. So the chances
are measured on a small group of travellers who hedge and are made to hold
every ticket until the chart, while everybody else plays as usual. The
market-wide rate is saved next to it for comparison.

The third check feeds the rules four sets of chances: the flat 25% of the
proposal, the measured rate of each train, the measured rate by train and
queue position, and the last with the independence assumption replaced by the
measured joint chances.
"""

import random
from statistics import mean

import matplotlib.pyplot as plt

from hedging.equilibrium import FOCAL_SHARE
from hedging.market import Season, make_travellers
from hedging.rules import Beliefs, Probe, fit, observe, observe_pairs
from hedging.simulate import assign, seasons, workers
from hedging.strategies import Fixed

from .common import TYPE_COLOURS, print_table, save_figure, save_table, setup

SHARES = [0.0, 0.25, 0.5, 0.75, 1.0]
LEVELS = ["flat 25%", "per train", "per train and position", "joint chances"]
PLOTTED = ["flat 25%", "per train and position", "joint chances"]


def mixed(cfg, share, hedge="hedge2-timely"):
    return {kind: {"single": 1 - share, hedge: share} for kind in cfg.type_names}


def focal_group(cfg, seed):
    picker = random.Random(f"focal-{seed}")
    return set(picker.sample(range(cfg.count), max(1, int(FOCAL_SHARE * cfg.count))))


def play(cfg, profile, seed, focal, strategy):
    """One season in which the focal group plays `strategy` and the rest play the profile."""
    travellers = make_travellers(cfg, seed)
    assign(cfg, travellers, profile, seed)
    for me in travellers:
        if me.id in focal:
            me.strategy = strategy
    return Season(cfg, travellers, seed).run()


def held_to_chart(args):
    """What happens to the tickets of a focal group that hedges and never releases."""
    cfg, profile, seed = args
    focal = focal_group(cfg, seed)
    season = play(cfg, profile, seed, focal, Fixed(2, timely=False))
    return observe(season, focal), observe_pairs(season, focal)


def probe_gains(args):
    """Replay one season with a focal group on each strategy and compare.

    For every focal traveller: the simulated gain from hedging and from
    releasing on time, and what Eqs. 2 and 3 predicted under each belief set.
    """
    cfg, profile, seed, beliefs = args
    focal = focal_group(cfg, seed)
    variants = {"single": Fixed(1), "timely": Fixed(2, timely=True), "late": Fixed(2, timely=False)}

    result = {}
    for label, strategy in variants.items():
        season = play(cfg, profile, seed, focal, Probe(strategy, beliefs))
        result[label] = {me.id: (season.utility(me), me)
                         for me in season.travellers if me.id in focal}

    out = []
    for i in focal:
        me = result["timely"][i][1]
        out.append({
            "type": me.type,
            "hedge_gain": result["timely"][i][0] - result["single"][i][0],
            "hedge_predicted": me.notes.get("hedge", {}),
            "release_gain": result["timely"][i][0] - result["late"][i][0],
            "release_predicted": me.notes.get("release", {}),
            "release_decisions": me.notes.get("release decisions", 0),
        })
    return out


def confirm_rate(observations, train):
    mine = [o for o in observations if o[0] == train]
    return sum(o[2] != "never" for o in mine) / max(len(mine), 1)


def independence(share, kind, pairs):
    """How often two waitlisted tickets of one traveller both confirm."""
    first = mean(p[3] for p in pairs)
    second = mean(p[4] for p in pairs)
    return {"hedging_share": share, "type": kind, "pairs": len(pairs),
            "better ticket confirms": first, "backup confirms": second,
            "both confirm": mean(p[3] and p[4] for p in pairs),
            "if independent": first * second}


def main():
    cfg, args = setup(__doc__)
    seeds = range(args.seasons)
    rates, joint, decisions = [], [], []

    with workers() as pool:
        for share in SHARES:
            profile = mixed(cfg, share)
            market_wide = [o for s in seasons(cfg, profile, seeds, observe, pool) for o in s]
            held = list(pool.map(held_to_chart, [(cfg, profile, s) for s in seeds], chunksize=4))
            seen = [o for obs, _ in held for o in obs]
            pairs = [p for _, both in held for p in both]
            beliefs = {
                "flat 25%": Beliefs(),
                "per train": fit(seen, by_position=False),
                "per train and position": fit(seen),
                "joint chances": fit(seen, pairs=pairs),
            }

            # 1. confirmation rate of each train at this level of hedging
            row = {"hedging_share": share}
            for k, train in enumerate(cfg.trains):
                row[train.name] = confirm_rate(seen, k)
                row[f"{train.name}, whole market"] = confirm_rate(market_wide, k)
            rates.append(row)

            # 2. independence, for everybody and for each type with enough pairs
            joint.append(independence(share, "all", pairs))
            for kind in cfg.type_names:
                mine = [p for p in pairs if p[0] == kind]
                if len(mine) >= 200:
                    joint.append(independence(share, kind, mine))

            # 3. predicted against simulated gains, by type
            jobs = [(cfg, profile, s, beliefs) for s in seeds]
            people = [p for batch in pool.map(probe_gains, jobs, chunksize=4) for p in batch]
            for kind in cfg.type_names:
                mine = [p for p in people if p["type"] == kind]
                asked = sum(p["release_decisions"] for p in mine)
                entry = {
                    "hedging_share": share,
                    "type": kind,
                    "hedge_gain simulated": mean(p["hedge_gain"] for p in mine),
                    "release_gain simulated": sum(p["release_gain"] for p in mine) / max(asked, 1),
                }
                for level in LEVELS:
                    entry[f"hedge_gain {level}"] = mean(
                        p["hedge_predicted"].get(level, 0.0) for p in mine)
                    entry[f"release_gain {level}"] = sum(
                        p["release_predicted"].get(level, 0.0) for p in mine) / max(asked, 1)
                decisions.append(entry)

    save_table(rates, "rule_check_rates")
    save_table(joint, "rule_check_independence")
    save_table(decisions, "rule_check_decisions")

    print("1. Confirmation rate of a hedger's waitlisted ticket held to the chart,")
    print("   by share of travellers who hedge")
    print_table(rates)
    print("\n2. Two waitlisted tickets held by the same traveller")
    print_table(joint)
    print("\n3a. Gain from a second ticket (Rs. per traveller): simulated and Eq. 2")
    print_table(decisions, ["hedging_share", "type", "hedge_gain simulated",
                            *[f"hedge_gain {level}" for level in LEVELS]])
    print("\n3b. Gain from releasing a redundant ticket on time (Rs. per ticket): simulated and Eq. 3")
    print_table(decisions, ["hedging_share", "type", "release_gain simulated",
                            *[f"release_gain {level}" for level in LEVELS]])

    summary = []
    for rule, key in (("Eq. 2 (hedge)", "hedge_gain"), ("Eq. 3 (release)", "release_gain")):
        for level in LEVELS:
            errors = [abs(d[f"{key} {level}"] - d[f"{key} simulated"]) for d in decisions]
            agree = [(d[f"{key} {level}"] > 0) == (d[f"{key} simulated"] > 0) for d in decisions]
            summary.append({"rule": rule, "chances it is given": level,
                            "mean abs. error (Rs.)": mean(errors),
                            "same decision": f"{sum(agree)} of {len(agree)}"})
    save_table(summary, "rule_check_summary")
    print("\nHow well the rules predict the simulated gains")
    print_table(summary)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    for train in cfg.trains:
        axes[0].plot(SHARES, [r[train.name] for r in rates], "o-", label=train.name)
    axes[0].axhline(0.25, color="grey", ls=":", lw=1)
    axes[0].set_xlabel("share of travellers who hedge")
    axes[0].set_ylabel("chance a hedger's waitlisted ticket confirms")
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1))
    for ax, level in zip(axes[1:], PLOTTED):
        for kind in cfg.type_names:
            mine = [d for d in decisions if d["type"] == kind]
            ax.scatter([d["hedge_gain simulated"] for d in mine],
                       [d[f"hedge_gain {level}"] for d in mine],
                       color=TYPE_COLOURS.get(kind), label=kind)
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], color="grey", lw=1)
        ax.set_title(f"Eq. 2 with {level}" + ("" if "chances" in level else " chances"), fontsize=9)
        ax.set_xlabel("simulated gain from hedging (Rs.)")
    axes[1].set_ylabel("predicted gain (Rs.)")
    axes[1].legend(frameon=False, fontsize=8)
    save_figure(fig, "rule_check")


if __name__ == "__main__":
    main()
