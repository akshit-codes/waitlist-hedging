"""Summary numbers for one season, and averages over many."""

from statistics import mean, stdev


EMPTY_CAUSES = ["unsold", "late cancellation", "switched after chart", "redundant ticket held"]


def summarise(season):
    cfg, people, trains = season.cfg, season.travellers, season.trains
    going = [p for p in people if not p.changed_plans]      # plans held, so they wanted to travel
    hedgers = [p for p in going if p.hedged]
    others = [p for p in going if not p.hedged]
    berths = sum(t.capacity for t in trains)
    empty = sum(sum(t.empty.values()) for t in trains)

    tickets = [t for p in people for t in p.tickets]
    waitlisted = [t for t in tickets if t.position > 0]
    waited = sum(_hours_waiting(t) for t in waitlisted)
    idle = sum(_hours_redundant(t) for t in waitlisted)

    travel_value = sum(p.values[p.boarded.index] for p in people if p.boarded)
    stay_value = cfg.outside_option * sum(1 for p in people if not p.boarded)
    fares = sum(t.fare * (sum(1 for p in people if p.boarded is t) + t.resold) for t in trains)

    welfare = travel_value + stay_value - cfg.empty_berth_cost * empty

    out = {
        "travel_rate": _share(going, lambda p: p.boarded),
        "travel_rate_hedgers": _share(hedgers, lambda p: p.boarded),
        "travel_rate_others": _share(others, lambda p: p.boarded),
        "hedger_share": len(hedgers) / max(len(going), 1),
        "tickets_per_traveller": len(tickets) / len(people),
        "turned_away": _share(people, lambda p: p.turned_away),
        "empty_berths": empty / berths,
        "redundant_waitlist_time": idle / waited if waited else 0.0,
        "waitlist_confirm_rate": _share(waitlisted, lambda t: t.confirmed_at is not None),
        # welfare is Eq. 4 without the refund term: refunds, fares and charges only move money
        "welfare": welfare,
        # Eq. 4 as the railway sees it: a refund is money out of its own pocket
        "railway_objective": welfare - season.refunds_paid,
        "mean_utility": mean(season.utility(p) for p in people),
        "refunds_paid": season.refunds_paid,
        "railway_income": fares + season.charges_collected - season.refunds_paid,
    }
    for cause in EMPTY_CAUSES:
        out[f"empty: {cause}"] = sum(t.empty[cause] for t in trains) / berths
    return out


def average(rows):
    """Mean and 95% half-width of every metric across seasons."""
    n = len(rows)
    out = {}
    for key in rows[0]:
        xs = [r[key] for r in rows]
        out[key] = mean(xs)
        out[key + "_ci"] = 1.96 * stdev(xs) / n**0.5 if n > 1 else 0.0
    return out


def _share(items, test):
    return sum(1 for x in items if test(x)) / len(items) if items else 0.0


def _hours_waiting(ticket):
    end = ticket.confirmed_at if ticket.confirmed_at is not None else ticket.closed_at
    return end - ticket.booked_at


def _hours_redundant(ticket):
    """Time a ticket sat in the waitlist after its holder stopped needing it."""
    if ticket.redundant_since is None:
        return 0.0
    end = ticket.confirmed_at if ticket.confirmed_at is not None else ticket.closed_at
    return max(0.0, end - ticket.redundant_since)
