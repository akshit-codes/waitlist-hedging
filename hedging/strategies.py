"""How a traveller books and when she lets go of tickets she no longer needs.

Every strategy books the same way. The traveller first takes the best train
that still has a free berth, as someone buying a single ticket would. She then
joins the waitlists of trains she likes enough better to pay for a switch, best
first, for as long as the strategy wants another ticket. If no train has a free
berth she waitlists her favourites instead. Strategies differ in how many
tickets they hold and in whether they cancel a redundant waitlisted ticket or
leave it to the chart.
"""


class Strategy:
    name = ""

    def book(self, me, market):
        trains = market.ranking(me)
        base = next((t for t in trains if t.free_berths > 0), None)
        if base:
            market.request(me, base)
        for train in trains:
            if train is base:
                break       # everything after this she likes less than the berth she has
            if not train.waitlist_open:
                continue
            if base and market.surplus(me, train) <= market.keep_value(me, base):
                continue    # not worth the charge for giving the base berth up
            if self.wants(me, market, train, base):
                market.request(me, train)

    def wants(self, me, market, train, base):
        """Add a waitlisted ticket on `train` to what she already holds?"""
        raise NotImplementedError

    def release(self, me, market, ticket):
        """Cancel a waitlisted ticket that has just become redundant?"""
        raise NotImplementedError


class Fixed(Strategy):
    """Hold at most `limit` tickets; release redundant ones at once, or never."""

    def __init__(self, limit, timely=True):
        self.limit = limit
        self.timely = timely
        if limit == 1:
            self.name = "single"
        else:
            self.name = f"hedge{limit}-{'timely' if timely else 'late'}"

    def wants(self, me, market, train, base):
        return len(me.tickets) < self.limit

    def release(self, me, market, ticket):
        return self.timely


def menu(n_trains=3):
    """The strategies travellers choose between in the equilibrium analysis."""
    options = [Fixed(1)]
    for limit in range(2, n_trains + 1):
        options += [Fixed(limit, timely=True), Fixed(limit, timely=False)]
    return {s.name: s for s in options}
