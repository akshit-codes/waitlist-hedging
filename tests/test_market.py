from hedging.metrics import summarise
from hedging.strategies import Fixed

from .helpers import A, B, market, person, run, ticket

WANTS_A = (2000, 1500)
WANTS_B = (1500, 2000)


def test_berths_then_waitlist_then_refusal():
    cfg = market(waitlist_cap=0.5)      # one waitlist slot per train
    only_a = (2000, 0)                  # would rather stay home than take train b
    people = [person(i, only_a, arrives=i) for i in range(4)]
    run(cfg, people)

    assert [ticket(p, A).position for p in people[:3]] == [0, 0, 1]
    assert people[3].turned_away                        # a is closed to her and b is not worth it


def test_single_ticket_traveller_takes_the_best_free_berth():
    cfg = market(waitlist_cap=0.0)
    people = [person(i, WANTS_A, arrives=i) for i in range(5)]
    run(cfg, people)

    assert [p.tickets[0].train.name for p in people[:4]] == ["a", "a", "b", "b"]
    assert people[4].turned_away


def test_cancellation_before_chart_promotes_first_in_queue():
    cfg = market()
    people = [person(i, (2000, 0), arrives=i) for i in range(4)]
    people[0].plan_change_at = 100
    season = run(cfg, people)

    assert people[0].paid == 180                        # flat charge, more than 72 h ahead
    assert ticket(people[2], A).confirmed_at == 100     # first in the queue gets the berth
    assert people[2].boarded.name == "a"
    assert ticket(people[3], A).ending == "lapsed"      # still waiting when the chart came out
    assert people[3].paid == 60 and people[3].releases == 0
    assert season.utility(people[3]) == cfg.outside_option - 60


def test_cancellation_charge_rises_towards_departure():
    cfg = market()
    people = [person(0, (2000, 0), 1, plan_change_at=700),      # 34 h before a leaves
              person(1, (2000, 0), 2, plan_change_at=720)]      # 14 h before
    run(cfg, people)

    assert people[0].paid == 0.25 * 800
    assert people[1].paid == 0.50 * 800


def test_berth_vacated_after_chart_is_lost_to_the_waitlist():
    cfg = market()
    people = [person(i, (2000, 0), arrives=i) for i in range(3)]
    people[0].plan_change_at = 730                      # chart for a was at 726
    season = run(cfg, people)

    assert people[0].paid == 800                        # nothing is refunded this late
    assert people[2].boarded is None
    assert season.trains[A].empty["late cancellation"] == 1


def test_refund_is_still_possible_between_chart_and_the_last_eight_hours():
    cfg = market(chart_hours=10)                        # a's chart is now at 724
    people = [person(i, (2000, 0), arrives=i) for i in range(3)]
    people[0].plan_change_at = 725                      # 9 h before a leaves
    season = run(cfg, people)

    assert people[0].paid == 0.50 * 800
    assert people[2].boarded is None                    # the waitlist is gone all the same
    assert season.trains[A].empty["late cancellation"] == 1


def test_berth_vacated_after_chart_can_be_resold():
    cfg = market(resale_prob=1.0)
    people = [person(0, (2000, 0), 1, plan_change_at=730)]
    season = run(cfg, people)

    assert season.trains[A].resold == 2                 # her berth and the one never sold
    assert sum(season.trains[A].empty.values()) == 0


def test_upgrade_before_chart_hands_the_backup_to_the_next_traveller():
    cfg = market()
    people = [
        person(0, WANTS_B, 1, plan_change_at=100),
        person(1, WANTS_B, 2),
        person(2, (1500, 2600), 3, Fixed(2)),           # hedger: berth on a, waitlisted on b
        person(3, (2000, 0), 4),
        person(4, (2000, 0), 5),                        # a is full by now, she waits
    ]
    season = run(cfg, people)
    hedger = people[2]

    assert hedger.boarded.name == "b"
    assert ticket(hedger, A).ending == "released"
    assert hedger.paid == 800 + 180 and hedger.releases == 1
    assert people[4].boarded.name == "a"                # she got the berth the hedger let go
    assert season.utility(hedger) == 2600 - 800 - 180 - cfg.hassle


def test_switch_after_chart_leaves_the_backup_berth_empty():
    """The 6 pm / 10 pm story of the proposal."""
    cfg = market()
    people = [
        person(0, WANTS_B, 1, plan_change_at=730),      # after a's chart, before b's
        person(1, WANTS_B, 2),
        person(2, (1500, 2600), 3, Fixed(2)),
    ]
    season = run(cfg, people)
    hedger = people[2]

    assert ticket(hedger, B).confirmed_at == 730
    assert hedger.boarded.name == "b"
    assert hedger.paid == 1600 and hedger.releases == 0  # both fares, and nothing left to cancel
    assert season.trains[A].empty["switched after chart"] == 1


def test_upgrade_stops_being_worth_it_when_the_charge_rises():
    # b is worth 900 net to her and a 700. Switching costs the charge on a: 180 at
    # first, 200 from 72 h before a leaves (t = 662). From then on b is redundant.
    def play(strategy, refund=0):
        cfg = market(release_refund=refund)
        people = [person(0, WANTS_B, 1), person(1, WANTS_B, 2), person(2, (1500, 1700), 3, strategy)]
        run(cfg, people)
        return people[2]

    on_time = play(Fixed(2, timely=True))
    assert ticket(on_time, B).redundant_since == 662
    assert ticket(on_time, B).ending == "released"
    assert on_time.paid == 800 + 60 and on_time.releases == 1

    late = play(Fixed(2, timely=False))
    assert ticket(late, B).ending == "lapsed"           # left for the chart to cancel
    assert late.paid == 800 + 60 and late.releases == 0

    rewarded = play(Fixed(2, timely=True), refund=30)
    assert rewarded.refunds == 30


def test_quota_berth_goes_to_the_waitlist_at_charting():
    cfg = market(quota_berths=1)                        # one berth per train on sale until the chart
    people = [person(i, (2000, 0), arrives=i) for i in range(3)]
    run(cfg, people)

    assert ticket(people[1], A).position == 1
    assert ticket(people[1], A).at_chart
    assert people[1].boarded.name == "a"
    assert ticket(people[2], A).ending == "lapsed"


def test_redundant_ticket_held_until_chart_wastes_a_berth():
    def play(strategy):
        cfg = market(quota_berths=1)
        people = [
            person(0, (2000, 0), 1, plan_change_at=100),
            person(1, (0, 2000), 2),
            person(2, (2000, 1500), 3, strategy),       # both trains full: waitlisted on both
        ]
        return run(cfg, people), people[2]

    season, late = play(Fixed(2, timely=False))
    assert ticket(late, A).confirmed_at == 100          # from then on her b ticket is redundant
    assert ticket(late, B).at_chart                     # ...but she kept it and it confirmed
    assert late.boarded.name == "a"
    assert late.paid == 1600
    assert season.trains[B].empty["redundant ticket held"] == 1

    season, on_time = play(Fixed(2, timely=True))
    assert on_time.paid == 800 + 60
    assert season.trains[B].empty["redundant ticket held"] == 0


def test_every_berth_is_accounted_for():
    from hedging import config
    from hedging.simulate import play

    cfg = config.load()
    for profile in ("single", "hedge3-late"):
        season = play(cfg, profile, seed=3)
        for train in season.trains:
            boarded = sum(1 for p in season.travellers if p.boarded is train)
            assert boarded + train.resold + sum(train.empty.values()) == train.capacity
        row = summarise(season)
        assert 0 <= row["empty_berths"] <= 1
