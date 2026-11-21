from dataclasses import replace

import pytest

from hedging import config, equilibrium, learning
from hedging.config import TravellerType
from hedging.planner import planner_welfare

from .helpers import market


def test_planner_seats_the_travellers_who_value_it_most():
    # two berths on each train, six identical travellers, nobody changes plans
    cfg = market(
        types=(TravellerType("anyone", 1.0, (1000, 2000)),),
        count=6, value_noise=0, plan_change_prob=0, outside_option=400,
    )
    # two ride b, two ride a, two stay home
    assert planner_welfare(cfg, seed=0) == pytest.approx(2 * 2000 + 2 * 1000 + 2 * 400)


def test_regret_is_zero_at_a_best_response():
    payoffs = {"single": (100.0, 0.0), "hedge2-timely": (90.0, 1.0)}
    assert equilibrium.regret({"single": 1.0, "hedge2-timely": 0.0}, payoffs) == 0
    assert equilibrium.regret({"single": 0.0, "hedge2-timely": 1.0}, payoffs) == 10


def test_solver_returns_a_mix_for_every_type():
    cfg = replace(config.load(), count=120)
    solved = equilibrium.solve(cfg, rounds=3, seeds_per_round=4)
    for kind in cfg.type_names:
        assert sum(solved["profile"][kind].values()) == pytest.approx(1.0)
        assert solved["regret"][kind] >= -1e-6


def test_qtable_averages_its_rewards():
    table = learning.QTable()
    for reward in (100, 200, 300):
        table.update("s", "a", reward)
    assert table.value["s", "a"] == pytest.approx(200)
    assert table.best("s", ["a", "b"]) == "a"


def test_learners_run_and_record_decisions():
    cfg = config.load()
    tables, log = learning.train(cfg, seasons=20, log_every=10)
    assert len(log) == 2
    assert any(table.visits for table in tables.values())
