"""Model parameters, loaded from config.yaml."""

from dataclasses import dataclass, replace
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass(frozen=True)
class TrainSpec:
    name: str
    departs: float      # hour of the travel day
    capacity: int
    fare: float


@dataclass(frozen=True)
class TravellerType:
    name: str
    share: float
    values: tuple       # value of travelling on each train, in train order


@dataclass(frozen=True)
class Config:
    trains: tuple
    types: tuple

    # railway policy: the three levers the project compares
    waitlist_cap: float
    release_refund: float
    chart_hours: float

    waitlist_cancel: float
    confirmed_flat: float
    confirmed_72h: float
    confirmed_24h: float
    no_refund_hours: float

    advance_days: int
    last_booking_hours: float
    quota_berths: int
    resale_prob: float
    empty_berth_cost: float

    count: int
    plan_change_prob: float
    hassle: float
    outside_option: float
    value_noise: float

    def with_policy(self, cap=None, refund=None, chart=None):
        """Same market under different railway rules."""
        return replace(
            self,
            waitlist_cap=self.waitlist_cap if cap is None else cap,
            release_refund=self.release_refund if refund is None else refund,
            chart_hours=self.chart_hours if chart is None else chart,
        )

    @property
    def type_names(self):
        return [t.name for t in self.types]


def load(path=DEFAULT_PATH):
    raw = yaml.safe_load(Path(path).read_text())
    people = dict(raw["travellers"])
    types = tuple(
        TravellerType(t["name"], t["share"], tuple(t["values"])) for t in people.pop("types")
    )
    trains = tuple(TrainSpec(**t) for t in raw["trains"])
    for t in types:
        if len(t.values) != len(trains):
            raise ValueError(f"type {t.name} needs one value per train")
    return Config(
        trains=trains, types=types, **raw["policy"], **raw["charges"], **raw["market"], **people
    )
