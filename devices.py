"""Household devices beyond the base load: a heat pump (uncontrolled load) and an electric vehicle (controllable).

HEAT PUMP: air-source, space heating + domestic hot water; electric kW = heat demand / COP, from the outdoor
temperature T (column temp_c). Follows the when2heat method (Ruhnau, Hirth & Praktiknjo 2019, Scientific Data 6:189):
    space heating   q_sh  = H * max(0, T_limit - T_24h)       T_24h: trailing 24 h mean (building inertia)
    hot water       q_dhw = dhw_kwh_per_year / 8760 * daily shape (morning and evening draws, mean 1)
    COP(dT)         0.85 * (6.08 - 0.09 dT + 0.0005 dT^2)     ASHP regression, dT = T_sink - T, 0.85 real-world factor
    sink            radiators 40 - 1.0 T, floor heating 30 - 0.5 T, hot water 50 °C
Defaults: H = 0.22 kW/K, T_limit 15 °C -> ~11-12 MWh space heat per year at this location (~52,000 degree-hours),
2,000 kWh hot water (4 persons), radiators -> ~4,500 kWh electricity per year (SCOP ~3). Uncontrolled: the
agent cannot shift it, it is part of the load the battery has to serve.

ELECTRIC VEHICLE: a parametric daily mobility model, calibrated to the order of magnitude of MiD 2017 (Mobilität
in Deutschland) and KBA mileage (~12,000-13,000 km/year per car). Per calendar day at most one round trip:
    used?           Bernoulli(p_use)                          weekday / weekend parameters differ
    departure hour  round(Normal(dep_mean, dep_sd)) in [4, 12]
    arrival hour    round(Normal(arr_mean, arr_sd)) in [departure + 1, 23]
    distance        LogNormal(log(km_median), km_sigma), at most max_km
The car is therefore always home at midnight, so episodes (which start at midnight) never start mid-trip.
Each date draws from its own generator seeded by (seed, date), so a date has the same trip in every slice of the
data (train / val / test, a week, the whole year) and the schedule is identical for every policy and the oracle.

Charging semantics (enforced by the env and the oracle alike):
    - the agent sets the wallbox power while the car is home (0 .. max_power_kw, grid side, efficiency on the way in)
    - at departure the car must hold target_soc; a shortfall is bought on the road at public_price (EUR/kWh) and
      tops the car up to target_soc (the user charged publicly). This turns the deadline into a cost.
    - the trip's energy (km * kwh_per_km) leaves the car at arrival. Trips are capped at the car's range from
      target_soc (max_km is lowered to target_soc * capacity / kwh_per_km for a small car: its household does
      not drive further on one charge), so the car never runs empty.
    - no vehicle-to-home: the EV only charges.
The planned departure time is assumed known (the user enters it in the charging app); the arrival time and the
trip energy are not known in advance: they are the unpredictable part.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
import numpy as np
import pandas as pd

# relative hot-water draw by hour of day (morning shower, evening), normalised to mean 1 below
_DHW_SHAPE = np.array([0.3, 0.2, 0.2, 0.2, 0.3, 0.8, 2.0, 2.4, 1.6, 1.0, 0.8, 0.8,
                       1.0, 0.9, 0.7, 0.7, 0.8, 1.1, 1.5, 1.8, 1.6, 1.2, 0.8, 0.5])
_DHW_SHAPE = _DHW_SHAPE / _DHW_SHAPE.mean()


@dataclass
class HeatPump:
    heat_loss_kw_per_k: float = 0.22      # H: space-heating demand per kelvin below the heating limit
    heating_limit_c: float = 15.0         # no space heating above this 24 h mean temperature
    dhw_kwh_per_year: float = 2000.0      # hot-water heat demand
    emitter: str = "radiator"             # 'radiator' (sink 40 - T) or 'floor' (sink 30 - 0.5 T)
    cop_correction: float = 0.85

    def cop(self, delta_t):
        return self.cop_correction * (6.08 - 0.09 * delta_t + 0.0005 * delta_t ** 2)

    def electric_kw(self, temp_c: pd.Series) -> pd.Series:
        t = temp_c.astype(float)
        t24 = t.rolling(24, min_periods=1).mean()
        q_sh = self.heat_loss_kw_per_k * (self.heating_limit_c - t24).clip(lower=0.0)
        q_dhw = self.dhw_kwh_per_year / 8760.0 * _DHW_SHAPE[t.index.hour]
        if self.emitter == "radiator":
            sink = 40.0 - 1.0 * t
        elif self.emitter == "floor":
            sink = 30.0 - 0.5 * t
        else:
            raise ValueError(f"unknown emitter {self.emitter!r}")
        kw = q_sh / self.cop(sink - t) + q_dhw / self.cop(50.0 - t)
        return kw.rename("hp_kw")


@dataclass
class Trips:
    p_use: float
    dep_mean: float
    dep_sd: float
    arr_mean: float
    arr_sd: float
    km_median: float
    km_sigma: float


@dataclass
class EV:
    capacity_kwh: float = 60.0
    max_power_kw: float = 11.0            # 3-phase wallbox
    efficiency: float = 0.90              # grid -> car battery
    kwh_per_km: float = 0.18
    target_soc: float = 0.8               # required at departure (user's setting in the app)
    public_price: float = 0.60            # EUR/kWh for a shortfall at departure (public AC/DC charging)
    energy_value: float = 0.25            # EUR/kWh: value of energy left in the car at the end of an evaluation
                                          # horizon (~night-time buy price); evaluation.py and the oracle use it
                                          # so no policy gains by ending with an emptier car than it started with
    max_km: float = 250.0
    seed: int = 0                         # which household's mobility (the schedule is a function of seed and date)
    weekday: Trips = field(default_factory=lambda: Trips(0.85, 7.5, 1.0, 17.5, 1.5, 35.0, 0.6))
    weekend: Trips = field(default_factory=lambda: Trips(0.55, 10.0, 2.0, 16.0, 2.5, 40.0, 0.8))

    def __post_init__(self):
        if isinstance(self.weekday, dict):
            self.weekday = Trips(**self.weekday)
        if isinstance(self.weekend, dict):
            self.weekend = Trips(**self.weekend)
        self.max_km = min(self.max_km, self.target_soc * self.capacity_kwh / self.kwh_per_km)   # range from target

    def trip(self, date: pd.Timestamp):
        """(departure hour, arrival hour, km) of the date, or None if the car stays home."""
        rng = np.random.default_rng([self.seed, date.toordinal()])
        p = self.weekend if date.dayofweek >= 5 else self.weekday
        used, dep, arr, km = rng.random(), rng.normal(p.dep_mean, p.dep_sd), rng.normal(p.arr_mean, p.arr_sd), rng.lognormal(np.log(p.km_median), p.km_sigma)
        if used >= p.p_use:
            return None
        dep = int(np.clip(round(dep), 4, 12))
        arr = int(np.clip(round(arr), dep + 1, 23))
        return dep, arr, float(min(km, self.max_km))

    def schedule(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Per hour of 'index' (local time): ev_home (car plugged in during this hour), ev_departs (this is the
        first hour away), ev_trip_kwh (energy used by the trip that arrives in this hour), ev_hours_to_departure
        (hours from this hour's start to the next departure; inf if none in the data)."""
        dates = index.tz_localize(None).normalize()
        hours = index.hour
        home = np.ones(len(index), dtype=bool)
        departs = np.zeros(len(index), dtype=bool)
        trip_kwh = np.zeros(len(index))
        for d in pd.unique(dates):
            t = self.trip(pd.Timestamp(d))
            if t is None:
                continue
            dep, arr, km = t
            rows = np.flatnonzero(dates == d)
            h = hours[rows]
            home[rows[(h >= dep) & (h < arr)]] = False
            departs[rows[h == dep]] = True
            trip_kwh[rows[h == arr]] = km * self.kwh_per_km
        dep_pos = np.flatnonzero(departs)
        nxt = np.searchsorted(dep_pos, np.arange(len(index)))
        to_dep = np.where(nxt < len(dep_pos), dep_pos[np.minimum(nxt, len(dep_pos) - 1)] - np.arange(len(index)), np.inf)
        return pd.DataFrame({"ev_home": home, "ev_departs": departs, "ev_trip_kwh": trip_kwh,
                             "ev_hours_to_departure": to_dep.astype(float)}, index=index)


def make_device(cls, cfg):
    """None/False -> None, True -> defaults, dict -> cls(**dict), instance -> itself."""
    if cfg is None or cfg is False:
        return None
    if cfg is True:
        return cls()
    if isinstance(cfg, cls):
        return cfg
    return cls(**cfg)


if __name__ == "__main__":
    from data import load_dataset, chronological_split
    df = load_dataset()
    hp = HeatPump()
    kw = hp.electric_kw(df["temp_c"])
    print("heat pump electricity kWh/year:", kw.groupby(kw.index.year).sum().round(0).to_dict(),
          f" peak {kw.max():.2f} kW")
    ev = EV()
    s = ev.schedule(chronological_split(df)["test"].index)
    print(f"EV test year: {s.ev_departs.sum()} trips, {s.ev_trip_kwh.sum() / ev.kwh_per_km:.0f} km, "
          f"{s.ev_trip_kwh.sum():.0f} kWh, home {100 * s.ev_home.mean():.0f} % of hours")
