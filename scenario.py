"""The scenario: every physical and tariff parameter of the household, defined once.

All scripts build their environments through make_env(), so changing the scenario is one edit here and
every baseline, oracle, agent and table follows. Values and their status:

    capacity_kwh              10.0    usable battery capacity
    max_power_kw               5.0    charge / discharge limit (0.5 C)
    efficiency                 0.95   one-way efficiency (round trip 0.90)
    degradation_cost_per_kwh   0.02   wear per kWh moved, both directions (placeholder: calibrate and cite)
    buy_markup                 0.17   EUR/kWh added to the spot price (grid fees, taxes, levies)
    feed_in_price              0.077  EUR/kWh for exports (EEG partial feed-in, <= 10 kWp)

PV size (10 kWp) and household demand (4000 kWh/year) are fixed earlier, when data.py builds the dataset.
sensitivity.py shows how the battery's value and the room for foresight move with these parameters.

Household types (HOUSEHOLDS, chosen with make_env(df, household=...)): which devices besides the base load
the house has. Device parameters and their sources are documented in devices.py.
    base     battery + PV only (the setting of all results before 2026-11)
    ev       + electric vehicle with controllable wallbox (2-D action)
    hp       + heat pump (uncontrolled load, driven by outdoor temperature)
    ev_hp    + both
"""

SCENARIO = dict(
    capacity_kwh=10.0,
    max_power_kw=5.0,
    efficiency=0.95,
    degradation_cost_per_kwh=0.02,
    buy_markup=0.17,
    feed_in_price=0.077,
)

HOUSEHOLDS = {
    "base": {},
    "ev": dict(ev=True),
    "hp": dict(heat_pump=True),
    "ev_hp": dict(ev=True, heat_pump=True),
}

def make_env(df, household="base", **overrides):
    """Env for one household type; overrides win over the scenario and the household's device settings
    (e.g. ev=dict(seed=3) for another household's mobility, heat_pump=dict(emitter='floor'))."""
    from microgrid_env import SmartMicrogridEnv
    if household not in HOUSEHOLDS:
        raise ValueError(f"unknown household {household!r}; choose from {list(HOUSEHOLDS)}")
    env = SmartMicrogridEnv(df=df, **{**SCENARIO, **HOUSEHOLDS[household], **overrides})
    env.household = household
    return env
