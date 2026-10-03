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
"""

SCENARIO = dict(
    capacity_kwh=10.0,
    max_power_kw=5.0,
    efficiency=0.95,
    degradation_cost_per_kwh=0.02,
    buy_markup=0.17,
    feed_in_price=0.077,
)

def make_env(df, **overrides):
    from microgrid_env import SmartMicrogridEnv
    return SmartMicrogridEnv(df=df, **{**SCENARIO, **overrides})
