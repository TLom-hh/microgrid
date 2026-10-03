"""Sensitivity of the battery's value and of the foresight headroom to the scenario parameters (no RL).

For each parameter setting, four controllers make one continuous run over a split (evaluation.py protocol
'continuous', start SoC 0.5): no battery, the solar-only heuristic, the rule-based controller (price
quantiles from the training split) and the cyclic perfect-foresight oracle. Reported as EUR/week:
    battery value       cost(no battery) - cost(oracle)
    heuristic share     (cost(no battery) - cost(solar-only)) / battery value
    foresight headroom  cost(solar-only) - cost(oracle), in EUR and as share of the battery value
Grid: degradation cost x buy markup x capacity (max power = capacity / 2, i.e. 0.5 C like the default
10 kWh / 5 kW); every other parameter comes from scenario.py. A second table runs the unchanged scenario on
each calendar year. Note: with markup 0.10 the buy price drops below the 0.077 feed-in whenever the spot
price is below -0.023 EUR/kWh, which opens a buy-low-sell-high pump that only the oracle exploits.

usage: python sensitivity.py [OUT_DIR] [DATA_CSV] [--split test|val|train]
writes OUT_DIR/sensitivity_<split>.csv and OUT_DIR/sensitivity_by_year.csv
"""

import sys, time, itertools
from pathlib import Path
import pandas as pd
from baselines import no_battery, solar_only, make_rule_based
from data import load_dataset, chronological_split, TIMEZONE
from evaluation import evaluate_continuous, oracle_continuous, scripted
from scenario import make_env

DEGRADATION = (0.0, 0.01, 0.02, 0.04)
MARKUP = (0.10, 0.17, 0.25)
CAPACITY = (5.0, 10.0, 20.0)


def scenario_costs(df, ref_df, **params):
    """Mean weekly cost of the scripted controllers and the cyclic oracle on one continuous run over df,
    plus the derived battery value and headroom. params override scenario.py."""
    env = make_env(df, **params)
    policies = (("no_battery", no_battery), ("solar_only", solar_only), ("rule_based", make_rule_based(env, ref_df=ref_df)))
    m = {name: float(evaluate_continuous(env, scripted(pol))["total"].mean()) for name, pol in policies}
    m["oracle"] = float(oracle_continuous(env)["total"].mean())
    value = m["no_battery"] - m["oracle"]
    return dict(**m, battery_value=value, heuristic_share=(m["no_battery"] - m["solar_only"]) / value,
                headroom_eur=m["solar_only"] - m["oracle"], headroom_share=(m["solar_only"] - m["oracle"]) / value)


if __name__ == "__main__":
    argv = sys.argv[1:]
    split = "test"
    if "--split" in argv:
        i = argv.index("--split"); split = argv[i + 1]; del argv[i:i + 2]
    out = Path(argv[0]) if argv else Path("results")
    out.mkdir(parents=True, exist_ok=True)
    full = load_dataset(argv[1]) if len(argv) > 1 else load_dataset()
    splits = chronological_split(full)
    df, train = splits[split], splits["train"]

    rows, t0 = [], time.time()
    for deg, mk, cap in itertools.product(DEGRADATION, MARKUP, CAPACITY):
        row = dict(degradation=deg, markup=mk, capacity_kwh=cap,
                   **scenario_costs(df, train, degradation_cost_per_kwh=deg, buy_markup=mk, capacity_kwh=cap, max_power_kw=cap / 2))
        rows.append(row)
        print(f"deg {deg:.2f} markup {mk:.2f} cap {cap:4.0f}: value {row['battery_value']:5.2f} EUR/wk, heuristic {100 * row['heuristic_share']:3.0f} %, "
              f"headroom {row['headroom_eur']:4.2f} EUR ({100 * row['headroom_share']:3.0f} %)  {time.time() - t0:5.0f}s", flush=True)
    sens = pd.DataFrame(rows)
    sens.to_csv(out / f"sensitivity_{split}.csv", index=False)

    pd.set_option("display.width", 200); pd.set_option("display.float_format", lambda x: f"{x:6.2f}")
    pct = lambda x: f"{100 * x:4.0f} %"
    print(f"\n=== split {split}: foresight headroom, EUR/week (solar-only minus oracle), capacity 10 kWh ===")
    print(sens[sens.capacity_kwh == 10].pivot(index="degradation", columns="markup", values="headroom_eur"))
    print("\n=== foresight headroom as share of battery value, capacity 10 kWh ===")
    print(sens[sens.capacity_kwh == 10].pivot(index="degradation", columns="markup", values="headroom_share").map(pct))
    print("\n=== headroom share by capacity at markup 0.17 ===")
    print(sens[sens.markup == 0.17].pivot(index="degradation", columns="capacity_kwh", values="headroom_share").map(pct))
    print("\n=== headroom EUR/week by capacity at markup 0.17 ===")
    print(sens[sens.markup == 0.17].pivot(index="degradation", columns="capacity_kwh", values="headroom_eur"))

    by_year = {}
    for year in sorted(set(full.index.year)):
        a, b = pd.Timestamp(f"{year}-01-01", tz=TIMEZONE), pd.Timestamp(f"{year + 1}-01-02", tz=TIMEZONE)
        df_y = full[(full.index >= a) & (full.index < b)]
        if len(df_y) > 24 * 28:                                   # skip the single 2024 row
            by_year[year] = scenario_costs(df_y, train)
    by_year = pd.DataFrame(by_year).T
    by_year.to_csv(out / "sensitivity_by_year.csv")
    print("\n=== scenario.py parameters, one continuous run per calendar year ===")
    print(by_year)
