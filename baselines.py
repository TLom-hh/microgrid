"""Scripted baseline policies for the microgrid environment.

Each policy is a function fn(env) -> action. It decides on a battery power in kW from the current row of the
data and returns env.kw_to_action(kw), so it works with every action_mode of the env.

    no_battery    battery idle: the floor every controller is compared with
    dump          always discharge at full power: a deliberately bad policy
    solar_only    store PV surplus, cover the deficit from the battery, never trade with the grid on purpose
    rule-based    solar_only plus price thresholds: charge from the grid when cheap, hold back when mid-priced

Scoring happens in evaluation.py (fixed weeks or one continuous run per split).

usage: 'python baselines.py [OUT_DIR] [DATA_CSV]'
    daily profile plot of the data and one example week per season for each baseline, on the test split.
"""

import sys
from pathlib import Path
import numpy as np

def no_battery(env):
    return env.kw_to_action(0.0)

def dump(env):
    return env.kw_to_action(-env.max_power_kw)

def solar_only(env):
    row = env.current_row()
    surplus = row["solar_kw"] - row["load_kw"]
    return env.kw_to_action(float(np.clip(surplus, -env.max_power_kw, env.max_power_kw)))

def make_rule_based(env, low_q=0.25, high_q=0.75, ref_df=None):
    ref = env.df if ref_df is None else ref_df
    prices = ref["buy_price"] if "buy_price" in ref else ref["spot_price"] + env.buy_markup
    low, high = prices.quantile([low_q, high_q])

    def policy(env):
        row = env.current_row()
        surplus = row["solar_kw"] - row["load_kw"]
        if surplus > 0:                       # store solar surplus
            kw = min(surplus, env.max_power_kw)
        elif row["buy_price"] <= low:         # cheap: charge from grid
            kw = env.max_power_kw
        elif row["buy_price"] >= high:        # expensive: cover deficit from battery
            kw = max(surplus, -env.max_power_kw)
        else:                                 # mid price: cover deficit only if well charged
            kw = max(surplus, -env.max_power_kw) if env.soc > 0.5 else 0.0
        return env.kw_to_action(kw)
    return policy

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from data import load_dataset, chronological_split
    from evaluation import run_episode, scripted
    from plotting import plot_daily_profiles, plot_episode
    from scenario import make_env

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("figures")
    out.mkdir(parents=True, exist_ok=True)
    splits = chronological_split(load_dataset(sys.argv[2]) if len(sys.argv) > 2 else load_dataset())
    env = make_env(splits["test"])
    plot_daily_profiles(env.df).savefig(out / "daily_profiles.png", dpi=130)

    policies = [("no battery", no_battery), ("dump", dump), ("solar-only", solar_only),
                ("rule-based", make_rule_based(env, ref_df=splits["train"]))]
    for name, pol in policies:
        for start in env.episode_starts()[::13]:                 # four example weeks, one per season
            hist = run_episode(env, scripted(pol), start)
            total = (hist["cost"] + hist["degradation"]).sum()
            fig = plot_episode(hist, f"{name}: {hist.index[0]:%d %b} – {hist.index[-1]:%d %b %Y}, total {total:.2f} EUR")
            fig.savefig(out / f"episode_{name.replace(' ', '_')}_{hist.index[0]:%Y%m%d}.png", dpi=130)
            plt.close(fig)
            print(f"{name:12s} week {hist.index[0]:%Y-%m-%d} cost {total:7.2f} EUR   clipped steps {int(hist['clipped'].sum()):3d}/{len(hist)}")
