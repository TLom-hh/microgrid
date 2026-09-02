"""Running simple policies on the microgrid environment"""

import sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from plotting import plot_daily_profiles, plot_episode
from smarthome import SmartMicrogridEnv

def no_battery(env):
    return np.array([0.0])

def random_policy(env):
    return env.action_space.sample()

def make_rule_based(env, low_q=0.25, high_q=0.75):
    low, high = env.df["buy_price"].quantile([low_q, high_q])

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
        return np.array([kw / env.max_power_kw])
    return policy

def run(env, policy, seed):
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(policy(env))
        done = term or trunc
    return env.history_frame()

if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path (".")
    out.mkdir(parents=True, exist_ok=True)
    df = None
    if len(sys.argv) > 2:
        import pandas as pd
        df = pd.read_csv(sys.argv[2], index_col="time")
        df.index = pd.to_datetime(df.index, utc=True).tz_convert("Europe/Berlin")
    env = SmartMicrogridEnv(df=df)
    seed = 3

    plot_daily_profiles(env.df).savefig(out / "daily_profiles.png", dpi=130)

    results = {}
    for name, pol in [("no battery", no_battery), ("random", random_policy), ("rule-based", make_rule_based(env))]:
        hist = run(env, pol, seed)
        total = (hist["cost"] + hist["degradation"]).sum()
        results[name] = total
        fig = plot_episode(hist, f"{name}: {hist.index[0]:%d %b} – {hist.index[-1]:%d %b %Y}, total {total:.2f} EUR")
        fig.savefig(out / f"episode_{name.replace(' ', '_')}.png", dpi=130)
        plt.close(fig)
        print(f"{name:12s} cost {total:7.2f} EUR   clipped steps {int(hist['clipped'].sum()):3d}/{len(hist)}")

    base, best = results["no battery"], results["rule-based"]
    print(f"rule-based saves {base - best:.2f} EUR ({100 * (base - best) / abs(base):.0f} %) vs. no battery this week")
