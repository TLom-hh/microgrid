"""Scripted baseline policies for the microgrid environment.

Each policy is a function fn(env) -> action. It decides on a battery power (and, with an EV, a wallbox power) in
kW from the current row of the data and returns env.kw_to_action(kw, ev_kw), so it works with every action_mode.

Battery rules (battery kW, given the wallbox kW the car actually takes this hour):
    no_battery    battery idle: the floor every controller is compared with
    dump          always discharge at full power: a deliberately bad policy
    solar_only    store PV surplus left after the car, cover the house's deficit (never the car's) from the battery
    rule-based    solar_only plus price thresholds: charge from the grid when cheap, hold back when mid-priced

EV rules (requested wallbox kW; 0 while the car is away; env.ev holds the car's parameters):
    immediate     charge at full power after arrival until target_soc (the default wallbox: state of practice)
    pv-surplus    charge from PV surplus (up to 100 %); if the target can no longer be reached otherwise, charge at
                  full power (smart PV wallbox, e.g. evcc's 'PV' mode with a departure plan)
    cheapest      the energy to target_soc in the cheapest hours before departure, ranked on day-ahead buy prices
                  (known up to 24 h ahead), plus any PV surplus (price-optimal smart charging, e.g. Tibber/evcc)

no_battery, dump, solar_only and make_rule_based use 'immediate' for the car; make_policy combines any pair.
Without an EV all of them act exactly as before.

Scoring happens in evaluation.py (fixed weeks or one continuous run per split).

usage: 'python baselines.py [OUT_DIR] [DATA_CSV] [--household base|ev|hp|ev_hp]'
    daily profile plot of the data and one example week per season for each baseline, on the test split.
"""

import sys
from pathlib import Path
import numpy as np

# ---- EV rules: env -> requested wallbox kW ------------------------------------------------------------------

def _ev_needed_kwh(env):
    """Grid-side kWh still needed to reach target_soc."""
    return max(0.0, env.ev.target_soc * env.ev.capacity_kwh - env.ev_energy) / env.ev.efficiency

def _ev_home(env):
    return bool(env.df["ev_home"].iat[env.current_step])

def ev_immediate(env):
    if not _ev_home(env):
        return 0.0
    return min(_ev_needed_kwh(env) / env.dt, env.ev.max_power_kw)

def _ev_deadline_kw(env):
    """Power that must flow now so the target is still reachable at full power in the remaining hours."""
    hours_left = env.df["ev_hours_to_departure"].iat[env.current_step]
    later = max(hours_left - 1.0, 0.0) * env.ev.max_power_kw * env.dt
    return min(max(_ev_needed_kwh(env) - later, 0.0) / env.dt, env.ev.max_power_kw)

def _pv_surplus_kw(env):
    row = env.current_row()
    return max(row["solar_kw"] - row["load_kw"], 0.0)

def ev_pv_surplus(env):
    if not _ev_home(env):
        return 0.0
    return max(min(_pv_surplus_kw(env), env.ev.max_power_kw), _ev_deadline_kw(env))

def ev_cheapest(env):
    if not _ev_home(env):
        return 0.0
    i = env.current_step
    hours_left = env.df["ev_hours_to_departure"].iat[i]
    window = int(min(hours_left, 24.0)) if np.isfinite(hours_left) else 24
    prices = env.df["buy_price"].values[i:i + max(window, 1)]
    need = _ev_needed_kwh(env)
    n_hours = int(np.ceil(need / (env.ev.max_power_kw * env.dt) - 1e-9))
    grid_kw = 0.0
    if n_hours > 0 and np.sum(prices < prices[0]) < n_hours:    # this hour is among the n cheapest of the window
        grid_kw = min(need / env.dt, env.ev.max_power_kw)
    return max(grid_kw, min(_pv_surplus_kw(env), env.ev.max_power_kw), _ev_deadline_kw(env))

EV_RULES = {"immediate": ev_immediate, "pv-surplus": ev_pv_surplus, "cheapest": ev_cheapest}

# ---- battery rules: (env, ev_kw) -> battery kW --------------------------------------------------------------

def _bat_idle(env, ev_kw):
    return 0.0

def _bat_dump(env, ev_kw):
    return -env.max_power_kw

def _bat_solar_only(env, ev_kw):
    row = env.current_row()
    house = row["solar_kw"] - row["load_kw"]
    if house - ev_kw > 0:                     # surplus left after the car: store it
        kw = house - ev_kw
    elif house < 0:                           # the house's own deficit: cover it from the battery
        kw = house
    else:                                     # the car takes the surplus, the battery stays out of it
        kw = 0.0
    return float(np.clip(kw, -env.max_power_kw, env.max_power_kw))

BATTERY_RULES = {"no battery": _bat_idle, "dump": _bat_dump, "solar-only": _bat_solar_only}

def make_policy(battery, ev="immediate"):
    """battery: a key of BATTERY_RULES or a function (env, ev_kw) -> kW; ev: a key of EV_RULES or env -> kW."""
    bat_fn = BATTERY_RULES[battery] if isinstance(battery, str) else battery
    ev_fn = EV_RULES[ev] if isinstance(ev, str) else ev

    def policy(env):
        ev_kw = ev_fn(env) if env.ev else 0.0
        return env.kw_to_action(bat_fn(env, env._feasible_ev(ev_kw)), ev_kw)
    return policy

no_battery = make_policy("no battery")
dump = make_policy("dump")
solar_only = make_policy("solar-only")

def make_rule_based(env, low_q=0.25, high_q=0.75, ref_df=None, ev="immediate"):
    ref = env.df if ref_df is None else ref_df
    prices = ref["buy_price"] if "buy_price" in ref else ref["spot_price"] + env.buy_markup
    low, high = prices.quantile([low_q, high_q])

    def battery_kw(env, ev_kw):
        row = env.current_row()
        surplus = row["solar_kw"] - row["load_kw"] - ev_kw
        deficit = min(row["solar_kw"] - row["load_kw"], 0.0)      # the house's, never the car's
        if surplus > 0:                       # store solar surplus
            return min(surplus, env.max_power_kw)
        if row["buy_price"] <= low:           # cheap: charge from grid
            return env.max_power_kw
        if row["buy_price"] >= high:          # expensive: cover deficit from battery
            return max(deficit, -env.max_power_kw)
        return max(deficit, -env.max_power_kw) if env.soc > 0.5 else 0.0   # mid price: only if well charged
    return make_policy(battery_kw, ev)

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from data import load_dataset, chronological_split
    from evaluation import run_episode, scripted
    from plotting import plot_daily_profiles, plot_episode
    from scenario import make_env

    argv = sys.argv[1:]
    household = "base"
    if "--household" in argv:
        i = argv.index("--household"); household = argv[i + 1]; del argv[i:i + 2]
    out = Path(argv[0]) if argv else Path("figures")
    out.mkdir(parents=True, exist_ok=True)
    splits = chronological_split(load_dataset(argv[1]) if len(argv) > 1 else load_dataset())
    env = make_env(splits["test"], household=household)
    plot_daily_profiles(env.df).savefig(out / ("daily_profiles.png" if household == "base" else f"daily_profiles_{household}.png"), dpi=130)

    policies = [("no battery", no_battery), ("dump", dump), ("solar-only", solar_only),
                ("rule-based", make_rule_based(env, ref_df=splits["train"]))]
    for name, pol in policies:
        for start in env.episode_starts()[::13]:                 # four example weeks, one per season
            hist = run_episode(env, scripted(pol), start)
            total = (hist["cost"] + hist["degradation"]).sum()
            fig = plot_episode(hist, f"{name}: {hist.index[0]:%d %b} – {hist.index[-1]:%d %b %Y}, total {total:.2f} EUR")
            prefix = "" if household == "base" else f"{household}_"
            fig.savefig(out / f"episode_{prefix}{name.replace(' ', '_')}_{hist.index[0]:%Y%m%d}.png", dpi=130)
            plt.close(fig)
            print(f"{name:12s} week {hist.index[0]:%Y-%m-%d} cost {total:7.2f} EUR   clipped steps {int(hist['clipped'].sum()):3d}/{len(hist)}")
