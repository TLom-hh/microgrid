"""Checks for the household devices: heat pump load, EV schedule, EV physics in the env, oracle and evaluation with an EV.

Run with  'python tests/test_households.py'
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import load_dataset, chronological_split  # noqa: E402
from scenario import make_env  # noqa: E402
from devices import EV, HeatPump  # noqa: E402
from baselines import make_policy, solar_only, no_battery  # noqa: E402
import evaluation as ev  # noqa: E402
import oracle  # noqa: E402

DF = load_dataset()
SPL = chronological_split(DF)


def test_base_household_is_the_old_env():
    env = make_env(SPL["test"])
    assert env.ev is None and env.heat_pump is None
    assert env.observation_space.shape == (56,) and env.action_space.shape == (1,)
    assert np.array_equal(env.df["load_kw"].values, SPL["test"]["load_kw"].values)
    assert "ev_soc" not in ev.run_episode(env, ev.scripted(solar_only), int(env.episode_starts()[0]))


def test_heat_pump_adds_a_plausible_temperature_driven_load():
    env = make_env(SPL["test"], household="hp")
    hp = env.df["hp_kw"]
    assert (hp >= 0).all()
    assert np.allclose(env.df["load_kw"], env.df["household_kw"] + hp)
    assert 3000 < hp.sum() < 6000                                       # kWh electricity in 2023
    winter, summer = hp[hp.index.month.isin([1, 2, 12])].mean(), hp[hp.index.month.isin([6, 7, 8])].mean()
    assert winter > 4 * summer                                          # driven by the outdoor temperature
    assert HeatPump().cop(40.0) < HeatPump().cop(20.0)                  # colder source -> worse COP


def test_ev_schedule_is_home_at_midnight_and_the_same_in_every_slice():
    car = EV()
    s = car.schedule(SPL["test"].index)
    assert s["ev_home"][s.index.hour == 0].all()                       # episodes start at midnight with the car home
    assert not s["ev_home"][s["ev_departs"]].any()                     # the departure hour is the first hour away
    arrivals = s["ev_trip_kwh"] > 0
    assert s["ev_home"][arrivals].all()                                 # the trip's energy leaves at arrival
    assert s["ev_trip_kwh"].max() <= car.max_km * car.kwh_per_km + 1e-9
    assert arrivals.sum() == s["ev_departs"].sum()                      # one return per departure
    full = car.schedule(DF.index).loc[s.index]                          # a date has the same trip in every slice
    assert full[["ev_home", "ev_departs", "ev_trip_kwh"]].equals(s[["ev_home", "ev_departs", "ev_trip_kwh"]])
    km = s["ev_trip_kwh"].sum() / car.kwh_per_km
    assert 9000 < km < 16000                                            # km per year, KBA order of magnitude


def test_ev_energy_and_grid_accounting_in_the_env():
    env = make_env(SPL["test"], household="ev")
    start = int(env.episode_starts()[5])
    hist = ev.run_episode(env, ev.scripted(make_policy("solar-only", "immediate")), start)
    car = env.ev
    e_before, e_after = hist["ev_soc_before"] * car.capacity_kwh, hist["ev_soc"] * car.capacity_kwh
    trips = env.df["ev_trip_kwh"].values[start + 1: start + 1 + len(hist)]
    expected = e_before + hist["ev_kw"] * car.efficiency + hist["ev_shortfall_kwh"] - trips
    assert np.allclose(e_after, expected)                               # charging, public top-up and trips, nothing else
    assert np.allclose(hist["grid_kw"], hist["load_kw"] - hist["solar_kw"] + hist["battery_kw"] + hist["ev_kw"])
    assert (hist["ev_kw"][~hist["ev_home"]] == 0).all()                 # no charging while away
    assert np.allclose(hist["ev_public_cost"], hist["ev_shortfall_kwh"] * car.public_price)
    assert (hist["ev_soc"] >= 0).all() and (hist["ev_soc"] <= 1 + 1e-9).all()


def test_ev_departure_shortfall_is_bought_at_the_public_price():
    env = make_env(SPL["test"], household="ev")
    start = int(env.episode_starts()[5])
    idle = lambda e, obs: e.kw_to_action(0.0, 0.0)                       # never charge the car
    hist = ev.run_episode(env, idle, start)
    deps = env.df["ev_departs"].values[start + 1: start + 1 + len(hist)]
    assert deps.any()
    first = int(np.flatnonzero(deps)[0])                                # the step leading into the first departure
    short = hist["ev_shortfall_kwh"].iloc[first]
    assert np.isclose(short, (env.ev.target_soc - ev.EVAL_EV_SOC) * env.ev.capacity_kwh)
    assert np.isclose(hist["ev_soc"].iloc[first], env.ev.target_soc)    # topped up to the target, then away


def test_kw_to_action_round_trip_with_an_ev():
    for mode in ("power", "target_soc", "residual"):
        env = make_env(SPL["val"], household="ev", action_mode=mode)
        env.reset(options={"start_step": int(env.episode_starts()[0]), "soc": 0.5, "ev_soc": 0.3})
        a = env.kw_to_action(1.0, 2.0)                                   # small enough for residual's +-5 kW range
        assert a.shape == (2,)
        assert np.isclose(env._feasible_ev(env._ev_request(a)), 2.0)
        assert np.isclose(env._action_to_kw(a), 1.0), mode


def test_oracle_with_an_ev_matches_the_env_and_beats_the_heuristics():
    for household in ("ev", "ev_hp"):
        env = make_env(SPL["test"], household=household)
        for start in env.episode_starts()[[1, 20, 40]]:
            opts = {"start_step": int(start), "soc": 0.5, "ev_soc": ev.EVAL_EV_SOC}
            free = ev.summarize(oracle.run_oracle(env, options=opts, cyclic=False), env)["total"]   # raises on solver/env mismatch
            cyc = ev.summarize(oracle.run_oracle(env, options=opts, cyclic=True), env)["total"]
            for battery in ("no battery", "solar-only"):
                for car in ("immediate", "pv-surplus", "cheapest"):
                    s = ev.summarize(ev.run_episode(env, ev.scripted(make_policy(battery, car)), start), env)
                    assert free <= s["total"] + 1e-6, (household, start, battery, car)       # free end: a true lower bound
                    if s["soc_end"] >= 0.5 - 1e-9:                                           # cyclic: for policies that refill too
                        assert cyc <= s["total"] + 1e-6, (household, start, battery, car)


def test_smart_ev_rules_meet_the_deadline_when_it_is_reachable():
    env = make_env(SPL["test"], household="ev")
    for car in ("pv-surplus", "cheapest"):
        hist = ev.run_episode(env, ev.scripted(make_policy("solar-only", car)), int(env.episode_starts()[0]), hours=24 * 28)
        assert hist["ev_shortfall_kwh"].sum() < 1.0, car                # 4 winter weeks, (almost) never short


def test_weekly_totals_add_up_with_the_ev_stock_valuation():
    env = make_env(SPL["val"], household="ev_hp")
    o = ev.continuous_options(env)
    hist = ev.run_episode(env, ev.scripted(no_battery), o["start_step"], o["soc"], o["episode_hours"])
    weeks = ev.weekly_table(hist, env)
    run = ev.summarize(hist, env)
    assert np.isclose(weeks["total"].sum(), run["total"])               # the per-week EV terms telescope
    assert np.allclose(weeks["ev_soc_start"].values[1:], weeks["ev_soc_end"].values[:-1])


def test_training_reward_baseline_excludes_the_car():
    from training_env import TrainingReward
    env = TrainingReward(make_env(SPL["val"], household="ev"))
    env.reset(options={"start_step": int(env.episode_starts()[0]), "soc": 0.5, "ev_soc": 0.3})
    _, r, _, _, info = env.step(env.kw_to_action(0.0, env.ev.max_power_kw))   # idle battery, car charging at midnight
    assert r < 0 and np.isclose(r, -env.history[-1]["ev_kw"] * env.history[-1]["buy_price"])


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} checks passed")
