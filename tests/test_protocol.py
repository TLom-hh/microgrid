"""Checks for the data split, the deterministic evaluation protocol and the oracle.

Run with  'python tests/test_protocol.py'
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import load_dataset, chronological_split, SPLITS  # noqa: E402
from baselines import solar_only, no_battery, dump  # noqa: E402
from scenario import make_env, SCENARIO  # noqa: E402
from training_env import TrainingReward  # noqa: E402
import evaluation as ev  # noqa: E402
import oracle  # noqa: E402

DF = load_dataset()
SPL = chronological_split(DF)

def SmartMicrogridEnv(df, **kwargs):
    """Every test env carries the scenario parameters, like the scripts."""
    return make_env(df, **kwargs)

def test_make_env_applies_the_scenario_and_overrides():
    env = make_env(SPL["val"])
    assert all(getattr(env, k) == v for k, v in SCENARIO.items())
    env = make_env(SPL["val"], capacity_kwh=20.0, action_mode="target_soc")
    assert env.capacity_kwh == 20.0 and env.action_mode == "target_soc" and env.efficiency == SCENARIO["efficiency"]

def test_training_reward_is_saving_minus_clip_penalty():
    plain, wrapped = make_env(SPL["val"]), TrainingReward(make_env(SPL["val"]), clip_penalty=0.05)
    opts = {"start_step": int(plain.episode_starts()[3]), "soc": 0.0}       # empty battery: discharge requests are infeasible
    plain.reset(options=opts); wrapped.reset(options=opts)
    idle = wrapped.step(no_battery(wrapped))                                # idle battery: saving 0, nothing infeasible
    assert abs(idle[1]) < 1e-12
    plain.step(no_battery(plain))
    _, r_env, *_ = plain.step(dump(plain))                                  # 5 kW discharge request on an empty battery
    _, r_train, _, _, info = wrapped.step(dump(wrapped))
    assert abs(info["env_reward"] - r_env) < 1e-12                          # the wrapper does not change the physics
    assert abs(info["saving"]) < 1e-12                                      # nothing was delivered, so nothing was saved
    assert abs(r_train + 0.05 * 5.0) < 1e-9                                 # 5 infeasible kWh at 0.05 EUR/kWh
    assert wrapped.soc == plain.soc and len(wrapped.history) == 2           # attributes forward to the wrapped env

def test_split_is_chronological_and_disjoint():
    names = list(SPLITS)
    for a, b in zip(names, names[1:]):
        assert SPL[a].index[-1] < SPL[b].index[0], (a, b)
    assert sum(len(s) for s in SPL.values()) == len(DF)                 # nothing lost, nothing duplicated
    assert SPL["train"].index[0].year == 2020 and SPL["test"].index[0].year == 2023

def test_env_on_a_split_cannot_see_other_splits():
    env = SmartMicrogridEnv(df=SPL["val"])
    starts = env.episode_starts()
    last_row_used = int(starts[-1]) + env.episode_hours + env.forecast_horizon + 1
    assert last_row_used <= len(env.df)                                 # forecasts of the last week stay inside the slice
    assert env.df.index[0] >= SPL["val"].index[0] and env.df.index[-1] <= SPL["val"].index[-1]
    assert len(starts) == 26                                            # 26 whole weeks in H2 2022

def test_episode_starts_are_midnights_seven_days_apart_across_dst():
    env = SmartMicrogridEnv(df=SPL["test"])
    starts = env.episode_starts()
    idx = env.df.index[starts]
    assert (idx.hour == 0).all()
    days = np.diff(idx.tz_localize(None).normalize().values).astype("timedelta64[D]").astype(int)
    assert (days == 7).all()                                            # calendar weeks, also across the March/October changes
    assert len(starts) == 52

def test_reset_options_are_honoured_and_deterministic():
    env = SmartMicrogridEnv(df=SPL["test"])
    start = int(env.episode_starts()[10])
    obs1, info = env.reset(options={"start_step": start, "soc": 0.5})
    assert info["soc"] == 0.5 and env.start_step == start
    hist_a = ev.run_episode(env, ev.scripted(solar_only), start, soc=0.5)
    hist_b = ev.run_episode(env, ev.scripted(solar_only), start, soc=0.5)
    assert np.allclose(hist_a["cost"].values, hist_b["cost"].values)     # same week, same SoC -> identical costs
    assert len(hist_a) == env.episode_hours
    try:
        env.reset(options={"start_step": start + 1, "soc": 0.5})        # not a midnight
        assert False, "expected ValueError"
    except ValueError:
        pass

def test_scoring_a_policy_that_reads_the_noisy_forecast_is_reproducible():
    env = SmartMicrogridEnv(df=SPL["val"])
    noisy = lambda e, obs: [float(np.clip(obs[-1] * 5 - 0.5, -1, 1))]     # acts on the last (noisy) solar-forecast entry
    start = int(env.episode_starts()[2])
    a = ev.run_episode(env, noisy, start)["cost"].values
    ev.run_episode(env, noisy, int(env.episode_starts()[7]))              # another episode in between must not matter
    b = ev.run_episode(env, noisy, start)["cost"].values
    assert np.array_equal(a, b)

def test_continuous_run_covers_all_weeks_and_carries_soc():
    env = SmartMicrogridEnv(df=SPL["val"])
    o = ev.continuous_options(env)
    hist = ev.run_episode(env, ev.scripted(solar_only), o["start_step"], o["soc"], o["episode_hours"])
    weeks = ev.weekly_table(hist, env)
    assert len(weeks) == 26
    assert np.allclose(weeks["soc_start"].values[1:], weeks["soc_end"].values[:-1])   # SoC carried across weeks
    assert abs(weeks["total"].sum() - (hist["cost"] + hist["degradation"]).sum()) < 1e-9

def test_oracle_lp_matches_env_and_beats_dp_and_heuristic():
    env = SmartMicrogridEnv(df=SPL["test"])
    for start in env.episode_starts()[[0, 13, 26, 39]]:
        opts = {"start_step": int(start), "soc": 0.5}
        lp = oracle.run_oracle(env, options=opts, solver="lp")          # raises on any solver/env mismatch
        dp = oracle.run_oracle(env, options=opts, solver="dp")
        heur = ev.run_episode(env, ev.scripted(solar_only), start)
        c = lambda h: (h["cost"] + h["degradation"]).sum()
        assert c(lp) <= c(dp) + 1e-6, (c(lp), c(dp))                     # exact LP never worse than the discretised DP
        assert c(dp) - c(lp) < 0.1                                       # ... and the DP is within a few cents
        assert c(lp) <= c(heur) + 1e-6                                   # ceiling never worse than the heuristic
        assert lp["soc"].iloc[-1] >= 0.5 - 1e-6                          # cyclic constraint

def test_oracle_free_end_is_a_lower_bound_of_cyclic():
    env = SmartMicrogridEnv(df=SPL["val"])
    opts = {"start_step": int(env.episode_starts()[5]), "soc": 0.5}
    free = oracle.oracle(env, options=opts, cyclic=False)["total"]
    cyc = oracle.oracle(env, options=opts, cyclic=True)["total"]
    assert free <= cyc + 1e-9

def test_no_battery_cost_is_independent_of_the_action_mode():
    """power and target_soc can both express 'do nothing' exactly; residual cannot when |solar - load| > max_power."""
    costs = {}
    for mode in ("power", "target_soc", "residual"):
        env = SmartMicrogridEnv(df=SPL["val"], action_mode=mode)
        costs[mode] = ev.evaluate_weeks(env, ev.scripted(no_battery))["total"].sum()
    assert np.isclose(costs["power"], costs["target_soc"])
    assert costs["residual"] >= costs["power"] - 1e-9

if __name__ == "__main__":
    oracle.selftest()
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests) + 1} checks passed")
