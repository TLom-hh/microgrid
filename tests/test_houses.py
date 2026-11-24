"""Checks for the generalization setting (houses.py): house sets, PV / demand scaling, context features, MultiHouseEnv.

Run with  'python tests/test_houses.py'
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import load_dataset, chronological_split  # noqa: E402
from scenario import make_env  # noqa: E402
from baselines import make_policy  # noqa: E402
from training_env import TrainingReward  # noqa: E402
import evaluation as ev  # noqa: E402
import houses as H  # noqa: E402

SPL = chronological_split(load_dataset())


def test_the_reference_house_is_the_fixed_ev_hp_house():
    ref = H.make_house_env(SPL["test"], H.REFERENCE)
    fixed = make_env(SPL["test"], household="ev_hp", control="ev")
    assert all(np.array_equal(ref.df[c].values, fixed.df[c].values) for c in fixed.df.columns)
    pol = ev.scripted(make_policy("solar-only", "cheapest"))
    start = ref.episode_starts()[10]
    a, b = ev.run_episode(ref, pol, start), ev.run_episode(fixed, pol, start)
    assert np.allclose(a.select_dtypes("number").values, b.select_dtypes("number").values)
    opts = {"start_step": int(start), "soc": 0.5, "ev_soc": 0.5}
    o_ref, o_fix = ref.reset(seed=1, options=opts)[0], fixed.reset(seed=1, options=opts)[0]
    assert np.allclose(o_ref[:-5], o_fix)                                        # the same obs + house_context


def test_pv_and_demand_scale_linearly_and_the_heat_pump_does_not():
    a = make_env(SPL["val"], household="ev_hp")
    b = make_env(SPL["val"], household="ev_hp", pv_kwp=15.0, annual_kwh=2000.0)
    assert np.allclose(b.df["solar_kw"], 1.5 * a.df["solar_kw"])
    assert np.allclose(b.df["household_kw"], 0.5 * a.df["household_kw"])
    assert np.allclose(b.df["hp_kw"], a.df["hp_kw"])
    assert np.allclose(b.df["load_kw"], b.df["household_kw"] + b.df["hp_kw"])


def test_house_sets_are_fixed_distinct_and_all_build():
    assert H.sample_house("x", 7) == H.sample_house("x", 7)
    names = [h["name"] for hs in H.HOUSE_SETS.values() for h in hs]
    assert len(names) == len(set(names))
    params = [str(H.house_overrides(h)) for hs in H.HOUSE_SETS.values() for h in hs]
    assert len(params) == len(set(params))
    shape = None
    for hs in H.HOUSE_SETS:
        for env in H.make_envs(SPL["val"], hs):
            shape = shape or env.observation_space.shape
            assert env.observation_space.shape == shape and env.action_space.shape == (1,)
            car, s = env.ev, env.df
            assert s["ev_trip_kwh"].max() <= car.target_soc * car.capacity_kwh + 1e-9      # a trip never empties the car


def test_context_features():
    house = dict(H.REFERENCE, pv_kwp=12.0, annual_kwh=5000.0, capacity_kwh=8.0, max_power_kw=4.0,
                 heat_pump=dict(heat_loss_kw_per_k=0.33), ev=dict(capacity_kwh=50.0, max_power_kw=7.4, kwh_per_km=0.2, target_soc=0.7))
    env = H.make_house_env(SPL["val"], house)
    obs, _ = env.reset(options={"start_step": int(env.episode_starts()[0]), "soc": 0.5, "ev_soc": 0.5})
    assert np.allclose(obs[-9:], [0.5, 7.4 / 11, 1.0, 0.7, 1.2, 1.25, 0.8, 0.8, 1.5])
    no_hp = H.make_house_env(SPL["val"], dict(house, heat_pump=None))
    assert no_hp.reset(options={"start_step": int(env.episode_starts()[0]), "soc": 0.5})[0][-1] == 0.0
    hidden = H.make_house_env(SPL["val"], house, context="hidden")
    assert hidden.observation_space.shape[0] == env.observation_space.shape[0] - 9


def test_multi_house_env_draws_houses_reproducibly_and_trains_with_shaping():
    def houses_seen(seed):
        env = TrainingReward(H.make_multi_env(SPL["train"], "train"), clip_penalty=0.05, ev_shaping=True, gamma=0.997)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        seen, rewards = [], []
        for _ in range(5):
            seen.append(env.house)
            done = False
            while not done:
                _, r, term, trunc, info = env.step(env.action_space.sample())
                rewards.append(r)
                assert info["clip_penalty"] == 0.0
                done = term or trunc
            env.reset()
        return seen, rewards
    a, b = houses_seen(3), houses_seen(3)
    assert a == b and len(set(a[0])) > 1
    multi = H.make_multi_env(SPL["val"], "val")
    multi.reset(seed=0, options={"house": 4})
    assert multi.house == "val4"


def test_validation_return_scores_every_week_with_two_houses():
    envs = H.make_envs(SPL["val"], "val")
    weeks = [s for k, e in enumerate(envs) for s in e.episode_starts(episode_hours=ev.WEEK)[k % 3::3]]
    n_weeks = len(envs[0].episode_starts(episode_hours=ev.WEEK))
    assert len(weeks) == 2 * n_weeks and len(set(weeks)) == n_weeks
    r = H.validation_return(envs, ev.scripted(make_policy("solar-only", "cheapest")))
    assert np.isfinite(r)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} checks passed")
