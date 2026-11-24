"""Checks for control='ev': the agent sets only the wallbox, the battery follows the solar-only rule.

Run with  'python tests/test_ev_control.py'
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import load_dataset, chronological_split  # noqa: E402
from scenario import make_env  # noqa: E402
from baselines import make_policy, EV_RULES  # noqa: E402
from training_env import TrainingReward  # noqa: E402
import evaluation as ev  # noqa: E402
import oracle  # noqa: E402

SPL = chronological_split(load_dataset())


def test_spaces_and_observed_context():
    full = make_env(SPL["val"], household="ev_hp")
    env = make_env(SPL["val"], household="ev_hp", control="ev", ev=dict(capacity_kwh=80.0, max_power_kw=7.4))
    assert env.action_space.shape == (1,) and full.action_space.shape == (2,)
    assert env.observation_space.shape[0] == full.observation_space.shape[0] + 4
    obs, _ = env.reset(options={"start_step": int(env.episode_starts()[0]), "soc": 0.5, "ev_soc": 0.5})
    assert np.allclose(obs[-4:], [0.8, 7.4 / 11.0, 0.18 / 0.2, 0.8])        # the car's parameters, last block
    try:
        make_env(SPL["val"], household="hp", control="ev")
        raise AssertionError("control='ev' without a car must fail")
    except ValueError:
        pass


def test_rules_run_identically_under_both_controls():
    full = make_env(SPL["test"], household="ev_hp")
    env = make_env(SPL["test"], household="ev_hp", control="ev")
    for car in EV_RULES:
        pol = ev.scripted(make_policy("solar-only", car))
        for start in full.episode_starts()[[3, 30]]:
            a, b = ev.run_episode(full, pol, start), ev.run_episode(env, pol, start)
            cols = ["battery_kw", "ev_kw", "soc", "ev_soc", "grid_kw", "cost", "degradation"]
            assert np.allclose(a[cols].values, b[cols].values), (car, start)


def test_the_battery_never_makes_an_infeasible_request():
    env = TrainingReward(make_env(SPL["train"], household="ev_hp", control="ev"), clip_penalty=0.05)
    env.reset(seed=1)
    env.action_space.seed(1)
    for _ in range(24 * 14):
        _, _, _, trunc, info = env.step(env.action_space.sample())
        assert info["clip_penalty"] == 0.0 and not env.history[-1]["clipped"]
        if trunc:
            env.reset()


def test_kw_to_action_round_trip():
    env = make_env(SPL["val"], household="ev", control="ev")
    env.reset(options={"start_step": int(env.episode_starts()[0]), "soc": 0.5, "ev_soc": 0.3})
    a = env.kw_to_action(3.0, 4.0)                                          # the battery kW is ignored
    assert a.shape == (1,) and np.isclose(env._feasible_ev(env._ev_request(a)), 4.0)


def test_ev_only_oracle_sits_between_the_full_oracle_and_the_rules():
    env = make_env(SPL["test"], household="ev_hp")
    for start in env.episode_starts()[[1, 20, 40]]:
        opts = {"start_step": int(start), "soc": 0.5, "ev_soc": ev.EVAL_EV_SOC}
        free = ev.summarize(oracle.run_oracle(env, options=opts, cyclic=False), env)["total"]
        sc = ev.summarize(oracle.run_oracle(env, options=opts, cyclic=False, battery="self-consumption"), env)["total"]
        assert free <= sc + 1e-6
        for car in EV_RULES:
            rule = ev.summarize(ev.run_episode(env, ev.scripted(make_policy("solar-only", car)), start), env)["total"]
            assert sc <= rule + 1e-6, (start, car, sc, rule)                 # every EV-only controller is above it
    try:
        oracle.run_oracle(make_env(SPL["test"], household="ev", control="ev"), options={"start_step": int(start)})
        raise AssertionError("the oracle must refuse an env with control='ev'")
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} checks passed")
