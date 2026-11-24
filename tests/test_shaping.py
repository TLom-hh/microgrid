"""Checks for the EV potential-based shaping in training_env.TrainingReward.

Run with  'python tests/test_shaping.py'
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import load_dataset, chronological_split  # noqa: E402
from scenario import make_env  # noqa: E402
from training_env import TrainingReward, ev_potential  # noqa: E402
from baselines import make_policy  # noqa: E402
import evaluation as ev  # noqa: E402

SPL = chronological_split(load_dataset())


def rollout(env, policy, start, gamma):
    """Per step: (unshaped training reward, shaped reward, phi before, phi after)."""
    env.reset(options={"start_step": int(start), "soc": 0.5, "ev_soc": 0.4})
    rows, done = [], False
    while not done:
        phi = ev_potential(env.unwrapped)
        _, r, term, trunc, info = env.step(policy(env, None))
        rows.append((r - info["shaping"], r, phi, ev_potential(env.unwrapped)))
        done = term or trunc
    return np.array(rows)


def test_shaped_return_is_the_plain_return_plus_the_potential_difference():
    """The exact identity behind policy invariance: sum_t g^t (r_t + g phi_{t+1} - phi_t) = sum_t g^t r_t + g^T phi_T - phi_0."""
    g = 0.997
    env = TrainingReward(make_env(SPL["train"], household="ev_hp"), clip_penalty=0.05, ev_shaping=True, gamma=g)
    for start in env.episode_starts()[[0, 30, 60]]:
        for pol in (ev.scripted(make_policy("solar-only", "immediate")), lambda e, o: e.kw_to_action(0.0, 0.0)):
            rows = rollout(env, pol, start, g)
            disc = g ** np.arange(len(rows))
            lhs = (disc * rows[:, 1]).sum()
            rhs = (disc * rows[:, 0]).sum() + g ** len(rows) * rows[-1, 3] - rows[0, 2]
            assert np.isclose(lhs, rhs), (start, lhs, rhs)


def test_shaping_cancels_the_public_top_up_at_departure():
    env = TrainingReward(make_env(SPL["train"], household="ev"), ev_shaping=True, gamma=1.0)
    start = int(env.episode_starts()[5])
    env.reset(options={"start_step": start, "soc": 0.5, "ev_soc": 0.3})
    found = False
    while True:
        _, _, term, trunc, info = env.step(env.kw_to_action(0.0, 0.0))       # never charge: every departure is short
        short = env.history[-1]["ev_shortfall_kwh"]
        if short > 0:
            assert np.isclose(info["shaping"], short * env.ev.public_price)   # + public cost of the top-up
            found = True
        if term or trunc:
            break
    assert found


def test_charging_below_target_is_credited_in_the_same_hour():
    env = TrainingReward(make_env(SPL["train"], household="ev"), ev_shaping=True, gamma=1.0)
    env.reset(options={"start_step": int(env.episode_starts()[0]), "soc": 0.5, "ev_soc": 0.3})   # midnight, car home
    _, _, _, _, info = env.step(env.kw_to_action(0.0, 5.0))
    assert np.isclose(info["shaping"], 5.0 * env.ev.efficiency * env.ev.public_price)


def test_shaping_is_off_by_default_and_without_a_car():
    for household, kw in (("ev_hp", {}), ("base", dict(ev_shaping=True)), ("hp", dict(ev_shaping=True))):
        env = TrainingReward(make_env(SPL["val"], household=household), **kw)
        env.reset(options={"start_step": int(env.episode_starts()[0]), "soc": 0.5})
        for _ in range(48):
            _, _, _, _, info = env.step(env.action_space.sample())
            assert info["shaping"] == 0.0, household


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"{len(tests)} checks passed")
