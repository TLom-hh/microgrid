"""Generalization across houses: one agent for many different households (WP: generalization setting).

A house is a dict of make_env parameters: PV size, household demand, home battery, heat pump (or none) and the car
(battery, wallbox, consumption, target SoC and its own mobility). Every house has a car, because the learners
control the wallbox (control='ev'; with full control they fail on the battery already in one fixed house).

Parameter ranges (uniform unless stated; plausible German single-family houses, not a calibrated census):
    pv_kwp            5 .. 15 kWp             typical rooftop systems
    annual_kwh        2500 .. 6000 kWh/year   household demand without heat pump, ~1-5 persons
    battery           5 .. 15 kWh, 0.5 C      home storage (max_power_kw = capacity / 2)
    heat pump         50 %: heat loss 0.15 .. 0.30 kW/K (building size and insulation), radiators or floor heating
    car battery       40 .. 100 kWh           small cars to large SUVs
    wallbox           3.7 / 7.4 / 11 kW       (1-phase 16 A, 1-phase 32 A or 2-phase, 3-phase 16 A)
    consumption       0.15 .. 0.22 kWh/km
    target SoC        0.7 .. 0.9
    mobility          its own trip seed; weekday use 0.6 .. 0.95, departure 6:30 .. 8:30, arrival 16 .. 19 h,
                      median trip 20 .. 50 km (weekends as devices.EV's default)

House sets, disjoint draws (a house's parameters are a function of its draw number only, so the sets are stable):
    train     32 houses, the training pool; an episode draws one at random (MultiHouseEnv)
    val        6 houses, checkpoint selection (validation_return: each house on every third validation week)
    test       8 new houses + 'reference' (the fixed ev_hp house of all earlier results), on the test year
    ood        4 houses outside the ranges, one axis each: 20 kWp PV, 3 kWp PV, 110 kWh car on a long commute,
               2.3 kW wallbox (a household socket); context values outside what the agent was trained on
Generalization is measured on new houses AND a new year: train houses on the train split, test houses on 2023.

context   'observed': the obs carries ev_context (car) and house_context (PV, demand, battery, heat pump), so the
          agent knows which house it is in. 'hidden': both are left out; the agent has to infer the house from
          what it sees (load and PV levels, how fast the car fills): a contextual MDP with a latent context.
The observation features are the same for every house (temperature is observed even without a heat pump), so one
network fits all of them.

Scripted baselines and the oracles do not depend on the learner, so their per-house test tables are computed once
and cached in results/houses/ (keyed by a hash of the house set; --refresh after changing env or oracle code).

usage: python houses.py [--sets test ood] [--refresh]     list the houses and compute (cache) the scripted tables
"""

from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
import gymnasium as gym
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "results" / "houses"
HOUSE_SEED = 20261121                     # fixes every draw; a house is a function of (HOUSE_SEED, draw number)
N_TRAIN, N_VAL, N_TEST = 32, 6, 8
WALLBOXES_KW = (3.7, 7.4, 11.0)
DEFAULT_WEEKDAY = dict(p_use=0.85, dep_mean=7.5, dep_sd=1.0, arr_mean=17.5, arr_sd=1.5, km_median=35.0, km_sigma=0.6)

# the fixed house of all earlier results (scenario.make_env(df, household='ev_hp'))
REFERENCE = dict(name="reference", pv_kwp=10.0, annual_kwh=4000.0, capacity_kwh=10.0, max_power_kw=5.0,
                 heat_pump=dict(), ev=dict())


def sample_house(name, draw):
    rng = np.random.default_rng([HOUSE_SEED, draw])
    cap = round(float(rng.uniform(5.0, 15.0)), 1)
    hp = dict(heat_loss_kw_per_k=round(float(rng.uniform(0.15, 0.30)), 3),
              emitter=str(rng.choice(["radiator", "floor"]))) if rng.random() < 0.5 else None
    weekday = dict(DEFAULT_WEEKDAY, p_use=round(float(rng.uniform(0.6, 0.95)), 2),
                   dep_mean=round(float(rng.uniform(6.5, 8.5)), 1), arr_mean=round(float(rng.uniform(16.0, 19.0)), 1),
                   km_median=round(float(rng.uniform(20.0, 50.0)), 0))
    ev = dict(capacity_kwh=round(float(rng.uniform(40.0, 100.0)), 0), max_power_kw=float(rng.choice(WALLBOXES_KW)),
              kwh_per_km=round(float(rng.uniform(0.15, 0.22)), 3), target_soc=round(float(rng.uniform(0.7, 0.9)), 2),
              seed=1000 + draw, weekday=weekday)
    return dict(name=name, pv_kwp=round(float(rng.uniform(5.0, 15.0)), 1),
                annual_kwh=round(float(rng.uniform(2500.0, 6000.0)), -1),
                capacity_kwh=cap, max_power_kw=round(cap / 2, 2), heat_pump=hp, ev=ev)


OOD = [
    dict(REFERENCE, name="ood-big-pv", pv_kwp=20.0, capacity_kwh=20.0, max_power_kw=10.0, heat_pump=None,
         ev=dict(seed=2001)),
    dict(REFERENCE, name="ood-small-pv", pv_kwp=3.0, capacity_kwh=5.0, max_power_kw=2.5, ev=dict(seed=2002)),
    dict(REFERENCE, name="ood-big-car", ev=dict(capacity_kwh=110.0, kwh_per_km=0.24, seed=2003,
                                                weekday=dict(DEFAULT_WEEKDAY, km_median=70.0))),
    dict(REFERENCE, name="ood-socket", ev=dict(capacity_kwh=77.0, max_power_kw=2.3, seed=2004)),
]

HOUSE_SETS = {
    "train": [sample_house(f"train{k:02d}", k) for k in range(N_TRAIN)],
    "val": [sample_house(f"val{k}", 100 + k) for k in range(N_VAL)],
    "test": [REFERENCE] + [sample_house(f"test{k}", 200 + k) for k in range(N_TEST)],
    "ood": OOD,
}


def house_overrides(house):
    """make_env keyword arguments of a house."""
    return {k: v for k, v in house.items() if k != "name"}


def obs_features(context="observed"):
    from microgrid_env import SNAPSHOT
    if context not in ("observed", "hidden"):
        raise ValueError(f"unknown context {context!r}; choose 'observed' or 'hidden'")
    return SNAPSHOT + ("ev", "temperature") + (("ev_context", "house_context") if context == "observed" else ())


def make_house_env(df, house, control="ev", context="observed", **kw):
    """The env of one house; the same observation layout for every house."""
    from scenario import make_env
    household = "ev_hp" if house.get("heat_pump") is not None else "ev"
    env = make_env(df, household=household, control=control, obs_features=obs_features(context),
                   **{**house_overrides(house), **kw})
    env.house = house["name"]
    return env


def make_envs(df, house_set, **kw):
    return [make_house_env(df, h, **kw) for h in HOUSE_SETS[house_set]]


class MultiHouseEnv(gym.Env):
    """One env per house; every reset draws a house uniformly (or options['house'] = index), then delegates.
    Attribute access is forwarded to the current house's env, so TrainingReward, the baselines and the evaluation
    helpers work on it unchanged."""
    metadata = {"render_modes": []}

    def __init__(self, envs):
        self.envs = list(envs)
        self.current = self.envs[0]
        self.observation_space = self.current.observation_space
        self.action_space = self.current.action_space
        for e in self.envs:
            if e.observation_space.shape != self.observation_space.shape or e.action_space.shape != self.action_space.shape:
                raise ValueError("every house needs the same observation and action layout")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = dict(options or {})
        k = options.pop("house", None)
        self.current = self.envs[int(self.np_random.integers(len(self.envs))) if k is None else int(k)]
        return self.current.reset(seed=int(self.np_random.integers(2 ** 31)), options=options)

    def step(self, action):
        return self.current.step(action)

    def __getattr__(self, name):
        if name in ("envs", "current") or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.current, name)


def make_multi_env(df, house_set="train", **kw):
    return MultiHouseEnv(make_envs(df, house_set, **kw))


# ---- evaluation ------------------------------------------------------------------------------------------------

def validation_return(envs, policy, stride=3):
    """Mean env return (EUR/week, higher is better) for checkpoint selection: house k on every 'stride'-th validation
    week starting at week k % stride, so every week is scored by len(envs) / stride houses."""
    from evaluation import run_episode, summarize, WEEK
    totals = [summarize(run_episode(env, policy, s), env)["total"]
              for k, env in enumerate(envs) for s in env.episode_starts(episode_hours=WEEK)[k % stride::stride]]
    return -float(np.mean(totals))


SCRIPTED_HEURISTICS = ("no battery", "rule-based", "solar-only", "solar-only + EV pv-surplus", "solar-only + EV cheapest")
ORACLES = ("oracle (EV only)", "oracle (cyclic)")


def _digest(house_set, split):
    return hashlib.sha1(json.dumps([HOUSE_SETS[house_set], split], sort_keys=True).encode()).hexdigest()[:10]


def scripted_table(house_set, splits, split="test", refresh=False, verbose=True):
    """Continuous-run weekly table of the scripted heuristics and both oracles, for every house of the set; cached."""
    from baselines import no_battery, solar_only, make_rule_based, make_policy
    from evaluation import compare, policy_entry, oracle_entry, scripted
    path = CACHE / f"scripted_{house_set}_{split}_{_digest(house_set, split)}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path)
    frames = []
    for house in HOUSE_SETS[house_set]:
        t0 = time.time()
        env = make_house_env(splits[split], house, control="full")
        entries = {
            "no battery": policy_entry(env, scripted(no_battery)),
            "rule-based": policy_entry(env, scripted(make_rule_based(env, ref_df=splits["train"]))),
            "solar-only": policy_entry(env, scripted(solar_only)),
            "solar-only + EV pv-surplus": policy_entry(env, scripted(make_policy("solar-only", "pv-surplus"))),
            "solar-only + EV cheapest": policy_entry(env, scripted(make_policy("solar-only", "cheapest"))),
            "oracle (EV only)": oracle_entry(env, cyclic=True, battery="self-consumption"),
            "oracle (cyclic)": oracle_entry(env, cyclic=True),
        }
        t = compare(entries, "continuous")
        t.insert(1, "house", house["name"])
        frames.append(t)
        if verbose:
            print(f"  {house_set}/{house['name']}: scripted rows and oracles in {time.time() - t0:.0f} s", flush=True)
    res = pd.concat(frames, ignore_index=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    res.to_csv(tmp, index=False)
    os.replace(tmp, path)                     # atomic: parallel seeds never read a half-written cache
    return res


def learner_table(policies, house_set, splits, split="test", control="ev", context="observed"):
    """policies: {name: policy(env, obs)}; continuous run per house of the set."""
    from evaluation import evaluate_continuous
    frames = []
    for house in HOUSE_SETS[house_set]:
        env = make_house_env(splits[split], house, control=control, context=context)
        for name, pol in policies.items():
            t = evaluate_continuous(env, pol)
            t.insert(0, "policy", name)
            t.insert(1, "house", house["name"])
            frames.append(t)
    return pd.concat(frames, ignore_index=True)


def test_table(policies, house_set, splits, split="test", control="ev", context="observed"):
    """Scripted heuristics, then the learners, then the oracles; one row per policy, house and week.
    experiments/seeds.py averages it over houses and weeks like a single-house table."""
    s = scripted_table(house_set, splits, split)
    learned = learner_table(policies, house_set, splits, split, control, context)
    return pd.concat([s[s["policy"].isin(SCRIPTED_HEURISTICS)], learned, s[s["policy"].isin(ORACLES)]], ignore_index=True)


def per_house(res):
    """Mean weekly cost, house x policy."""
    houses = list(dict.fromkeys(res["house"]))
    order = list(dict.fromkeys(res["policy"]))
    return res.groupby(["house", "policy"])["total"].mean().unstack("policy").reindex(index=houses, columns=order)


def report(res, title=""):
    tot = per_house(res)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(f"\n=== {title}: mean weekly cost per house (EUR/week, lower is better) ===")
    print(pd.concat([tot, tot.mean().to_frame("mean").T]).round(2).to_string())
    if "oracle (EV only)" in tot:
        gap = 100 * tot.rsub(tot["solar-only"], axis=0).div(tot["solar-only"] - tot["oracle (EV only)"], axis=0)
        m = tot.mean()
        pooled = 100 * (m["solar-only"] - m) / (m["solar-only"] - m["oracle (EV only)"])
        print(f"\n=== {title}: EV gap closed per house (% of solar-only -> EV-only oracle) ===")
        print(pd.concat([gap, pooled.to_frame("pooled").T]).round(0).to_string())
    return tot


def describe(house_set):
    rows = []
    for h in HOUSE_SETS[house_set]:
        from devices import EV
        car = EV(**h["ev"])
        hp = h["heat_pump"]
        rows.append(dict(house=h["name"], pv_kwp=h["pv_kwp"], kwh_year=h["annual_kwh"], battery_kwh=h["capacity_kwh"],
                         heat_pump="-" if hp is None else f"{hp.get('heat_loss_kw_per_k', 0.22)} {hp.get('emitter', 'radiator')}",
                         car_kwh=car.capacity_kwh, wallbox_kw=car.max_power_kw, kwh_per_km=car.kwh_per_km,
                         target=car.target_soc, km_median=car.weekday.km_median, weekday_use=car.weekday.p_use,
                         max_km=round(car.max_km)))
    return pd.DataFrame(rows).set_index("house")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sets", nargs="*", default=["test", "ood"], help="house sets whose scripted tables to compute")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--refresh", action="store_true", help="recompute cached tables")
    args = p.parse_args()
    pd.set_option("display.width", 250)
    for name in HOUSE_SETS:
        print(f"\n--- house set '{name}' ({len(HOUSE_SETS[name])} houses) ---")
        print(describe(name).to_string())
    from data import load_dataset, chronological_split
    splits = chronological_split(load_dataset())
    for name in args.sets:
        report(scripted_table(name, splits, args.split, refresh=args.refresh), f"{name} houses, {args.split} split")
