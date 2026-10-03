"""Evaluation protocol shared by every controller.

Data: the chronological split from data.py. Training touches the train split only; hyper-parameters,
rule-based thresholds and checkpoints are selected on the validation split; the test split is used once.

Two protocols:
    weeks        independent one-week episodes on every non-overlapping week of a split, start SoC 0.5.
                 Cheap; used for validation during training (validation_return = mean env return, EUR/week).
    continuous   one uninterrupted run over the whole split (52 weeks on test) with the SoC carried over;
                 the weekly table is the calendar breakdown of that run. The battery never 'ends', so there is
                 nothing to gain from draining it at week boundaries, which matches the continuing-task view of
                 the critic. The oracle is then solved over the whole horizon with end SoC >= start SoC.

usage: python evaluation.py [OUT_DIR] [name=actor.pt[:vpg] ...] [--data CSV] [--split test|val|train]
                            [--protocol continuous|weeks] [--action-mode power|target_soc|residual]
Baselines and the oracle run on the physical (power) interface; --action-mode is the interface the actor
checkpoints were trained with. Writes OUT_DIR/eval_<split>_<protocol>.csv (one row per policy and week).
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from data import load_dataset, chronological_split, DATASET
from oracle import run_oracle
from baselines import no_battery, dump, solar_only, make_rule_based
from scenario import make_env

WEEK = 24 * 7
EVAL_SOC = 0.5
EVAL_SEED = 0
RULE_GRID = [(lo, hi) for lo in (0.0, 0.1, 0.25, 0.4) for hi in (0.6, 0.75, 0.9)]

def scripted(fn):
    """fn(env) -> action"""
    return lambda env, obs: fn(env)

def actor_policy(actor):
    """Deterministic torch actor (sac.Actor or vpg.GaussianPolicy): obs -> action in [-1, 1]."""
    import torch

    def policy(env, obs):
        with torch.no_grad():
            a, _ = actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0), deterministic=True)
        return a.numpy().reshape(-1)
    return policy

def load_actor(path, env):
    """Actor checkpoint -> policy. 'path:vpg' loads a vpg.GaussianPolicy instead of sac.Actor."""
    import torch
    path, _, flag = str(path).partition(":")
    if flag == "vpg":
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))
        from vpg import GaussianPolicy
        actor = GaussianPolicy(env.observation_space.shape[0], 1)
    else:
        from sac import Actor
        actor = Actor(env.observation_space.shape[0], 1)
    actor.load_state_dict(torch.load(path))
    actor.eval()
    return actor_policy(actor)

def run_episode(env, policy, start_step, soc=EVAL_SOC, hours=None):
    """Roll policy from a fixed start and SoC; returns the history frame. The env's random stream (the noise
    on the solar forecast in the observation) is seeded from the start step, so every policy sees the same
    forecasts for a given week and every scoring run gives exactly the same numbers."""
    options = {"start_step": int(start_step), "soc": soc}
    if hours is not None:
        options["episode_hours"] = int(hours)
    obs, _ = env.reset(seed=EVAL_SEED + int(start_step), options=options)
    done = False
    while not done:
        obs, _, term, trunc, _ = env.step(policy(env, obs))
        done = term or trunc
    return env.history_frame()

def summarize(hist, env):
    """Cost and behaviour counters of one history frame (one week or one whole run)."""
    charging = hist["battery_kw"].clip(lower=0)
    discharging = (-hist["battery_kw"]).clip(lower=0)
    surplus = (hist["solar_kw"] - hist["load_kw"]).clip(lower=0)
    export = (-hist["grid_kw"]).clip(lower=0)
    return dict(
        week=hist.index[0].strftime("%Y-%m-%d"), month=hist.index[0].strftime("%b"),
        total=(hist["cost"] + hist["degradation"]).sum(), cost=hist["cost"].sum(), degradation=hist["degradation"].sum(),
        soc_start=hist["soc_before"].iloc[0], soc_end=hist["soc"].iloc[-1],
        solar_stored_kwh=np.minimum(charging, surplus).sum() * env.dt,
        grid_charge_kwh=(charging - surplus).clip(lower=0).sum() * env.dt,
        discharged_to_export_kwh=np.minimum(discharging, export).sum() * env.dt,
        throughput_kwh=hist["battery_kw"].abs().sum() * env.dt,
        clipped_steps=int(hist["clipped"].sum()))

def weekly_table(hist, env):
    """Per-week metrics of a (long) history frame, cut into calendar weeks from its first day (DST-safe)."""
    wall = hist.index.tz_localize(None)                      # wall-clock time: a DST day is still one day
    week = (wall.normalize() - wall[0].normalize()).days // 7
    return pd.DataFrame([summarize(block, env) for _, block in hist.groupby(week) if len(block) >= WEEK - 8])

def continuous_options(env, soc=EVAL_SOC):
    """reset options for one run from the first midnight of env.df to the end of its last whole week."""
    midnights = np.flatnonzero(env.df.index.hour == 0)
    starts = env.episode_starts(episode_hours=WEEK)
    k = int(np.searchsorted(midnights, starts[-1])) + 7
    end = int(midnights[k]) if k < len(midnights) else int(starts[-1] + WEEK)
    return {"start_step": int(starts[0]), "soc": soc, "episode_hours": end - int(starts[0])}

def evaluate_weeks(env, policy, soc=EVAL_SOC):
    """Independent one-week episodes on every non-overlapping week of env.df."""
    return pd.DataFrame([summarize(run_episode(env, policy, s, soc), env) for s in env.episode_starts(episode_hours=WEEK)])

def evaluate_continuous(env, policy, soc=EVAL_SOC):
    """One run over the whole split, SoC carried across weeks; returns the weekly table."""
    o = continuous_options(env, soc)
    return weekly_table(run_episode(env, policy, o["start_step"], o["soc"], o["episode_hours"]), env)

def oracle_weeks(env, soc=EVAL_SOC, cyclic=True):
    return pd.DataFrame([summarize(run_oracle(env, cyclic=cyclic, options={"start_step": int(s), "soc": soc}), env) for s in env.episode_starts(episode_hours=WEEK)])

def oracle_continuous(env, soc=EVAL_SOC, cyclic=True):
    return weekly_table(run_oracle(env, cyclic=cyclic, options=continuous_options(env, soc)), env)

def validation_return(env, policy, soc=EVAL_SOC):
    """Mean plain env return (EUR/week, higher is better) over the weeks of env.df; for training-time evals."""
    return -float(evaluate_weeks(env, policy, soc)["total"].mean())

def tune_rule_based(env, ref_df, grid=RULE_GRID, soc=EVAL_SOC):
    """Pick the price-quantile thresholds with the lowest weekly cost on env.df (the validation split).
    Quantiles are taken from ref_df (the training split). Returns (low_q, high_q, mean_cost)."""
    best = None
    for lo, hi in grid:
        cost = evaluate_weeks(env, scripted(make_rule_based(env, lo, hi, ref_df=ref_df)), soc)["total"].mean()
        if best is None or cost < best[2]:
            best = (lo, hi, float(cost))
    return best

def policy_entry(env, policy):
    return lambda protocol, soc: (evaluate_continuous if protocol == "continuous" else evaluate_weeks)(env, policy, soc)

def oracle_entry(env, cyclic=True):
    return lambda protocol, soc: (oracle_continuous if protocol == "continuous" else oracle_weeks)(env, soc, cyclic)

def baseline_entries(env, splits, verbose=True):
    """The scripted rows of every comparison table, for an env on any split. 'rule-based' keeps the default
    quantiles (0.25 / 0.75) and is the gap-closed denominator; 'rule-based (tuned)' has its thresholds
    picked on the validation split. Both take their price quantiles from the training split."""
    lo, hi, val_cost = tune_rule_based(make_env(splits["val"]), splits["train"])
    if verbose:
        print(f"rule-based thresholds tuned on val: low_q {lo}, high_q {hi} (val cost {val_cost:.2f} EUR/week; quantiles from train prices)")
    return {
        "no battery": policy_entry(env, scripted(no_battery)),
        "dump": policy_entry(env, scripted(dump)),
        "rule-based": policy_entry(env, scripted(make_rule_based(env, ref_df=splits["train"]))),
        "rule-based (tuned)": policy_entry(env, scripted(make_rule_based(env, lo, hi, ref_df=splits["train"]))),
        "solar-only": policy_entry(env, scripted(solar_only)),
    }

def compare(entries, protocol="continuous", soc=EVAL_SOC):
    """entries: {name: fn(protocol, soc) -> weekly table}. One row per policy and week."""
    frames = []
    for name, fn in entries.items():
        t = fn(protocol, soc)
        t.insert(0, "policy", name)
        frames.append(t)
    return pd.concat(frames, ignore_index=True)

def report(res, order=None, floor="no battery", ceiling="oracle (cyclic)", rule="rule-based", ref="solar-only"):
    order = order or list(dict.fromkeys(res["policy"]))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda x: f"{x:7.2f}")
    tot = res.pivot(index="week", columns="policy", values="total")[order]
    n = len(tot)
    summary = pd.DataFrame({"mean": tot.mean(), "std": tot.std(), "sum": tot.sum()})
    if ref in tot:
        diff = tot.sub(tot[ref], axis=0)
        summary[f"vs {ref}"] = diff.mean()
        summary["se"] = diff.std() / np.sqrt(n)
    mean = summary["mean"]
    if floor in mean and ceiling in mean:
        summary["value share %"] = 100 * (mean[floor] - mean) / (mean[floor] - mean[ceiling])
    if rule in mean and ceiling in mean:
        summary["gap closed %"] = 100 * (mean[rule] - mean) / (mean[rule] - mean[ceiling])
    last = res.groupby("policy").tail(1).set_index("policy")["soc_end"]
    summary["soc_end"] = last
    print(f"\n=== weekly cost EUR (cost + degradation, lower is better), {n} weeks; paired difference vs {ref} with its standard error ===")
    print(summary)
    print("\n=== mean weekly cost by month ===")
    months = res.drop_duplicates("week").set_index("week")["month"]
    by_month = tot.groupby(months).mean()
    by_month = by_month.reindex([m for m in ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec") if m in by_month.index])
    print(by_month.T)
    beh = ["solar_stored_kwh", "grid_charge_kwh", "discharged_to_export_kwh", "throughput_kwh", "clipped_steps"]
    print("\n=== behaviour, mean per week ===")
    print(res.groupby("policy")[beh].mean().loc[order])
    return summary

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", nargs="?", default="results/eval")
    p.add_argument("actors", nargs="*", help="name=checkpoint.pt, append ':vpg' for vpg.py policies")
    p.add_argument("--data", default=str(DATASET))
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--protocol", default="continuous", choices=["continuous", "weeks"])
    p.add_argument("--action-mode", default="power", choices=["power", "target_soc", "residual"])
    args = p.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    splits = chronological_split(load_dataset(args.data))
    df = splits[args.split]
    env = make_env(df)
    env_actor = env if args.action_mode == "power" else make_env(df, action_mode=args.action_mode)

    entries = baseline_entries(env, splits)
    for arg in args.actors:
        name, path = arg.split("=", 1)
        entries[name] = policy_entry(env_actor, load_actor(path, env_actor))
    entries["oracle (free end)"] = oracle_entry(env, cyclic=False)
    entries["oracle (cyclic)"] = oracle_entry(env, cyclic=True)

    res = compare(entries, args.protocol)
    print(f"\nsplit {args.split}: {df.index[0]:%Y-%m-%d} .. {df.index[-1]:%Y-%m-%d}, protocol {args.protocol}, start SoC {EVAL_SOC}")
    report(res, list(entries))
    res.to_csv(out / f"eval_{args.split}_{args.protocol}.csv", index=False)
