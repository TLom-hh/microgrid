"""How do learned policies handle the EV? Dreamer vs SAC vs the 'cheapest' heuristic on the test year (continuous run).

Per departure: the car's SoC just before departure (before any public top-up) and the shortfall bought publicly.
Per plugged-in hour: wallbox power against hours to departure and hour of day, i.e. does the policy react to the deadline?
Plus a week-by-week comparison of the worst Dreamer week and a typical one.

usage: python experiments/ev_diagnosis.py [--dreamer RUN_DIR] [--sac ACTOR.pt] [--out DIR]
Writes DIR/ev_departures.csv, DIR/ev_diagnosis.png, DIR/ev_week_<date>.png.
"""

import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from data import load_dataset, chronological_split
from scenario import make_env
from baselines import make_policy
import evaluation as ev
from plotting import C, _style

COLORS = {"heuristic (cheapest)": C["muted"], "SAC": C["price"], "Dreamer": C["ev"]}


def run_all(env, policies):
    o = ev.continuous_options(env)
    return {name: ev.run_episode(env, pol, o["start_step"], o["soc"], o["episode_hours"]) for name, pol in policies.items()}


def departures(hist, env, start):
    """One row per departure: the step leading into it carries the public top-up."""
    deps = env.df["ev_departs"].values[start + 1: start + 1 + len(hist)].astype(bool)
    h = hist[deps]
    cap = env.ev.capacity_kwh
    return pd.DataFrame({"time": h.index, "soc_at_departure": h["ev_soc"] - h["ev_shortfall_kwh"] / cap,
                         "shortfall_kwh": h["ev_shortfall_kwh"].values})


def hours_to_departure(env, start, n):
    return env.df["ev_hours_to_departure"].values[start: start + n]


def week_figure(hists, t0, path):
    names = list(hists)
    fig, axs = plt.subplots(2 + len(names), 1, figsize=(12, 3 + 2.0 * (2 + len(names))), sharex=True)
    fig.patch.set_facecolor(C["surface"])
    week = {n: h.loc[t0: t0 + pd.Timedelta(hours=167)] for n, h in hists.items()}
    ref = week[names[0]]
    t = ref.index

    ax = axs[0]
    ax.plot(t, ref["buy_price"], color=C["price"], lw=2, label="buy price (EUR/kWh)")
    ax2 = ax.twinx()
    ax2.plot(t, ref["solar_kw"], color=C["solar"], lw=1.5, label="solar kW")
    ax2.set_ylabel("solar kW", color=C["ink2"], fontsize=9)
    _style(ax, "EUR/kWh")
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    ax = axs[1]
    away = ~ref["ev_home"].values.astype(bool)
    ax.fill_between(t, 0, 1, where=away, step="post", color=C["muted"], alpha=0.15, lw=0, label="EV away")
    for n, h in week.items():
        ax.plot(t, h["ev_soc"], color=COLORS[n], lw=2, label=f"{n}: public {h['ev_shortfall_kwh'].sum():.0f} kWh")
    ax.axhline(0.8, color=C["ink2"], lw=1, ls=":")
    ax.set_ylim(0, 1)
    _style(ax, "EV SoC")
    ax.legend(loc="lower left", fontsize=8, frameon=False, ncol=4)

    for ax, n in zip(axs[2:], names):
        h = week[n]
        ax.fill_between(t, -5, 11, where=away, step="post", color=C["muted"], alpha=0.15, lw=0)
        ax.step(t, h["ev_kw"], where="post", color=COLORS[n], lw=1.8, label="EV charging")
        ax.step(t, h["battery_kw"], where="post", color=C["battery"], lw=1.4, label="battery (+ charge)")
        ax.axhline(0, color=C["grid"], lw=1)
        ax.set_ylim(-5.5, 11.5)
        cost = (h["cost"] + h["degradation"]).sum()
        _style(ax, f"{n}\nkW")
        ax.set_title(f"{n}: {cost:.2f} EUR this week", fontsize=9, color=C["ink2"], loc="left")
        ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    fig.suptitle(f"Week from {t0:%d %b %Y}: EV handling", color=C["ink"], fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def summary_figure(hists, deps, env, start, path):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.patch.set_facecolor(C["surface"])
    bins = np.arange(0, 25, 2)
    for n, h in hists.items():
        home = h["ev_home"].values.astype(bool)
        htd = hours_to_departure(env, start, len(h))[home]
        kw = h["ev_kw"].values[home]
        idx = np.digitize(htd, bins) - 1
        means = [kw[idx == i].mean() if (idx == i).any() else np.nan for i in range(len(bins) - 1)]
        axs[0].plot(bins[:-1] + 1, means, color=COLORS[n], lw=2, marker="o", ms=3, label=n)
        by_hour = h[home].groupby(h.index[home].hour)["ev_kw"].mean()
        axs[1].plot(by_hour.index, by_hour.values, color=COLORS[n], lw=2, label=n)
        axs[2].hist(deps[n]["soc_at_departure"], bins=np.linspace(0, 1, 21), histtype="step", lw=2, color=COLORS[n],
                    label=f"{n}: {(deps[n]['shortfall_kwh'] > 0.5).mean():.0%} short")
    axs[0].set_xlabel("hours to departure (24 = day or more)", fontsize=9, color=C["ink2"])
    _style(axs[0], "mean EV kW while plugged in")
    axs[1].set_xlabel("hour of day", fontsize=9, color=C["ink2"])
    _style(axs[1], "mean EV kW while plugged in")
    axs[2].axvline(0.8, color=C["ink2"], lw=1, ls=":")
    axs[2].set_xlabel("EV SoC at departure, before public top-up (target 0.8)", fontsize=9, color=C["ink2"])
    _style(axs[2], "departures")
    for ax in axs:
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("EV charging on the test year", color=C["ink"], fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dreamer", default="runs/dreamer_s0")
    p.add_argument("--sac", default="runs/seeds/ev_hp/s0/ev_hp_best.pt")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = Path(args.out or Path(args.dreamer) / "diagnosis"); out.mkdir(parents=True, exist_ok=True)

    from dreamer import load_player, dreamer_policy
    splits = chronological_split(load_dataset())
    env = make_env(splits["test"], household="ev_hp")
    best_ckpt = (Path(args.dreamer) / "dreamer_best_checkpoint.txt").read_text().strip()
    policies = {"heuristic (cheapest)": ev.scripted(make_policy("solar-only", "cheapest")),
                "SAC": ev.load_actor(args.sac, env),
                "Dreamer": dreamer_policy(load_player(best_ckpt))}
    hists = run_all(env, policies)
    start = ev.continuous_options(env)["start_step"]

    deps = {n: departures(h, env, start) for n, h in hists.items()}
    pd.concat({n: d for n, d in deps.items()}, names=["policy"]).to_csv(out / "ev_departures.csv")
    for n, d in deps.items():
        h = hists[n]
        print(f"{n:22s} departures {len(d)}  short {(d['shortfall_kwh'] > 0.5).mean():5.0%}  "
              f"mean SoC at departure {d['soc_at_departure'].mean():.2f}  public {d['shortfall_kwh'].sum():6.0f} kWh  "
              f"EV kWh {h['ev_kw'].sum():6.0f}  battery discharge {(-h['battery_kw'].clip(upper=0)).sum():6.0f} kWh")
    summary_figure(hists, deps, env, start, out / "ev_diagnosis.png")

    weekly = hists["Dreamer"]["ev_shortfall_kwh"].resample("W-SUN", label="left", closed="left").sum()
    weekly = weekly[weekly.index >= hists["Dreamer"].index[0].normalize()]
    worst = weekly.idxmax()
    typical = (weekly - weekly.median()).abs().idxmin()
    for t0 in dict.fromkeys([worst, typical]):
        week_figure(hists, t0, out / f"ev_week_{t0:%Y%m%d}.png")
        print(f"week {t0:%Y-%m-%d}: Dreamer public {weekly[t0]:.0f} kWh -> {out / f'ev_week_{t0:%Y%m%d}.png'}")
