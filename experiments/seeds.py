"""Run one training configuration over several seeds and report mean ± std across the seeds.

usage: python experiments/seeds.py NAME [--seeds 5] [--first-seed 0] [--jobs 5] [--script sac.py]
                                   [--aggregate-only] [-- ARGS FOR THE TRAINING SCRIPT]
example: python experiments/seeds.py pclip --seeds 5 -- --action-mode power --clip-penalty 0.05

Every seed k is its own process:  python SCRIPT runs/seeds/NAME/s<k> --seed k --tag NAME --threads 1 ARGS
with its console output in runs/seeds/NAME/s<k>.log. A seed whose test table already exists is skipped, so
an interrupted batch resumes where it stopped (delete the seed folder to rerun it). The training script must
accept that command line and write <tag>_log.csv (step, validation_return) and <tag>_test_continuous.csv
(evaluation.compare output) into its out folder, as sac.py does.

Outputs, after all seeds have finished:
    results/seeds/NAME_per_seed.csv    one row per seed and learned policy: test-year mean weekly cost etc.
    results/seeds/NAME_summary.csv     mean, std (ddof 1), min, max over seeds per policy and metric;
                                       the scripted baselines and the oracle are deterministic and appear once
    figures/seeds_NAME_training.png    validation return vs env steps: mean ± std band and every seed

Metrics per seed (test year, continuous protocol, EUR/week; lower is better, negative = net income from exports):
    total            mean weekly cost (cost + degradation)
    vs_solar_only    total minus solar-only's total (negative = better than solar-only)
    vs_best_heuristic  total minus the best scripted heuristic's (with an EV usually 'solar-only + EV cheapest')
    value_share      % of the battery's value (no battery -> cyclic oracle) the policy captures
    gap_closed       % of the rule-based -> cyclic oracle gap closed
    plus the behaviour counters of evaluation.summarize (kWh/week).
"""

import argparse, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SCRIPTED = ("no battery", "dump", "rule-based", "rule-based (tuned)", "solar-only", "oracle (cyclic)", "oracle (free end)")
is_scripted = lambda policy: policy in SCRIPTED or policy.startswith("solar-only + EV")      # EV heuristics (evaluation.py)
BEHAVIOUR = ["solar_stored_kwh", "grid_charge_kwh", "discharged_to_export_kwh", "throughput_kwh", "clipped_steps"]
METRICS = ["total", "vs_solar_only", "vs_best_heuristic", "value_share", "gap_closed"] + BEHAVIOUR


def run_seed(name, seed, script, extra, out_root):
    out = out_root / f"s{seed}"
    if (out / f"{name}_test_continuous.csv").exists():
        print(f"seed {seed}: done already, skipped")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, script, str(out), "--seed", str(seed), "--tag", name, "--threads", "1", *extra]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    print(f"seed {seed}: started  ({' '.join(cmd[1:])})", flush=True)
    with open(out_root / f"s{seed}.log", "w") as log:
        code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    print(f"seed {seed}: {'finished' if code == 0 else f'FAILED (exit {code}), see {out_root}/s{seed}.log'}", flush=True)
    return code


def per_policy(table):
    """Mean weekly metrics of one seed's test table, one row per policy, with the derived shares."""
    m = table.groupby("policy", sort=False)[["total"] + [b for b in BEHAVIOUR if b in table]].mean()
    t = m["total"]
    m["vs_solar_only"] = t - t["solar-only"]
    m["vs_best_heuristic"] = t - t[[p for p in t.index if is_scripted(p) and not p.startswith("oracle")]].min()
    m["value_share"] = 100 * (t["no battery"] - t) / (t["no battery"] - t["oracle (cyclic)"])
    m["gap_closed"] = 100 * (t["rule-based"] - t) / (t["rule-based"] - t["oracle (cyclic)"])
    return m[[c for c in METRICS if c in m]]


def aggregate(name, seeds, out_root):
    rows, curves, scripted = [], {}, None
    for s in seeds:
        out = out_root / f"s{s}"
        test = out / f"{name}_test_continuous.csv"
        if not test.exists():
            print(f"seed {s}: no test table, left out of the aggregate")
            continue
        m = per_policy(pd.read_csv(test))
        learned = [p for p in m.index if not is_scripted(p)]
        rows.append(m.loc[learned].reset_index().assign(seed=s))
        scripted = m.drop(index=learned) if scripted is None else scripted
        log = out / f"{name}_log.csv"
        if log.exists():
            curves[s] = pd.read_csv(log).set_index("step")["validation_return"]
    if not rows:
        raise SystemExit("no finished seeds to aggregate")

    metrics = [c for c in METRICS if c in rows[0]]
    per_seed = pd.concat(rows, ignore_index=True)[["seed", "policy"] + metrics]
    stats = per_seed.groupby("policy", sort=False)[metrics].agg(["mean", "std", "min", "max"])
    stats.insert(0, ("n_seeds", ""), per_seed.groupby("policy", sort=False).size())
    base = scripted.copy()
    base.columns = pd.MultiIndex.from_product([base.columns, ["mean"]])
    base.insert(0, ("n_seeds", ""), np.nan)
    summary = pd.concat([stats, base])

    res = ROOT / "results" / "seeds"; res.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(res / f"{name}_per_seed.csv", index=False)
    flat = summary.copy()
    flat.columns = [f"{a}_{b}" if b else a for a, b in flat.columns]
    flat.rename_axis("policy").to_csv(res / f"{name}_summary.csv")

    n = len(per_seed["seed"].unique())
    print(f"\n=== {name}: test year 2023, continuous run, {n} seeds; mean ± std across seeds (EUR/week, lower is better) ===")
    for policy, r in stats.iterrows():
        f = lambda k, unit="": f"{r[(k, 'mean')]:7.2f} ± {r[(k, 'std')]:.2f}{unit}"
        print(f"{policy:<20} cost {f('total')}   vs solar-only {f('vs_solar_only')}   vs best heuristic {f('vs_best_heuristic')}   value share {f('value_share', ' %')}"
              f"   gap closed {f('gap_closed', ' %')}   exported from battery {f('discharged_to_export_kwh', ' kWh')}")
    print("\nper seed (cost):")
    print(per_seed.pivot(index="seed", columns="policy", values="total").round(2).to_string())
    print("\nscripted baselines (deterministic):")
    print(scripted[["total", "vs_solar_only", "vs_best_heuristic", "value_share", "gap_closed"]].round(2).to_string())

    if curves:
        c = pd.DataFrame(curves).sort_index()
        fig, ax = plt.subplots(figsize=(8, 4))
        for s in c:
            ax.plot(c.index, c[s], lw=0.6, alpha=0.4, color="tab:blue")
        mu, sd = c.mean(axis=1), c.std(axis=1)
        ax.plot(c.index, mu, color="tab:blue", lw=2, label=f"mean over {c.shape[1]} seeds")
        ax.fill_between(c.index, mu - sd, mu + sd, color="tab:blue", alpha=0.2, label="± 1 std")
        ax.set_xlabel("env steps"); ax.set_ylabel("validation return (EUR/week)"); ax.set_title(name)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(ROOT / "figures" / f"seeds_{name}_training.png", dpi=130)
        plt.close(fig)
    print(f"\nwrote results/seeds/{name}_per_seed.csv, results/seeds/{name}_summary.csv" + (f", figures/seeds_{name}_training.png" if curves else ""))
    return summary


if __name__ == "__main__":
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", help="run name: folder runs/seeds/NAME and the --tag of every seed")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--first-seed", type=int, default=0)
    p.add_argument("--jobs", type=int, default=0, help="parallel processes (default: one per seed, at most CPUs - 2)")
    p.add_argument("--script", default="sac.py")
    p.add_argument("--aggregate-only", action="store_true")
    args = p.parse_args(argv)

    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    out_root = ROOT / "runs" / "seeds" / args.name
    out_root.mkdir(parents=True, exist_ok=True)
    if not args.aggregate_only:
        (out_root / "command.txt").write_text(" ".join([args.script, *extra]) + "\n")
        jobs = args.jobs or max(1, min(len(seeds), (os.cpu_count() or 2) - 2))
        with ThreadPoolExecutor(jobs) as pool:
            codes = list(pool.map(lambda s: run_seed(args.name, s, args.script, extra, out_root), seeds))
        if any(codes):
            print(f"{sum(c != 0 for c in codes)} seed(s) failed; aggregating the rest")
    aggregate(args.name, seeds, out_root)
