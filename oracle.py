"""Perfect-foresight oracle for the microgrid environment.

The oracle knows the horizon's prices, PV and load in advance and picks the battery schedule with the
lowest cost + degradation. Two solvers, both returning the same (total_cost, battery_kw, soc) triple:

    lp   linear program in cvxpy, solved with HiGHS (default). Variables: charge / discharge power,
         grid import / export, stored energy; constraints: SoC dynamics with efficiency, capacity, power
         limits, power balance. The LP is exact whenever buy_t >= max(sell_t, 0): then netting simultaneous
         import/export or charge/discharge never raises the cost. In hours where that fails (spot below
         -markup + feed-in, i.e. deep negative prices) a binary picks the sign of the grid flow and of the
         battery power, so the program stays exact (a MILP with a handful of binaries).
    dp   backward induction over a discretised SoC (401 levels = 0.025 kWh on 10 kWh): always exact for the
         single-power model up to the discretisation (a few cents per week). Kept as the cross-check.

Two horizon variants: free end (battery may end empty; lower bound for the raw cost) and cyclic
(end SoC >= start SoC; the fair ceiling for a continuing task).

Every schedule is replayed through the Gymnasium env: the replayed cost must equal the solver's cost
(run_oracle raises otherwise), which catches env/oracle physics mismatches.

usage: python oracle.py [OUT_DIR] [DATA_CSV] [--split test]
    self-test on toy cases, LP-vs-DP-vs-env check on every week of the split, oracle plots for a few weeks.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

LP_TOL = 1e-6


def solve_schedule_lp(buy, sell, net_kw, soc0, capacity_kwh, max_power_kw, efficiency, degradation_cost_per_kwh,
                      cyclic=True, dt=1.0, solver="HIGHS"):
    """Cost-optimal schedule by (MI)LP. net_kw is load - solar. Returns (total_cost, battery_kw, soc)."""
    import cvxpy as cp
    buy, sell, net_kw = (np.asarray(x, dtype=float) for x in (buy, sell, net_kw))
    T = len(buy)
    c, d = cp.Variable(T, nonneg=True), cp.Variable(T, nonneg=True)         # grid-side charge / discharge kW
    imp, exp = cp.Variable(T, nonneg=True), cp.Variable(T, nonneg=True)     # grid import / export kW
    e = cp.Variable(T + 1)                                                  # stored energy kWh
    cons = [e[0] == soc0 * capacity_kwh, e >= 0, e <= capacity_kwh,
            c <= max_power_kw, d <= max_power_kw,
            e[1:] == e[:-1] + (c * efficiency - d / efficiency) * dt,
            imp - exp == net_kw + c - d]
    if cyclic:
        cons.append(e[T] >= soc0 * capacity_kwh)
    bad = np.flatnonzero(buy < np.maximum(sell, 0.0) - 1e-12)              # hours where the relaxation is loose
    if len(bad):
        big = np.abs(net_kw[bad]) + max_power_kw
        zg, zb = cp.Variable(len(bad), boolean=True), cp.Variable(len(bad), boolean=True)
        cons += [imp[bad] <= cp.multiply(big, zg), exp[bad] <= cp.multiply(big, 1 - zg),
                 c[bad] <= max_power_kw * zb, d[bad] <= max_power_kw * (1 - zb)]
    cost = dt * (buy @ imp - sell @ exp + degradation_cost_per_kwh * cp.sum(c + d))
    prob = cp.Problem(cp.Minimize(cost), cons)
    prob.solve(solver=solver)
    if prob.status != "optimal":
        raise RuntimeError(f"oracle LP status {prob.status}")
    overlap = max(float(np.max(np.minimum(c.value, d.value))), float(np.max(np.minimum(imp.value, exp.value))))
    if overlap > 1e-4:
        raise RuntimeError(f"oracle LP relaxation is loose (simultaneous flows up to {overlap:.3g} kW)")
    battery_kw = np.where(c.value > d.value, c.value, -d.value)             # net, drop solver noise on the zero side
    return float(prob.value), battery_kw, np.asarray(e.value[1:]) / capacity_kwh


def solve_schedule_dp(buy, sell, net_kw, soc0, capacity_kwh, max_power_kw, efficiency, degradation_cost_per_kwh,
                      cyclic=True, n_levels=401, dt=1.0):
    """Cost-optimal schedule by backward induction over a discretised SoC. Same interface as the LP."""
    T = len(buy)
    grid = np.union1d(np.linspace(0.0, 1.0, n_levels), [soc0])
    dE = (grid[None, :] - grid[:, None]) * capacity_kwh                # [i, j]: change in stored energy
    bat = np.where(dE > 0, dE / efficiency, dE * efficiency) / dt      # grid-side kW for that transition
    feasible = np.abs(bat) <= max_power_kw + 1e-9
    deg = np.abs(bat) * dt * degradation_cost_per_kwh
    V = np.where(grid >= soc0 - 1e-9, 0.0, np.inf) if cyclic else np.zeros(len(grid))
    choice = np.zeros((T, len(grid)), dtype=int)
    for t in range(T - 1, -1, -1):                                     # backward induction
        g = (net_kw[t] + bat) * dt
        cost = np.where(g > 0, g * buy[t], g * sell[t]) + deg
        total = np.where(feasible, cost, np.inf) + V[None, :]
        choice[t] = np.argmin(total, axis=1)
        V = total[np.arange(len(grid)), choice[t]]
    i = int(np.flatnonzero(grid == soc0)[0])
    total = float(V[i])
    battery_kw, soc = np.empty(T), np.empty(T)
    for t in range(T):                                                 # forward pass
        j = choice[t, i]
        battery_kw[t], soc[t] = bat[i, j], grid[j]
        i = j
    return total, battery_kw, soc


def solve_schedule(*args, solver="lp", **kwargs):
    """Dispatch: solver 'lp' (cvxpy/HiGHS, exact) or 'dp' (discretised SoC, cross-check)."""
    if solver == "lp":
        return solve_schedule_lp(*args, **kwargs)
    if solver == "dp":
        return solve_schedule_dp(*args, **kwargs)
    raise ValueError(f"unknown solver {solver!r}")


def oracle(env, seed=None, options=None, cyclic=True, solver="lp"):
    """Solve the horizon that env.reset(seed, options) selects (see SmartMicrogridEnv.reset for options)."""
    env.reset(seed=seed, options=options)
    rows = env.df.iloc[env.start_step: env.start_step + env._hours]
    total, battery_kw, soc = solve_schedule(
        rows["buy_price"].values, rows["sell_price"].values,
        (rows["load_kw"] - rows["solar_kw"]).values, env.soc,
        env.capacity_kwh, env.max_power_kw, env.efficiency, env.degradation_cost_per_kwh,
        cyclic=cyclic, dt=env.dt, solver=solver)
    return dict(total=total, battery_kw=battery_kw, soc=soc, soc_start=env.soc)


def replay(env, battery_kw, seed=None, options=None):
    """Step the env through a fixed kW schedule (same reset as the oracle) and return its history frame."""
    env.reset(seed=seed, options=options)
    for kw in battery_kw:
        env.step(env.kw_to_action(kw))
    return env.history_frame()


def run_oracle(env, seed=None, cyclic=True, options=None, solver="lp", tol=1e-4):
    """Oracle schedule replayed through the env, with the solver-vs-env physics cross-check."""
    sol = oracle(env, seed, options, cyclic, solver)
    hist = replay(env, sol["battery_kw"], seed, options)
    replayed = (hist["cost"] + hist["degradation"]).sum()
    if abs(replayed - sol["total"]) > tol:
        raise RuntimeError(f"oracle/env mismatch ({solver}, seed {seed}, options {options}): "
                           f"solver {sol['total']:.6f} vs env {replayed:.6f} EUR")
    return hist


def selftest():
    """Hand-checkable toy cases for both solvers."""
    for solver in ("lp", "dp"):
        # buy 1 kWh in a cheap hour, use it in the expensive one
        total, bat, _ = solve_schedule(buy=np.array([0.1, 0.1, 1.0]), sell=np.zeros(3), net_kw=np.array([0.0, 0.0, 1.0]),
                                       soc0=0.0, capacity_kwh=10.0, max_power_kw=5.0, efficiency=1.0,
                                       degradation_cost_per_kwh=0.0, cyclic=False, solver=solver)
        assert abs(total - 0.1) < 1e-6, (solver, total)
        assert abs(bat.sum()) < 1e-6 and abs(bat[2] + 1.0) < 1e-6, (solver, bat)
        # efficiency + degradation: storing 1 kWh of surplus and using it later saves buy - deg*(1/eta+eta) ... vs export
        total, bat, _ = solve_schedule(buy=np.array([0.3, 0.3]), sell=np.array([0.08, 0.08]), net_kw=np.array([-2.0, 2.0]),
                                       soc0=0.0, capacity_kwh=10.0, max_power_kw=5.0, efficiency=0.9,
                                       degradation_cost_per_kwh=0.02, cyclic=False, solver=solver)
        # store all 2 kWh surplus (1.8 stored, 1.62 back), buy the remaining 0.38 kWh at 0.30, degradation on 3.62 kWh
        expected = 0.38 * 0.3 + 0.02 * (2.0 + 1.62)
        assert abs(total - expected) < 1e-3, (solver, total, expected)
    # buy price below the feed-in price: the plain LP relaxation would import and export at once; the binaries stop it
    total, bat, _ = solve_schedule_lp(buy=np.array([0.02, 0.5]), sell=np.array([0.08, 0.08]), net_kw=np.array([1.0, 1.0]),
                                      soc0=0.0, capacity_kwh=10.0, max_power_kw=5.0, efficiency=1.0,
                                      degradation_cost_per_kwh=0.0, cyclic=False)
    total_dp, _, _ = solve_schedule_dp(buy=np.array([0.02, 0.5]), sell=np.array([0.08, 0.08]), net_kw=np.array([1.0, 1.0]),
                                       soc0=0.0, capacity_kwh=10.0, max_power_kw=5.0, efficiency=1.0,
                                       degradation_cost_per_kwh=0.0, cyclic=False, n_levels=1001)
    assert abs(total - total_dp) < 1e-6, (total, total_dp)     # charge 1 kWh at 0.02, cover the 0.5 hour from it


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from data import load_dataset, chronological_split
    from plotting import plot_episode
    from scenario import make_env

    selftest()
    argv = sys.argv[1:]
    split = "test"
    if "--split" in argv:
        i = argv.index("--split"); split = argv[i + 1]; del argv[i:i + 2]
    out = Path(argv[0]) if argv else Path("results")
    out.mkdir(parents=True, exist_ok=True)
    df = chronological_split(load_dataset(argv[1]) if len(argv) > 1 else load_dataset())[split]
    env = make_env(df)

    rows = []
    for start in env.episode_starts():
        opts = {"start_step": int(start), "soc": 0.5}
        lp = run_oracle(env, cyclic=True, options=opts, solver="lp")
        dp = run_oracle(env, cyclic=True, options=opts, solver="dp")
        rows.append(dict(week=f"{lp.index[0]:%Y-%m-%d}", lp=(lp["cost"] + lp["degradation"]).sum(),
                         dp=(dp["cost"] + dp["degradation"]).sum(), bad_hours=int((lp["buy_price"] < lp["sell_price"]).sum())))
    res = pd.DataFrame(rows)
    res["dp_minus_lp"] = res["dp"] - res["lp"]
    pd.set_option("display.width", 200); pd.set_option("display.float_format", lambda x: f"{x:8.4f}")
    print(f"cyclic oracle on the {split} split, {len(res)} weeks, start SoC 0.5 (env replay matched both solvers):")
    print(res.to_string(index=False))
    print(f"\nmean LP {res.lp.mean():.4f}  mean DP {res.dp.mean():.4f}  DP - LP: mean {res.dp_minus_lp.mean():.4f}, max {res.dp_minus_lp.max():.4f} EUR/week")
    res.to_csv(out / f"oracle_{split}_weeks.csv", index=False)

    for start in env.episode_starts()[::13]:
        hist = run_oracle(env, cyclic=True, options={"start_step": int(start), "soc": 0.5})
        total = (hist["cost"] + hist["degradation"]).sum()
        fig = plot_episode(hist, f"oracle: {hist.index[0]:%d %b} – {hist.index[-1]:%d %b %Y}, total {total:.2f} EUR")
        fig.savefig(out / f"episode_oracle_{hist.index[0]:%Y%m%d}.png", dpi=130)
        plt.close(fig)
