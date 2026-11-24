"""Does the DreamerV3 world model predict the rewards that matter? Its reward head against the actual training reward.

The trained player acts greedily on a continuous run (training reward: TrainingReward with the run's own settings,
clip penalty and EV shaping, read from its config.yaml) and at every
step the reward head is read twice:
    posterior   from (z_t, h_t), z_t inferred from the real observation: the head as trained
    prior       from (z_t ~ prior(h_t), h_t), without seeing obs t: what the actor sees one step into imagination
sheeprl stores each reward with the observation that follows it, so the head at latent t predicts r_{t-1}.
Steps are grouped by what happened: departures with a public top-up (lumped penalty), other departures, the rest.

usage: python experiments/reward_check.py [--run runs/dreamer_s0] [--ckpt PATH] [--split test|val|train]
Writes RUN/diagnosis/reward_check_<split>.csv (one row per step) and reward_check_<split>.png.
"""

import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from data import load_dataset, chronological_split
from scenario import make_env
from training_env import TrainingReward
import evaluation as ev
from dreamer import load_agent, MLP_KEY
from plotting import C, _style

GROUPS = {"departure, public top-up": C["import"], "departure, no top-up": C["ev"], "other": C["muted"]}


def reward_mean(world_model, latent):
    from sheeprl.utils.distribution import TwoHotEncodingDistribution
    return TwoHotEncodingDistribution(world_model.reward_model(latent), dims=1).mean.item()


def rollout(world_model, player, env, seed=0):
    base = env.unwrapped
    o = ev.continuous_options(base)
    obs, _ = env.reset(seed=ev.EVAL_SEED + o["start_step"], options=o)
    torch.manual_seed(seed)
    player.init_states()
    rows, done = [], False
    while not done:
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).view(1, 1, -1)
        actions = player.get_actions({MLP_KEY: x}, greedy=True)
        h, z = player.recurrent_state, player.stochastic_state
        _, prior = world_model.rssm._transition(h)
        rows.append((reward_mean(world_model, torch.cat((z, h), -1)),
                     reward_mean(world_model, torch.cat((prior.reshape(*h.shape[:-1], -1), h), -1))))
        obs, r, term, trunc, _ = env.step(torch.cat(actions, -1).reshape(-1).numpy())
        rows[-1] += (r,)
        done = term or trunc
    pred = np.array(rows)
    hist = base.history_frame()
    # head at latent t+1 predicts the reward of step t
    df = pd.DataFrame({"actual": pred[:-1, 2], "pred_posterior": pred[1:, 0], "pred_prior": pred[1:, 1]}, index=hist.index[:-1])
    deps = base.df["ev_departs"].values[o["start_step"] + 1: o["start_step"] + len(df) + 1].astype(bool)
    short = hist["ev_shortfall_kwh"].values[:-1]
    df["shortfall_kwh"] = short
    df["group"] = np.where(deps & (short > 0.01), "departure, public top-up", np.where(deps, "departure, no top-up", "other"))
    return df


def summary(df):
    g = df.groupby("group")
    out = pd.DataFrame({"steps": g.size(), "actual": g["actual"].mean(), "pred_posterior": g["pred_posterior"].mean(),
                        "pred_prior": g["pred_prior"].mean(),
                        "mae_posterior": g.apply(lambda d: (d["pred_posterior"] - d["actual"]).abs().mean()),
                        "mae_prior": g.apply(lambda d: (d["pred_prior"] - d["actual"]).abs().mean())})
    return out.reindex([k for k in GROUPS if k in out.index])


def figure(df, path, title):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.patch.set_facecolor(C["surface"])
    lo, hi = df[["actual", "pred_posterior", "pred_prior"]].min().min(), df[["actual", "pred_posterior", "pred_prior"]].max().max()
    for ax, col, name in ((axs[0], "pred_posterior", "posterior (obs seen)"), (axs[1], "pred_prior", "prior (one step imagined)")):
        for grp in ("other", "departure, no top-up", "departure, public top-up"):
            d = df[df["group"] == grp]
            ax.scatter(d["actual"], d[col], s=6 if grp == "other" else 14, alpha=0.35 if grp == "other" else 0.8,
                       color=GROUPS[grp], label=f"{grp} ({len(d)})", lw=0)
        ax.plot([lo, hi], [lo, hi], color=C["ink2"], lw=1, ls=":")
        ax.set_xlabel("actual training reward", fontsize=9, color=C["ink2"])
        _style(ax, f"predicted, {name}")
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    d = df[df["group"] == "departure, public top-up"]
    ax = axs[2]
    ax.scatter(d["shortfall_kwh"], -d["actual"], s=14, color=C["import"], label="actual", lw=0)
    ax.scatter(d["shortfall_kwh"], -d["pred_prior"], s=14, color=C["price"], label="predicted (prior)", lw=0)
    ax.set_xlabel("public top-up at departure (kWh)", fontsize=9, color=C["ink2"])
    _style(ax, "penalty (−reward)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle(title, color=C["ink"], fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="runs/dreamer_s0")
    p.add_argument("--ckpt", default=None, help="default: the run's best checkpoint on validation")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--household", default="ev_hp")
    args = p.parse_args()

    run = Path(args.run)
    ckpt = args.ckpt or (run / "dreamer_best_checkpoint.txt").read_text().strip()
    out = run / "diagnosis"; out.mkdir(parents=True, exist_ok=True)
    world_model, player = load_agent(ckpt)
    splits = chronological_split(load_dataset())
    from omegaconf import OmegaConf
    from sheeprl.utils import fs
    wrapper = OmegaConf.load(fs.run_config(ckpt)).env.wrapper          # the training reward the run was trained on
    shaping = dict(ev_shaping=bool(wrapper.get("ev_shaping", False)), gamma=float(wrapper.get("gamma", 0.99)))
    env = TrainingReward(make_env(splits[args.split], household=args.household, action_mode=wrapper.get("action_mode", "power")),
                         clip_penalty=float(wrapper.clip_penalty), **shaping)
    print("training reward:", dict(clip_penalty=float(wrapper.clip_penalty), **shaping))
    df = rollout(world_model, player, env)
    df.to_csv(out / f"reward_check_{args.split}.csv")

    pd.set_option("display.width", 160)
    print(f"{Path(ckpt).name} on {args.split}: {len(df)} steps")
    print(summary(df).round(3).to_string())
    total = df[["actual", "pred_posterior", "pred_prior"]].sum()
    print("sum over the run:", total.round(1).to_dict())
    d = df[df["group"] == "departure, public top-up"]
    if len(d) > 1:
        slope = np.polyfit(-d["actual"], -d["pred_prior"], 1)[0]
        print(f"top-up penalty: predicted/actual slope {slope:.2f} (prior), corr {np.corrcoef(d['actual'], d['pred_prior'])[0, 1]:.2f}")
    figure(df, out / f"reward_check_{args.split}.png", f"DreamerV3 reward head vs actual reward ({args.split}, {Path(ckpt).stem})")
