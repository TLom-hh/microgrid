"""Vanilla policy gradient (REINFORCE with a GAE baseline, SpinningUp style) on the microgrid environment.
Policy: Gaussian with a tanh-bounded mean (see GaussianPolicy for why the bound is needed).

Same task setup as sac.py: savings reward (env reward minus the idle-battery reward), continuing task
(the value function is bootstrapped at the one-week time limit, no terminal masking), S3 observations,
training on the train split, deterministic evaluation on the validation weeks in plain EUR (evaluation.py).
On-policy: each epoch collects fresh steps, takes ONE policy-gradient step (no PPO clipping, no importance
ratios) and refits the value baseline.

This file is a diagnosis experiment, not a baseline: VPG does not work on this task, but its collapse made
the clipping asymmetry visible (see GaussianPolicy and training_env.py).

usage: python experiments/vpg.py --seed 0 --epochs 750 --steps-per-epoch 4096 --out runs/vpg [--data data/microgrid_hourly.csv]
       add --residual for the residual action (a = 0 is the solar-only heuristic, see sac_ablations.py)
Writes <out>/<tag>_log.csv, <out>/<tag>_best.pt, <out>/<tag>_final.pt. Score on the test split with
evaluation.py using 'name=path:vpg'.
"""

import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))        # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))            # this folder (sac_ablations)
from microgrid_env import SNAPSHOT  # noqa: E402
from sac import REWARD_SCALE  # noqa: E402
from sac_ablations import evaluate, load_data, to_env_action  # noqa: E402
from scenario import make_env  # noqa: E402
from training_env import TrainingReward  # noqa: E402


class GaussianPolicy(nn.Module):
    """Gaussian in action space with a tanh-bounded mean in [-1, 1] and a state-independent log-std.

    Samples outside [-1, 1] are clipped (the env clips too); log-probs are those of the unclipped sample,
    the usual 'clipped Gaussian' policy. The tanh keeps the mean inside the action range, but it does not
    prevent the collapse seen on this task: the policy still drifts to a bound (full discharge), because at an
    empty battery discharge requests are clipped for free while charge requests cost money now. The saved
    policies have small pre-tanh outputs, so this is not tanh saturation; --clip-penalty removes the bias.
    forward() has sac.Actor's (action, logp) signature so the evaluation.py harness can run it.
    """

    def __init__(self, obs_dim, act_dim, log_std_init=-0.7):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.mu_layer = nn.Linear(256, act_dim)
        with torch.no_grad():                       # start near a = 0 (idle battery, or the heuristic in residual mode)
            self.mu_layer.weight.mul_(0.01)
            self.mu_layer.bias.zero_()
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

    def dist(self, obs):
        mu = torch.tanh(self.mu_layer(self.trunk(obs)))
        return torch.distributions.Normal(mu, self.log_std.clamp(-5.0, 0.5).exp())

    def forward(self, obs, deterministic=False):
        d = self.dist(obs)
        u = d.mean if deterministic else d.sample()
        return u.clamp(-1.0, 1.0), d.log_prob(u).sum(-1)

    def act(self, obs):
        with torch.no_grad():
            u = self.dist(obs).sample()
        return u.clamp(-1.0, 1.0), u

    def log_prob(self, obs, u):
        return self.dist(obs).log_prob(u).sum(-1)

    def entropy(self):
        return (0.5 + 0.5 * np.log(2 * np.pi) + self.log_std.clamp(-5.0, 0.5)).sum()


class ValueNet(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


def gae(rews, vals, last_val, gamma, lam):
    """Generalized advantage estimation for one path; last_val bootstraps the final step (no terminal masking)."""
    vals_ext = np.append(vals, last_val)
    deltas = rews + gamma * vals_ext[1:] - vals_ext[:-1]
    adv = np.zeros_like(rews)
    running = 0.0
    for t in reversed(range(len(rews))):
        running = deltas[t] + gamma * lam * running
        adv[t] = running
    return adv, adv + vals                                  # advantages, TD(lambda) value targets


def main(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed); torch.set_num_threads(args.threads)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"vpg{'_residual' if args.residual else ''}{'_shaped' if args.shaping_price else ''}{'_clip' if args.clip_penalty else ''}_s{args.seed}"
    splits = load_data(args.data)                         # chronological train / val / test (data.py)
    env = TrainingReward(make_env(splits["train"], obs_features=SNAPSHOT), clip_penalty=args.clip_penalty)
    eval_env = make_env(splits["val"], obs_features=SNAPSHOT)
    # optional potential-based shaping on the SoC (optimal policy unchanged): Phi = SoC * capacity * eff * price
    phi = lambda soc: soc * env.capacity_kwh * env.efficiency * args.shaping_price
    obs_dim = env.observation_space.shape[0]
    pi, vf = GaussianPolicy(obs_dim, 1, args.log_std_init), ValueNet(obs_dim)
    pi_opt, vf_opt = torch.optim.Adam(pi.parameters(), lr=args.pi_lr), torch.optim.Adam(vf.parameters(), lr=args.vf_lr)

    obs, _ = env.reset(seed=args.seed)
    best, log, t0, total_steps = -np.inf, [], time.time(), 0
    ep_rets, ep_ret = [], 0.0
    for epoch in range(args.epochs):
        O, U, R, V = [], [], [], []
        path_start, advs, rets = 0, [], []
        for step in range(args.steps_per_epoch):
            o_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            a, u = pi.act(o_t)
            with torch.no_grad():
                v = vf(o_t).item()
            soc0 = env.soc
            obs2, r, term, trunc, info = env.step(to_env_action(env, a.numpy(), args.residual))   # saving minus clip penalty
            r = (r + args.gamma * phi(env.soc) - phi(soc0)) * REWARD_SCALE                        # optional shaping
            O.append(obs); U.append(u.numpy().reshape(-1)); R.append(r); V.append(v)
            ep_ret += info["saving"]
            obs = obs2
            end_of_path = term or trunc or step == args.steps_per_epoch - 1
            if end_of_path:
                with torch.no_grad():                       # continuing task: always bootstrap from the real next state
                    last_val = vf(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).item()
                adv, ret = gae(np.array(R[path_start:]), np.array(V[path_start:]), last_val, args.gamma, args.lam)
                advs.append(adv); rets.append(ret); path_start = len(R)
                if term or trunc:
                    ep_rets.append(ep_ret); ep_ret = 0.0
                    obs, _ = env.reset()
        total_steps += args.steps_per_epoch

        O_t = torch.as_tensor(np.array(O), dtype=torch.float32)
        U_t = torch.as_tensor(np.array(U), dtype=torch.float32)
        adv_t = torch.as_tensor(np.concatenate(advs), dtype=torch.float32)
        ret_t = torch.as_tensor(np.concatenate(rets), dtype=torch.float32)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        logp = pi.log_prob(O_t, U_t)                        # one policy-gradient step
        loss_pi = -(logp * adv_t).mean() - args.ent_coef * pi.entropy()
        pi_opt.zero_grad(); loss_pi.backward(); pi_opt.step()
        for _ in range(args.vf_iters):                      # refit the baseline
            loss_v = F.mse_loss(vf(O_t), ret_t)
            vf_opt.zero_grad(); loss_v.backward(); vf_opt.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            ev = evaluate(eval_env, pi, args.residual)
            log.append((epoch, total_steps, ev))
            recent = np.mean(ep_rets[-24:]) if ep_rets else float("nan")
            print(f"[{tag}] epoch {epoch:>4d} steps {total_steps:>8d} eval return {ev:7.2f}  "
                  f"train saving/week {recent:6.2f}  std {pi.log_std.exp().item():.3f}  loss_v {loss_v.item():7.3f}  {time.time() - t0:6.0f}s", flush=True)
            if ev > best:
                best = ev; torch.save(pi.state_dict(), out / f"{tag}_best.pt")
    torch.save(pi.state_dict(), out / f"{tag}_final.pt")
    pd.DataFrame(log, columns=["epoch", "steps", "eval_return"]).to_csv(out / f"{tag}_log.csv", index=False)
    print(f"[{tag}] done, best eval return {best:.2f}, final {log[-1][2]:.2f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=750)
    p.add_argument("--steps-per-epoch", type=int, default=4096)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.97)
    p.add_argument("--pi-lr", type=float, default=3e-4)
    p.add_argument("--vf-lr", type=float, default=1e-3)
    p.add_argument("--vf-iters", type=int, default=80)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--log-std-init", type=float, default=-0.7)
    p.add_argument("--shaping-price", type=float, default=0.0,
                   help="EUR/kWh reference value of stored energy for potential-based shaping (0 = off)")
    p.add_argument("--clip-penalty", type=float, default=0.0,
                   help="EUR per kWh of clipped (infeasible) request, training reward only (0 = off)")
    p.add_argument("--residual", action="store_true",
                   help="action is a deviation from the solar-only heuristic: battery_kw = (solar - load) + a * max_power_kw")
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--data", default="data/microgrid_hourly.csv")
    p.add_argument("--out", default="runs/vpg")
    p.add_argument("--tag", default=None)
    main(p.parse_args())
