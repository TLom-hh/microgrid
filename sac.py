"""Soft Actor-Critic on the microgrid environment.

Training reward (training_env.py): the battery's hourly saving relative to an idle battery, minus a small
penalty per infeasible kWh requested. Evaluation reports the plain environment cost, so numbers stay
comparable with the baselines and the oracle. The critic target has no terminal masking: the week's time
limit is not an end of the task, the battery keeps its value.

Data protocol (data.py / evaluation.py): training episodes come from the train split (2020-01 .. 2022-06),
the periodic eval and the 'best' checkpoint use the validation weeks (2022-07 .. 2022-12), and the final
table is one continuous run over the test year 2023 next to the baselines and the LP oracle.

usage: 'python sac.py [OUT_DIR] [DATA_CSV] [--action-mode power|target_soc|residual] [--clip-penalty 0.05] [--steps 100000] [--warmup 25000] [--seed 0] [--tag sac]'

Writes OUT_DIR/<tag>_best.pt (best validation return), <tag>_final.pt, <tag>_log.csv, <tag>_training.png, <tag>_test_continuous.csv (test-year table) and four example week plots.
"""

import copy, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from plotting import plot_episode
from data import load_dataset, chronological_split, DATASET
from evaluation import validation_return, actor_policy
from baselines import make_rule_based
from scenario import make_env
from training_env import TrainingReward
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

REWARD_SCALE = 10.0

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.rew = np.zeros(size, dtype=np.float32)
        self.obs2 = np.zeros((size, obs_dim), dtype=np.float32)
        self.done = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, o, a, r, o2, d):
        for buf, val in ((self.obs, o), (self.act, a), (self.rew, r), (self.obs2, o2), (self.done, d)):
            buf[self.ptr] = val
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {k: torch.as_tensor(getattr(self, k)[idx]) for k in ("obs", "act", "rew", "obs2", "done")}


class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1))

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(),
                                   nn.Linear(256, 256), nn.ReLU())
        self.mu_layer = nn.Linear(256, act_dim)
        self.log_std_layer = nn.Linear(256, act_dim)

    def forward(self, obs, deterministic=False):
        h = self.trunk(obs)
        mu = self.mu_layer(h)
        log_std = torch.clamp(self.log_std_layer(h), -20, 2)
        dist = torch.distributions.Normal(mu, torch.exp(log_std))
        u = mu if deterministic else dist.rsample()
        a = torch.tanh(u)
        logp = dist.log_prob(u).sum(-1)
        logp -= (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1)
        return a, logp

def update(batch, actor, q1, q2, q1_targ, q2_targ, q_opt, pi_opt, log_alpha, alpha_opt, gamma=0.99, tau=0.005):
    alpha = log_alpha.exp().detach()
    o, a, r, o2, d = (batch[k] for k in ("obs", "act", "rew", "obs2", "done"))

    with torch.no_grad():
        a2, logp_a2 = actor(o2)
        q_t = torch.min(q1_targ(o2, a2), q2_targ(o2, a2))
        y = r + gamma * (q_t - alpha * logp_a2)
    loss_q = F.mse_loss(q1(o, a), y) + F.mse_loss(q2(o, a), y)
    q_opt.zero_grad(); loss_q.backward(); q_opt.step()

    a_new, logp = actor(o)
    loss_pi = (alpha * logp - torch.min(q1(o, a_new), q2(o, a_new))).mean()
    pi_opt.zero_grad(); loss_pi.backward(); pi_opt.step()
    loss_alpha = -(log_alpha * (logp.detach() - 1.0)).mean()
    alpha_opt.zero_grad(); loss_alpha.backward(); alpha_opt.step()

    with torch.no_grad():
        for p, pt in zip(itertools.chain(q1.parameters(), q2.parameters()),
                        itertools.chain(q1_targ.parameters(), q2_targ.parameters())):
            pt.mul_(1 - tau).add_(tau * p)

    return {"loss_q": loss_q.item(), "loss_pi": loss_pi.item(), "alpha": alpha.item()}


def evaluate(env, actor):
    return validation_return(env, actor_policy(actor))

def train(env, eval_env, total_steps=100_000, warmup=5_000, batch_size=256, eval_every=5_000, save_path=None, seed=0):
    obs_dim = env.observation_space.shape[0]
    actor = Actor(obs_dim, 1)
    q1, q2 = QNet(obs_dim, 1), QNet(obs_dim, 1)
    q1_targ, q2_targ = copy.deepcopy(q1), copy.deepcopy(q2)
    for p in itertools.chain(q1_targ.parameters(), q2_targ.parameters()):
        p.requires_grad = False
    q_opt = torch.optim.Adam(itertools.chain(q1.parameters(), q2.parameters()), lr=3e-4)
    pi_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    log_alpha = torch.zeros(1, requires_grad=True)
    alpha_opt = torch.optim.Adam([log_alpha], lr=3e-4)
    buffer = ReplayBuffer(obs_dim, 1, size=200_000)

    log, losses = [], {}
    obs, _ = env.reset(seed=seed)
    rule_pol = make_rule_based(env)
    best = -np.inf;
    for step in range(total_steps):
        if step < warmup // 2:
            a = np.clip(rule_pol(env) + np.random.normal(0.0, 0.1, size=1), -1.0, 1.0)
        elif step < warmup:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a_t, _ = actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            a = a_t.numpy().reshape(-1)
        obs2, r, term, trunc, _ = env.step(a)                    # r: saving minus clip penalty (training_env.py)
        buffer.store(obs, a, r * REWARD_SCALE, obs2, 0.0)        # done is always 0: continuing task, no terminal masking
        obs = env.reset()[0] if (term or trunc) else obs2

        if step >= warmup:
            losses = update(buffer.sample(batch_size), actor, q1, q2, q1_targ, q2_targ, q_opt, pi_opt, log_alpha, alpha_opt)
        if step % eval_every == 0:
            ret = evaluate(eval_env, actor)
            print(f"step {step:>7d} eval return {ret:8.2f} {losses}")
            log.append((step, ret))
            if ret > best:
                best = ret
                if save_path is not None:
                    torch.save(actor.state_dict(), save_path)
    return actor, log

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", nargs="?", default="runs/sac")
    p.add_argument("data", nargs="?", default=str(DATASET))
    p.add_argument("--action-mode", default="power", choices=["power", "target_soc", "residual"])
    p.add_argument("--clip-penalty", type=float, default=0.05, help="EUR per infeasible kWh requested, training reward only")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--warmup", type=int, default=25_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="sac")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    splits = chronological_split(load_dataset(args.data))
    env = TrainingReward(make_env(splits["train"], action_mode=args.action_mode), clip_penalty=args.clip_penalty)
    eval_env = make_env(splits["val"], action_mode=args.action_mode)       # checkpoint selection: validation weeks

    best_path = out / f"{args.tag}_best.pt"
    actor, log = train(env, eval_env, total_steps=args.steps, warmup=args.warmup, save_path=best_path, seed=args.seed)
    torch.save(actor.state_dict(), out / f"{args.tag}_final.pt")
    pd.DataFrame(log, columns=["step", "validation_return"]).to_csv(out / f"{args.tag}_log.csv", index=False)

    # training curve
    steps, rets = zip(*log)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, rets)
    ax.set_xlabel("env steps"); ax.set_ylabel("validation return (EUR/week)")
    fig.savefig(out / f"{args.tag}_training.png", dpi=130)
    plt.close(fig)

    # final comparison on the test split: one continuous run over 2023, SoC carried across weeks (evaluation.py)
    from evaluation import compare, report, policy_entry, oracle_entry, baseline_entries, run_episode
    test_env = make_env(splits["test"])
    test_env_actor = test_env if args.action_mode == "power" else make_env(splits["test"], action_mode=args.action_mode)
    best = Actor(env.observation_space.shape[0], 1)
    best.load_state_dict(torch.load(best_path))
    entries = baseline_entries(test_env, splits)
    entries["SAC (final)"] = policy_entry(test_env_actor, actor_policy(actor))
    entries["SAC (best on val)"] = policy_entry(test_env_actor, actor_policy(best))
    entries["oracle (cyclic)"] = oracle_entry(test_env, cyclic=True)
    res = compare(entries, "continuous")
    report(res, list(entries))
    res.to_csv(out / f"{args.tag}_test_continuous.csv", index=False)

    for start in test_env_actor.episode_starts()[::13]:                          # four example weeks, one per season
        hist = run_episode(test_env_actor, actor_policy(best), start)
        total = (hist["cost"] + hist["degradation"]).sum()
        fig = plot_episode(hist, f"SAC: {hist.index[0]:%d %b} – {hist.index[-1]:%d %b %Y}, total {total:.2f} EUR")
        fig.savefig(out / f"{args.tag}_episode_{hist.index[0]:%Y%m%d}.png", dpi=130)
        plt.close(fig)
        print(f"SAC week {hist.index[0]:%Y-%m-%d} cost {total:7.2f} EUR   clipped steps {int(hist['clipped'].sum()):3d}/{len(hist)}")
