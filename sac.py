"""Running Soft Actor-Critic on the microgrid environment"""

import copy
import sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from plotting import plot_episode
from smarthome import SmartMicrogridEnv
from smarthome_agents import make_rule_based
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
        y = r + gamma * (1 - d) * (q_t - alpha * logp_a2)
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


def evaluate(env, actor, seed=None, n_episodes=5):
    returns = []
    for i in range(n_episodes):
        if seed is None: 
            obs, _ = env.reset(seed=1000 + i)
        else:
            obs, _ = env.reset(seed=seed)
        done, ep_ret = False, 0.0
        while not done:
            with torch.no_grad():
                a, _ = actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0),
                            deterministic=True)
            obs, r, term, trunc, _ = env.step(a.numpy().reshape(-1))
            ep_ret += r
            done = term or trunc
        returns.append(ep_ret)
    return float(np.mean(returns))


def train(env, eval_env, total_steps=100_000, warmup=5_000, batch_size=256, eval_every=5_000):
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
    obs, _ = env.reset(seed=0)
    rule_pol = make_rule_based(env)
    best = 0.0;
    for step in range(total_steps):
        if step < warmup // 2:
            a = np.clip(rule_pol(env) +  + np.random.normal(0.0, 0.1, size=1), -1.0, 1.0)
        elif step < warmup:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a_t, _ = actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
            a = a_t.numpy().reshape(-1)
        obs2, rew, term, trunc, _ = env.step(a)
        buffer.store(obs, a, rew * REWARD_SCALE, obs2, float(term))
        obs = env.reset()[0] if (term or trunc) else obs2

        if step >= warmup:
            losses = update(buffer.sample(batch_size), actor, q1, q2, q1_targ, q2_targ, q_opt, pi_opt, log_alpha, alpha_opt)
        if step % eval_every == 0:
            ret = evaluate(eval_env, actor)
            print(f"step {step:>7d} eval return {ret:8.2f} {losses}")
            log.append((step, ret))
            if ret > best: best = ret; torch.save(actor.state_dict(), out / "sac_actor.pt")
    return actor, log

if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out.mkdir(parents=True, exist_ok=True)
    df = None
    if len(sys.argv) > 2:
        import pandas as pd
        df = pd.read_csv(sys.argv[2], index_col="time")
        df.index = pd.to_datetime(df.index, utc=True).tz_convert("Europe/Berlin")
    obs = ("soc", "price", "solar", "load", "time", "season")
    env, eval_env = SmartMicrogridEnv(df=df), SmartMicrogridEnv(df=df)

    actor, log = train(env, eval_env, total_steps=200_000, warmup=25_000)
    torch.save(actor.state_dict(), out / "sac_actor_finish.pt")

    # training curve
    steps, rets = zip(*log)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, rets)
    ax.set_xlabel("env steps"); ax.set_ylabel("eval return (EUR/week)")
    fig.savefig(out / "sac_training.png", dpi=130)
    plt.close(fig)

    actor = Actor(env.observation_space.shape[0], 1)
    actor.load_state_dict(torch.load("plots/sac_actor.pt"))
    # one deterministic episode
    seed = 1004
    evaluate(eval_env, actor, seed, n_episodes=1)
    hist = eval_env.history_frame()
    total = (hist["cost"] + hist["degradation"]).sum()
    fig = plot_episode(hist, f"SAC: {hist.index[0]:%d %b} – {hist.index[-1]:%d %b %Y}, total {total:.2f} EUR")
    fig.savefig(out / "episode_sac.png", dpi=130)
    plt.close(fig)
    print(f"SAC cost {total:7.2f} EUR   clipped steps {int(hist['clipped'].sum()):3d}/{len(hist)}")
