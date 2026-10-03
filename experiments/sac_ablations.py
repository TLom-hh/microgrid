"""SAC ablations on the microgrid environment: one script, one --mode flag.

Everything not mentioned is as in sac.py: savings reward (env reward minus the idle-battery reward),
no terminal masking, S3 observations with forecasts, rule-based + random warmup. Training uses the train
split, the periodic eval is the mean plain return over all validation weeks (evaluation.py), and the
'best' checkpoint is the best validation score. Nothing here touches the test split.

modes   plain     sac.py as is
        mix       after warmup, 20 % of steps act with the solar-only heuristic + noise (SoC coverage)
        short     48 h training episodes, i.e. more random-SoC resets (eval episodes stay one week)
        residual  action a means battery_kw = (solar - load) + a * max_power_kw, so a = 0 is solar-only
        shaped    potential-based shaping r + gamma * Phi(s') - Phi(s), Phi = SoC * capacity * eff * P_REF
levers  --nstep N     N-step critic targets (per-sample discount; pending steps are flushed at the time limit)
        --clip-penalty C   charge C EUR per kWh of clipped request in the training reward (removes the free-discharge bias)
        --batch B --lr LR --steps S --warmup W
        --obs S2      snapshot + time + season, no forecasts
        --action-mode power|target_soc|residual   env action interface (microgrid_env.py); default power

This file is the record of the diagnosis experiments, not the baseline: the baseline is sac.py. The
outcome of every lever is in results/ablation_summary.txt.

usage:  python experiments/sac_ablations.py --mode plain --nstep 4 --seed 0 --out runs/ablations
Writes <out>/<tag>_log.csv (validation return per 5k steps), <out>/<tag>_best.pt and <out>/<tag>_final.pt.
Score checkpoints on the test split with evaluation.py (pass --action-mode if not power).
"""

import argparse, copy, itertools, sys, time
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))        # repo root: the modules below live there
from baselines import make_rule_based, solar_only  # noqa: E402
from data import load_dataset, chronological_split  # noqa: E402
from evaluation import validation_return, actor_policy  # noqa: E402
from microgrid_env import SNAPSHOT  # noqa: E402
from sac import Actor, QNet, REWARD_SCALE  # noqa: E402
from scenario import make_env  # noqa: E402
from training_env import TrainingReward  # noqa: E402

S2 = ("soc", "price", "solar", "load", "time", "season")
P_MIX, P_REF, GAMMA, TAU = 0.2, 0.20, 0.99, 0.005


class NStepBuffer:
    """Stores (obs, act, R, obs_n, disc): R = sum_k gamma^k r_{t+k} over up to n steps, disc = gamma^k."""

    def __init__(self, obs_dim, act_dim, size, n, gamma):
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.act = np.zeros((size, act_dim), np.float32)
        self.rew = np.zeros(size, np.float32)
        self.obs2 = np.zeros((size, obs_dim), np.float32)
        self.disc = np.zeros(size, np.float32)
        self.ptr, self.size, self.max_size, self.n, self.gamma = 0, 0, size, n, gamma
        self.pending = deque()

    def _emit(self, o2):
        R = sum(self.gamma ** k * r for k, (_, _, r) in enumerate(self.pending))
        disc = self.gamma ** len(self.pending)
        o, a, _ = self.pending.popleft()
        for buf, v in ((self.obs, o), (self.act, a), (self.rew, R), (self.obs2, o2), (self.disc, disc)):
            buf[self.ptr] = v
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def store(self, o, a, r, o2, end_of_episode):
        self.pending.append((o, a, r))
        if len(self.pending) == self.n:
            self._emit(o2)
        if end_of_episode:                      # time limit: bootstrap the tail from the last real state
            while self.pending:
                self._emit(o2)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {k: torch.as_tensor(getattr(self, k)[idx]) for k in ("obs", "act", "rew", "obs2", "disc")}


def update(batch, actor, q1, q2, q1_targ, q2_targ, q_opt, pi_opt, log_alpha, alpha_opt):
    """sac.update with a per-sample discount (n-step)."""
    alpha = log_alpha.exp().detach()
    o, a, r, o2, disc = (batch[k] for k in ("obs", "act", "rew", "obs2", "disc"))
    with torch.no_grad():
        a2, logp_a2 = actor(o2)
        q_t = torch.min(q1_targ(o2, a2), q2_targ(o2, a2))
        y = r + disc * (q_t - alpha * logp_a2)
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
            pt.mul_(1 - TAU).add_(TAU * p)
    return {"loss_q": round(loss_q.item(), 3), "loss_pi": round(loss_pi.item(), 3), "alpha": round(alpha.item(), 4)}


def to_env_action(env, a, residual):
    if not residual:
        return np.asarray(a, dtype=np.float64).reshape(-1)
    row = env.current_row()
    kw = (row["solar_kw"] - row["load_kw"]) + float(np.asarray(a).reshape(-1)[0]) * env.max_power_kw
    return np.array([np.clip(kw / env.max_power_kw, -1.0, 1.0)])


def evaluate(env, actor, residual=False):
    """Mean plain env return (EUR/week) of the deterministic policy on every week of env.df (the validation split)."""
    policy = actor_policy(actor)
    if residual:
        policy = (lambda base: (lambda e, o: to_env_action(e, base(e, o), True)))(policy)
    return validation_return(env, policy)


def load_data(path):
    """Chronological {'train', 'val', 'test'} slices of the dataset (see data.py)."""
    return chronological_split(load_dataset(path))


def main(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed); torch.set_num_threads(args.threads)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.mode}_{args.action_mode}_{args.obs}_n{args.nstep}_b{args.batch}_lr{args.lr:g}_s{args.seed}"
    splits = load_data(args.data)
    feats = S2 if args.obs == "S2" else SNAPSHOT
    env = TrainingReward(make_env(splits["train"], obs_features=feats, action_mode=args.action_mode,
                                  episode_hours=48 if args.mode == "short" else 24 * 7), clip_penalty=args.clip_penalty)
    eval_env = make_env(splits["val"], obs_features=feats, action_mode=args.action_mode)
    residual = args.mode == "residual"
    phi = (lambda soc: soc * env.capacity_kwh * env.efficiency * P_REF) if args.mode == "shaped" else (lambda soc: 0.0)

    obs_dim = env.observation_space.shape[0]
    actor = Actor(obs_dim, 1)
    q1, q2 = QNet(obs_dim, 1), QNet(obs_dim, 1)
    q1_targ, q2_targ = copy.deepcopy(q1), copy.deepcopy(q2)
    for p in itertools.chain(q1_targ.parameters(), q2_targ.parameters()):
        p.requires_grad = False
    q_opt = torch.optim.Adam(itertools.chain(q1.parameters(), q2.parameters()), lr=args.lr)
    pi_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    log_alpha = torch.zeros(1, requires_grad=True)
    alpha_opt = torch.optim.Adam([log_alpha], lr=args.lr)
    buffer = NStepBuffer(obs_dim, 1, 200_000, args.nstep, GAMMA)
    rule_pol = make_rule_based(env)

    obs, _ = env.reset(seed=args.seed)
    losses, best, log, t0 = {}, -np.inf, [], time.time()
    for step in range(args.steps):
        if step < args.warmup // 2:                              # heuristic + noise
            a = np.random.normal(0.0, 0.2, size=1) if residual else rule_pol(env) + np.random.normal(0.0, 0.1, size=1)
            a = np.clip(a, -1.0, 1.0)
        elif step < args.warmup:
            a = env.action_space.sample()
        elif args.mode == "mix" and np.random.rand() < P_MIX:
            a = np.clip(solar_only(env) + np.random.normal(0.0, 0.1, size=1), -1.0, 1.0)
        else:
            with torch.no_grad():
                a = actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0].numpy().reshape(-1)
        soc0 = env.soc
        obs2, r, term, trunc, _ = env.step(to_env_action(env, a, residual))     # r: saving minus clip penalty (training_env.py)
        r += GAMMA * phi(env.soc) - phi(soc0)                                   # optional potential-based shaping
        buffer.store(obs, a, r * REWARD_SCALE, obs2, term or trunc)
        obs = env.reset()[0] if (term or trunc) else obs2

        if step >= args.warmup:
            losses = update(buffer.sample(args.batch), actor, q1, q2, q1_targ, q2_targ, q_opt, pi_opt, log_alpha, alpha_opt)
        if step % args.eval_every == 0:
            ret = evaluate(eval_env, actor, residual)
            log.append((step, ret))
            print(f"[{tag}] step {step:>7d} eval return {ret:8.2f} {losses} {time.time() - t0:6.0f}s", flush=True)
            if ret > best:
                best = ret
                torch.save(actor.state_dict(), out / f"{tag}_best.pt")
    torch.save(actor.state_dict(), out / f"{tag}_final.pt")
    pd.DataFrame(log, columns=["step", "eval_return"]).to_csv(out / f"{tag}_log.csv", index=False)
    print(f"[{tag}] done, best eval return {best:.2f}, final {log[-1][1]:.2f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="plain", choices=["plain", "mix", "short", "residual", "shaped"])
    p.add_argument("--obs", default="S3", choices=["S3", "S2"])
    p.add_argument("--action-mode", default="power", choices=["power", "target_soc", "residual"],
                   help="env action interface (see microgrid_env.py); the wrapper-level --mode residual predates this")
    p.add_argument("--clip-penalty", type=float, default=0.0, help="EUR per kWh of clipped (infeasible) request, training reward only")
    p.add_argument("--nstep", type=int, default=1)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--warmup", type=int, default=10_000)
    p.add_argument("--eval-every", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--data", default="data/microgrid_hourly.csv")
    p.add_argument("--out", default="runs/ablations")
    p.add_argument("--tag", default=None)
    main(p.parse_args())
