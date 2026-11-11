"""Implementation check: Stable-Baselines3 SAC on the same task, reward and protocol as sac.py.

If SB3 lands where our SAC does, the gap to the heuristics is a property of model-free SAC on this task, not a bug
in sac.py. Everything except the implementation is held fixed:
    training reward   TrainingReward (saving vs idle battery + clip penalty), times REWARD_SCALE = 10 as in sac.py
    hyper-parameters  SB3 defaults equal sac.py's (256x256 MLPs, lr 3e-4, batch 256, tau 0.005, gamma 0.99,
                      auto entropy with target -act_dim); buffer 200k and 25k warmup steps set to match
    time limit        SB3 stores week ends as truncations, not terminals: no terminal masking, as in sac.py
    evaluation        validation return every 5k steps (best checkpoint), final table on the test year (continuous)
Differences that remain: SB3's warmup is uniform random actions (sac.py: half noisy rule-based, half random), and
SB3's internals (initialisation, the order of updates within a step).

usage: python experiments/sb3_sac.py [OUT_DIR] [--household base|ev|hp|ev_hp] [--action-mode power|target_soc|residual]
                                     [--clip-penalty 0.05] [--steps 100000] [--warmup 25000] [--seed 0] [--tag sb3] [--threads 1]
       several seeds: python experiments/seeds.py NAME --script experiments/sb3_sac.py -- --household ev_hp
Writes the same files as sac.py: <tag>_log.csv, <tag>_best.zip, <tag>_final.zip, <tag>_test_continuous.csv.
"""

import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from data import load_dataset, chronological_split, DATASET
from evaluation import validation_return, compare, report, policy_entry, oracle_entry, baseline_entries
from scenario import make_env
from training_env import TrainingReward

REWARD_SCALE = 10.0


class ScaleReward(gym.RewardWrapper):
    def __init__(self, env, scale):
        super().__init__(env)
        self.scale = scale

    def reward(self, r):
        return r * self.scale


def sb3_policy(model):
    return lambda env, obs: model.predict(obs, deterministic=True)[0]


class ValidationCallback(BaseCallback):
    """Validation return every 'every' env steps (step 0 included, like sac.py); keeps the best model."""
    def __init__(self, eval_env, every, best_path):
        super().__init__()
        self.eval_env, self.every, self.best_path = eval_env, every, best_path
        self.log, self.best = [], -np.inf

    def _evaluate(self, step):
        ret = validation_return(self.eval_env, sb3_policy(self.model))
        print(f"step {step:>7d} eval return {ret:8.2f}", flush=True)
        self.log.append((step, ret))
        if ret > self.best:
            self.best = ret
            self.model.save(self.best_path)

    def _on_training_start(self):
        self._evaluate(0)

    def _on_step(self):
        if self.num_timesteps % self.every == 0:
            self._evaluate(self.num_timesteps)
        return True


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", nargs="?", default="runs/sb3")
    p.add_argument("data", nargs="?", default=str(DATASET))
    p.add_argument("--household", default="base")
    p.add_argument("--action-mode", default="power", choices=["power", "target_soc", "residual"])
    p.add_argument("--clip-penalty", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--warmup", type=int, default=25_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="sb3")
    p.add_argument("--threads", type=int, default=0)
    args = p.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    splits = chronological_split(load_dataset(args.data))
    hh = args.household
    env = ScaleReward(TrainingReward(make_env(splits["train"], household=hh, action_mode=args.action_mode),
                                     clip_penalty=args.clip_penalty), REWARD_SCALE)
    eval_env = make_env(splits["val"], household=hh, action_mode=args.action_mode)

    model = SAC("MlpPolicy", env, buffer_size=200_000, learning_starts=args.warmup, seed=args.seed, device="cpu",
                policy_kwargs=dict(net_arch=[256, 256]))
    cb = ValidationCallback(eval_env, 5_000, out / f"{args.tag}_best")
    model.learn(total_timesteps=args.steps, callback=cb)
    model.save(out / f"{args.tag}_final")
    pd.DataFrame(cb.log, columns=["step", "validation_return"]).to_csv(out / f"{args.tag}_log.csv", index=False)

    test_env = make_env(splits["test"], household=hh)
    test_env_actor = test_env if args.action_mode == "power" else make_env(splits["test"], household=hh, action_mode=args.action_mode)
    best = SAC.load(out / f"{args.tag}_best", device="cpu")
    entries = baseline_entries(test_env, splits)
    entries["SB3 SAC (final)"] = policy_entry(test_env_actor, sb3_policy(model))
    entries["SB3 SAC (best on val)"] = policy_entry(test_env_actor, sb3_policy(best))
    entries["oracle (cyclic)"] = oracle_entry(test_env, cyclic=True)
    res = compare(entries, "continuous")
    report(res, list(entries))
    res.to_csv(out / f"{args.tag}_test_continuous.csv", index=False)
