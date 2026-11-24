"""DreamerV3 (sheeprl) on the microgrid environment: WP6.

Training runs through sheeprl's own command line with our configs (configs/sheeprl: env/microgrid.yaml,
exp/dreamer_microgrid.yaml), so sheeprl stays unmodified. What is held equal to sac.py:
    training env      train split, TrainingReward (saving vs idle battery + clip penalty); no reward scaling, since
                      DreamerV3 learns symlog rewards and normalises returns itself
    continuing task   the env truncates its weeks and never terminates; sheeprl's continue target is
                      1 - terminated, so a week end is not a terminal (as in sac.py's critic)
    selection         a checkpoint every 5k env steps; each is scored on the validation weeks (evaluation.py) and
                      the best one plus the final one go to the test year (continuous run)
The policy is recurrent: dreamer_policy keeps the RSSM state across the steps of a run and resets it when an
episode starts (env.current_step == env.start_step), so it plugs into evaluation.run_episode unchanged. The posterior
is sampled even when acting greedily; the torch seed is fixed at every episode start, so scoring is deterministic.

usage: python dreamer.py [OUT_DIR] [--household ev_hp] [--action-mode power|target_soc|residual] [--control full|ev] [--ev-shaping]
                         [--houses fixed|gen] [--context observed|hidden]
                         [--steps 100000] [--seed 0] [--tag dreamer]
                         [--threads 1] [--accelerator cpu|mps] [--size XS|S|M] [--eval-only] [-- HYDRA OVERRIDES]
--ev-shaping: potential-based shaping of the EV deadline (training_env.py) with DreamerV3's own discount (algo.gamma).
--houses gen: generalization setting (houses.py), as in sac.py: random training houses, validation houses, test on new
       houses (<tag>_test_continuous.csv with a 'house' column) and on out-of-range houses (<tag>_test_ood_continuous.csv).
       several seeds: python experiments/seeds.py NAME --script dreamer.py -- --household ev_hp
Writes OUT_DIR/<tag>_log.csv (step, validation_return), <tag>_best_checkpoint.txt, <tag>_test_continuous.csv;
sheeprl's run (checkpoints, config.yaml, tensorboard) lands in OUT_DIR/sheeprl/<tag>/version_*/.
"""

from __future__ import annotations
import argparse, functools, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MLP_KEY = "state"


# ---- the environment factory sheeprl instantiates (configs/sheeprl/env/microgrid.yaml) ---------------------------

@functools.lru_cache(maxsize=1)
def _splits():
    from data import load_dataset, chronological_split
    return chronological_split(load_dataset(ROOT / "data" / "microgrid_hourly.csv"))

def make_microgrid(household="base", split="train", action_mode="power", clip_penalty=0.05, ev_shaping=False,
                   gamma=0.99, control="full", houses="fixed", context="observed", **_):
    from scenario import make_env
    from training_env import TrainingReward
    if houses == "gen":                                        # one of the split's house set per episode (houses.py)
        import houses as H
        env = H.make_multi_env(_splits()[split], split, action_mode=action_mode, control=control, context=context,
                               record_history=False)
    else:
        env = make_env(_splits()[split], household=household, action_mode=action_mode, control=control, record_history=False)
    return TrainingReward(env, clip_penalty=clip_penalty, ev_shaping=ev_shaping, gamma=gamma)


# ---- a trained checkpoint as a policy for evaluation.py --------------------------------------------------------------

def load_player(ckpt_path):
    """The DreamerV3 player (encoder, RSSM, actor) of a sheeprl checkpoint, on the CPU, for one environment."""
    return load_agent(ckpt_path)[1]


def load_agent(ckpt_path):
    """(world_model, player) of a sheeprl checkpoint, on the CPU; the world model has the reward and continue heads."""
    import gymnasium as gym
    import torch
    from lightning import Fabric
    from sheeprl.algos.dreamer_v3.agent import build_agent
    from sheeprl.utils import fs
    from sheeprl.utils.env import make_env as sheeprl_make_env
    from sheeprl.utils.utils import dotdict

    from omegaconf import OmegaConf
    cfg = dotdict(OmegaConf.to_container(fs.load_yaml(fs.run_config(ckpt_path)), resolve=True))
    cfg.env.num_envs = 1
    cfg.env.capture_video = False
    fabric = Fabric(accelerator="cpu", devices=1, precision=cfg.fabric.precision)
    state = fs.load_checkpoint(fabric, ckpt_path, weights_only=False)
    env = sheeprl_make_env(cfg, cfg.seed, 0)()                 # only for the (wrapped) spaces
    action_space = env.action_space
    assert isinstance(action_space, gym.spaces.Box)
    world_model, _, _, _, player = build_agent(fabric, tuple(action_space.shape), True, cfg, env.observation_space,
                                               state["world_model"], state["actor"])
    env.close()
    player.num_envs = 1
    player.eval()
    world_model.eval()
    torch.set_grad_enabled(False)
    return world_model, player


def dreamer_policy(player, record=None, seed=0):
    """policy(env, obs) for evaluation.run_episode. record: optional list that receives, per step, the latent
    state the actor saw (recurrent h and stochastic z, flattened), for the probing analysis (WP7)."""
    import torch

    def policy(env, obs):
        if env.current_step == env.start_step:                 # a new run: fresh RSSM state, fixed sampling seed
            torch.manual_seed(seed)
            player.init_states()
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=player.device).view(1, 1, -1)
        actions = player.get_actions({MLP_KEY: x}, greedy=True)
        if record is not None:
            record.append(torch.cat((player.recurrent_state, player.stochastic_state), -1).reshape(-1).cpu().numpy())
        return torch.cat(actions, -1).reshape(-1).cpu().numpy()
    return policy


def checkpoints(run_dir):
    """(env step, path) of every checkpoint of the run, oldest first."""
    paths = sorted(Path(run_dir).glob("version_*/checkpoint/ckpt_*_0.ckpt"))
    return sorted((int(p.name.split("_")[1]), p) for p in paths)


# ---- command line: train through sheeprl, then score -----------------------------------------------------------

def run_sheeprl(hydra_args):
    """sheeprl's command line, with one patch: the replay buffer keeps rewards and flags as float64 (gymnasium's
    dtype) and moves them to the device as they are, which the Apple GPU (MPS) cannot hold; they go as float32."""
    import torch
    import sheeprl.data.buffers as buffers
    original = buffers.get_tensor

    def get_tensor(array, dtype=None, clone=False, device="cpu", from_numpy=False):
        if dtype is None and str(device).startswith("mps") and getattr(array, "dtype", None) == np.float64:
            dtype = torch.float32
        return original(array, dtype=dtype, clone=clone, device=device, from_numpy=from_numpy)

    buffers.get_tensor = get_tensor
    sys.argv = [sys.argv[0], *hydra_args]
    from sheeprl.cli import run
    run()


def train(out, args, extra):
    cmd = [sys.executable, str(ROOT / "dreamer.py"), "--sheeprl", "--config-dir", str(ROOT / "configs" / "sheeprl"), "exp=dreamer_microgrid",
           f"algo=dreamer_v3_{args.size}", f"env.wrapper.household={args.household}", f"env.wrapper.action_mode={args.action_mode}", f"env.wrapper.control={args.control}", f"env.wrapper.ev_shaping={args.ev_shaping}",
           f"env.wrapper.houses={args.houses}", f"env.wrapper.context={args.context}", f"algo.total_steps={args.steps}",
           f"seed={args.seed}", f"num_threads={max(args.threads, 1)}", f"fabric.accelerator={args.accelerator}",
           f"log_root={out}", "root_dir=sheeprl", f"run_name={args.tag}", f"hydra.run.dir={out}/sheeprl/{args.tag}/hydra",
           *extra]
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    print("training:", " ".join(cmd[1:]), flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    print(f"training took {(time.time() - t0) / 60:.1f} min", flush=True)


def score(out, args):
    from evaluation import validation_return, compare, report, policy_entry, oracle_entry, baseline_entries
    from scenario import make_env

    splits = _splits()
    ckpts = checkpoints(out / "sheeprl" / args.tag)
    if not ckpts:
        raise SystemExit(f"no checkpoints under {out / 'sheeprl' / args.tag}")
    if args.houses == "gen":
        import houses as H
        val_envs = H.make_envs(splits["val"], "val", action_mode=args.action_mode, control=args.control, context=args.context)
        val_fn = lambda pol: H.validation_return(val_envs, pol)
    else:
        val_env = make_env(splits["val"], household=args.household, action_mode=args.action_mode, control=args.control)
        val_fn = lambda pol: validation_return(val_env, pol)
    log, best = [], None
    for step, path in ckpts:
        t0 = time.time()
        ret = val_fn(dreamer_policy(load_player(path)))
        print(f"step {step:>7d} eval return {ret:8.2f}   ({time.time() - t0:.0f} s)", flush=True)
        log.append((step, ret))
        if best is None or ret > best[0]:
            best = (ret, step, path)
    pd.DataFrame(log, columns=["step", "validation_return"]).to_csv(out / f"{args.tag}_log.csv", index=False)
    (out / f"{args.tag}_best_checkpoint.txt").write_text(f"{best[2]}\n")
    print(f"best on validation: step {best[1]} ({best[0]:.2f})")

    if args.houses == "gen":
        pols = {"Dreamer (final)": dreamer_policy(load_player(ckpts[-1][1])), "Dreamer (best on val)": dreamer_policy(load_player(best[2]))}
        for house_set, suffix in (("test", ""), ("ood", "_ood")):
            res = H.test_table(pols, house_set, splits, control=args.control, context=args.context)
            H.report(res, f"Dreamer, {house_set} houses, test year")
            res.to_csv(out / f"{args.tag}_test{suffix}_continuous.csv", index=False)
        return

    test_env = make_env(splits["test"], household=args.household)
    actor_env = test_env if (args.action_mode, args.control) == ("power", "full") else \
        make_env(splits["test"], household=args.household, action_mode=args.action_mode, control=args.control)
    entries = baseline_entries(test_env, splits)
    entries["Dreamer (final)"] = policy_entry(actor_env, dreamer_policy(load_player(ckpts[-1][1])))
    entries["Dreamer (best on val)"] = policy_entry(actor_env, dreamer_policy(load_player(best[2])))
    if args.control == "ev":
        entries["oracle (EV only)"] = oracle_entry(test_env, cyclic=True, battery="self-consumption")
    entries["oracle (cyclic)"] = oracle_entry(test_env, cyclic=True)
    res = compare(entries, "continuous")
    report(res, list(entries))
    res.to_csv(out / f"{args.tag}_test_continuous.csv", index=False)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--sheeprl"]:                         # internal: the training process (see train)
        sys.path.insert(0, str(ROOT))
        run_sheeprl(sys.argv[2:])
        sys.exit(0)
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", nargs="?", default="runs/dreamer")
    p.add_argument("--household", default="ev_hp")
    p.add_argument("--action-mode", default="power", choices=["power", "target_soc", "residual"],
                   help="battery action: absolute kW, target SoC, or kW on top of the house's own surplus (0 = solar-only)")
    p.add_argument("--control", default="full", choices=["full", "ev"],
                   help="full: battery (+ wallbox); ev: wallbox only, battery on the solar-only rule, car parameters observed")
    p.add_argument("--ev-shaping", action="store_true", help="potential-based shaping of the EV deadline")
    p.add_argument("--houses", default="fixed", choices=["fixed", "gen"],
                   help="fixed: the one --household house; gen: random houses (houses.py), tested on new houses")
    p.add_argument("--context", default="observed", choices=["observed", "hidden"],
                   help="with --houses gen: car and house parameters in the observation or not")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="dreamer")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--accelerator", default="cpu", choices=["cpu", "mps"])
    p.add_argument("--size", default="S", choices=["XS", "S", "M", "L"])
    p.add_argument("--eval-only", action="store_true", help="score the checkpoints of an existing run")
    args = p.parse_args(argv)

    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    if not args.eval_only:
        train(out, args, extra)
    score(out, args)
