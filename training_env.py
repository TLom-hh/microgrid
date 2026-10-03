"""The training reward, shared by every learning agent (SAC, VPG, later DreamerV3).

TrainingReward wraps a SmartMicrogridEnv and replaces the reward of step() by

    saving - clip_penalty * infeasible_kwh

    saving          env reward minus the reward an idle battery would have earned in the same hour
                    (no_battery_reward). The subtracted term depends only on the hour's load, PV and prices,
                    not on the action or the SoC, so the optimal policy is unchanged; it removes the large
                    uncontrollable part of the bill that the critic would otherwise have to model.
    infeasible_kwh  |requested - delivered| battery energy of the step. At an empty battery a discharge
                    request is clipped and free while a charge request costs money now, which pulls every
                    gradient learner towards 'discharge'. Charging clip_penalty EUR per infeasible kWh
                    removes that bias; an optimal policy makes no infeasible requests, so it pays nothing.

Evaluation never uses this wrapper: evaluation.py scores the plain env cost. The wrapped env's history and
info["env_reward"] still hold the plain values.
"""

import gymnasium as gym

def no_battery_reward(env):
    """Reward the env would give this hour with the battery idle (grid covers load - solar).
    Call before env.step(): it reads the row the agent is about to act in."""
    row = env.current_row()
    net_kwh = (row["load_kw"] - row["solar_kw"]) * env.dt
    return -(net_kwh * row["buy_price"] if net_kwh > 0 else net_kwh * row["sell_price"])

class TrainingReward(gym.Wrapper):
    def __init__(self, env, clip_penalty: float = 0.0):
        super().__init__(env)
        self.clip_penalty = clip_penalty

    def step(self, action):
        env = self.env
        base = no_battery_reward(env)
        requested_kw = env._action_to_kw(action)
        infeasible_kwh = abs(requested_kw - env._feasible_power(requested_kw)) * env.dt
        obs, reward, terminated, truncated, info = env.step(action)
        saving = reward - base
        penalty = self.clip_penalty * infeasible_kwh
        info = dict(info, env_reward=reward, saving=saving, clip_penalty=penalty)
        return obs, saving - penalty, terminated, truncated, info

    def __getattr__(self, name):
        """Forward everything else (soc, df, current_row, kw_to_action, history, ...) to the wrapped env,
        so baselines and helpers that take an env work on the wrapped one too."""
        if name == "env" or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.env, name)
