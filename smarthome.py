"""Smart Microgrid Environment: a home battery under dynamic prices and PV.

Action: one continuous scalar a in [-1, 1] -> battery power setpoint a * max_power_kw
    a > 0 charges the battery, a < 0 discharges it
    agent never chooses where energy goes
    grid balances what is left: grid_kw = load_kw - solar_kw + battery_kw
    grid_kw > 0 -> buying at spot_price
    grid_kw < 0 -> selling at feed_in_price
    request that the battery cannot physically satisfy are clipped

Observation: see FEATURE_DOC

Time: one step = 'dt' hours (1.0). kW * dt = kWh
"""

from __future__ import annotations
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

FEATURE_DOC = {
    "soc":              "battery state of charge in [0, 1]",
    "price":            "current buy price in EUR/kWh",
    "solar":            "current PV generation in kW",
    "load":             "current household load in kW",
    "time":             "sin/cos of hour of day",
    "season":           "sin/cos of day of year",
    "price_forecast":   "buy prices for the next 'forecast_horizon' hours",
    "solar_forecast":   "PV generation for the next 'forecast_horizon' hours, plus noise (synthetic forecast)",
    "distractor":       "'n_distractors' observable signals with no effect"
}

SNAPSHOT = ("soc", "price", "solar", "load", "time", "season", "price_forecast", "solar_forecast")

class SmartMicrogridEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
            self,
            df: pd.DataFrame | None = None,
            obs_features: tuple[str, ...] = SNAPSHOT,
            forecast_horizon: int = 24,
            n_distractors: int = 2,
            n_discrete_actions: int | None = None,
            episode_hours: int = 24 * 7,
            capacity_kwh: float = 10.0,
            max_power_kw: float = 5.0,
            efficiency: float = 0.95,
            degradation_cost_per_kwh:float = 0.02,
            buy_markup: float = 0.17,
            feed_in_price: float = 0.077,
            normalize: bool = True,
            record_history: bool = True
    ):
        super().__init__()
        unknown = set(obs_features) - set(FEATURE_DOC)
        if unknown:
            raise ValueError(f"unkown obs features {unknown}; choose from {list(FEATURE_DOC)}")
        self.obs_features = tuple(obs_features)
        self.forecast_horizon = forecast_horizon
        self.n_distractors = n_distractors
        self.episode_hours = episode_hours
        self.dt = 1.0
        self.capacity_kwh = capacity_kwh
        self.max_power_kw = max_power_kw
        self.efficiency = efficiency
        self.degradation_cost_per_kwh = degradation_cost_per_kwh
        self.buy_markup = buy_markup
        self.feed_in_price = feed_in_price
        self.normalize = normalize
        self.record_history = record_history

        self.df = df if df is not None else generate_dummy_data(days=365)
        self.df = self.df.copy()
        self.df["buy_price"] = self.df["spot_price"] + buy_markup
        self.df["sell_price"] = feed_in_price
        needed = episode_hours + forecast_horizon + 1
        midnights = np.flatnonzero(self.df.index.hour == 0)
        self._start_positions = midnights[midnights + needed <= len(self.df)]
        if len(self._start_positions) == 0:
            raise ValueError("data too short for episode length + forecast horizon")

        self._scales = {"price": 0.5, "solar": 5.0, "load": 5.0}

        self.n_discrete_actions = n_discrete_actions
        if n_discrete_actions is None:
            self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(n_discrete_actions)
            self._discrete_levels = np.linspace(-1.0, 1.0, n_discrete_actions)

        n = self._obs_dim()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n,), dtype=np.float32)

        self.history: list[dict] = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.start_step = int(self.np_random.choice(self._start_positions))
        self.current_step = self.start_step
        self.soc = float(self.np_random.uniform(0.2, 0.8))
        self._distractor_state = self.np_random.normal(size=self.n_distractors)
        self._distractor_phase = self.np_random.uniform(0, 2 * np.pi, size=self.n_distractors)
        self.history = []
        return self._get_obs(), self._info()

    def step(self, action):
        row = self.df.iloc[self.current_step]
        requested_kw = self._action_to_kw(action)
        actual_kw = self._feasible_power(requested_kw)
        soc_before = self.soc

        if actual_kw >= 0:
            self.soc += actual_kw * self.dt * self.efficiency / self.capacity_kwh
        else:
            self.soc += actual_kw * self.dt / self.efficiency / self.capacity_kwh
        self.soc = float(np.clip(self.soc, 0.0, 1.0))

        grid_kw = row["load_kw"] - row["solar_kw"] + actual_kw
        energy_kwh = grid_kw * self.dt
        if energy_kwh >= 0:
            cost = energy_kwh * row["buy_price"]
        else:
            cost = energy_kwh * row["sell_price"]
        degradation = abs(actual_kw) * self.dt * self.degradation_cost_per_kwh
        reward = -(cost + degradation)

        if self.record_history:
            self.history.append(dict(
                time=self.df.index[self.current_step], 
                spot_price=row["spot_price"],
                buy_price=row["buy_price"], 
                sell_price=row["sell_price"],
                solar_kw=row["solar_kw"], 
                load_kw=row["load_kw"],
                requested_kw=requested_kw, 
                battery_kw=actual_kw,
                clipped=abs(requested_kw - actual_kw) > 1e-9,
                soc_before=soc_before, 
                soc=self.soc, 
                grid_kw=grid_kw,
                cost=cost, 
                degradation=degradation, 
                reward=reward,
            ))

        self.current_step += 1
        self._advance_distractors()
        terminated = False
        truncated = (self.current_step - self.start_step) >= self.episode_hours
        return self._get_obs(), float(reward), terminated, truncated, self._info()

    def _action_to_kw(self, action) -> float:
        if self.n_discrete_actions is None:
            a = float(np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[0], -1.0, 1.0))
        else:
            a = float(self._discrete_levels[int(action)])
        return a * self.max_power_kw

    def _feasible_power(self, requested_kw: float) -> float:
        if requested_kw >= 0:
            headroom_kw = (1.0 - self.soc) * self.capacity_kwh / (self.efficiency * self.dt)
            return min(requested_kw, self.max_power_kw, headroom_kw)
        available_kw = self.soc * self.capacity_kwh * self.efficiency / self.dt
        return max(requested_kw, -self.max_power_kw, -available_kw)

    def _obs_dim(self) -> int:
        sizes = {   
            "soc": 1, 
            "price": 1, 
            "solar": 1, 
            "load": 1, 
            "time": 2, 
            "season": 2,
            "price_forecast": self.forecast_horizon, 
            "solar_forecast": self.forecast_horizon,
            "distractor": self.n_distractors
        }
        return sum(sizes[f] for f in self.obs_features)

    def _get_obs(self) -> np.ndarray:
        i = self.current_step
        row = self.df.iloc[i]
        ts = self.df.index[i]
        s = self._scales if self.normalize else {"price": 1.0, "solar": 1.0, "load": 1.0}
        blocks = []
        for f in self.obs_features:
            match f:
                case "soc":
                    blocks.append([self.soc])
                case "price":
                    blocks.append([row["buy_price"] / s["price"]])
                case "solar":
                    blocks.append([row["solar_kw"] / s["solar"]])
                case "load":
                    blocks.append([row["load_kw"] / s["load"]])
                case "time":
                    ang = 2 * np.pi * ts.hour / 24
                    blocks.append([np.sin(ang), np.cos(ang)])
                case "season":
                    ang = 2 * np.pi * ts.dayofyear / 365.25
                    blocks.append([np.sin(ang), np.cos(ang)])
                case "price_forecast":
                    fut = self.df["buy_price"].values[i+1: i+1+self.forecast_horizon]
                    blocks.append(fut / s["price"])
                case "solar_forecast":
                    fut = self.df["solar_kw"].values[i + 1: i + 1 + self.forecast_horizon]
                    noise = self.np_random.normal(0, 0.15, size=fut.shape) * fut  # 15% relative error
                    blocks.append(np.clip(fut + noise, 0, None) / s["solar"])
                case "distractor":
                    blocks.append(self._distractor_values())
        return np.concatenate(blocks).astype(np.float32)

    def _distractor_values(self) -> np.ndarray:
        vals = np.empty(self.n_distractors)
        for k in range(self.n_distractors):
            if k == 0:
                vals[k] = self._distractor_state[k]
            else:
                hour = self.df.index[self.current_step].hour
                vals[k] = np.sin(2 * np.pi * hour / 24 + self._distractor_phase[k])
        return vals;

    def _advance_distractors(self):
        if self.n_distractors > 0:
            self._distractor_state[0] = np.clip(self._distractor_state[0] + self.np_random.normal(0, 0.3), -2, 2)

    def current_row(self) -> pd.Series:
        return self.df.iloc[self.current_step]

    def _info(self) -> dict:
        return {"step": self.current_step - self.start_step, "soc": self.soc}

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.history).set_index("time")

#DUMMY DATA
def generate_dummy_data(days: int = 365, seed: int = 0, start: str = "2025-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 24, freq="h")
    hour = idx.hour.values;
    doy = idx.dayofyear.values
    weekend = (idx.dayofweek.values >= 5).astype(float)

    season = 0.5 -0.5 * np.cos(2 * np.pi * (doy - 172) / 365.25)    # 1 in midsummer, 0 midwinter
    daylight = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)    # 0 at night
    cloud = np.repeat(rng.uniform(0.3, 1.0, size=days), 24)
    solar = 4.0 * (0.4 + 0.6 * season) * daylight ** 1.5 * cloud

    base = 0.4
    morning = 1.2 * np.exp(-0.5 * ((hour - 7.5) / 1.2) ** 2)
    evening = 2.0 * np.exp(-0.5 * ((hour -19) / 1.8) ** 2)
    load = base + (morning + evening) * (1 + 0.3 * weekend) * (1.2 - 0.4 * season)
    load = load * rng.lognormal(0, 0.15, size=len(idx))

    peak = 0.12 * np.exp(-0.5 * ((hour - 18.5) / 2.0) ** 2) + 0.05 * np.exp(-0.5 * ((hour - 8) / 1.5) ** 2)
    midday_dip = - 0.06 * season * daylight **2
    spot = 0.08 + peak + midday_dip - 0.02 * weekend + rng.normal(0, 0.015, size=len(idx))
    spot = spot + np.repeat(rng.normal(0, 0.02, size=days), 24)
    spot = np.clip(spot, -0.02, None)

    return pd.DataFrame({"spot_price": spot, "solar_kw": solar, "load_kw": load}, index=idx)