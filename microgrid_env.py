"""Smart Microgrid Environment: a home battery under dynamic prices and PV, optionally with a heat pump and an EV.

Household devices (devices.py), switched on with the 'heat_pump' and 'ev' arguments (None / True / dict of params):
    heat pump   uncontrolled: its electric demand (from df['temp_c']) is added to load_kw; household_kw keeps the
                base load and hp_kw the heat pump's share
    ev          controllable: a second action dimension sets the wallbox power while the car is home; a missed
                departure target is bought at the public charging price (part of 'cost'). See devices.EV.

Action: a in [-1, 1]^d, d = 1 (battery) or 2 (battery, EV). The EV part maps linearly to (a[1] + 1) / 2 * EV
max power (a[1] = -1: off). The battery part's meaning depends on 'action_mode':
    power       battery power setpoint a * max_power_kw (default)
    target_soc  a -> target SoC (a + 1) / 2; the env applies the grid-side power that reaches it this step
    residual    battery_kw = (solar - load - ev_kw) + a * max_power_kw; a = 0 is pure self-consumption
    a > 0 charges the battery, a < 0 discharges it (in 'power' and 'residual')
    kw_to_action(kw) is the inverse map, used by the scripted baselines and the oracle replay
    agent never chooses where energy goes
    grid balances what is left: grid_kw = load_kw - solar_kw + battery_kw + ev_kw
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
from devices import EV, HeatPump, make_device

FEATURE_DOC = {
    "soc":              "battery state of charge in [0, 1]",
    "price":            "current buy price in EUR/kWh",
    "solar":            "current PV generation in kW",
    "load":             "current household load in kW",
    "time":             "sin/cos of hour of day",
    "season":           "sin/cos of day of year",
    "price_forecast":   "buy prices for the next 'forecast_horizon' hours",
    "solar_forecast":   "PV generation for the next 'forecast_horizon' hours, plus noise (synthetic forecast)",
    "distractor":       "'n_distractors' observable signals with no effect",
    "ev":               "EV plugged in (0/1), EV SoC, hours to the planned departure / 24 (capped at 1, 0 while away); zeros without an EV",
    "temperature":      "outdoor temperature / 20 °C (drives the heat pump load)",
    "ev_context":       "the car's fixed parameters: battery kWh / 100, wallbox kW / 11, kWh per km / 0.2, target SoC "
                        "(observed context, so one agent can serve different cars); zeros without an EV",
    "house_context":    "the house's fixed parameters: PV kWp / 10, household kWh per year / 4000, battery kWh / 10, "
                        "battery kW / 5, heat pump heat loss (kW/K) / 0.22 (0 without a heat pump); for one agent across houses (houses.py)",
}

REFERENCE_KWP = 10.0       # the PV size and household demand data.py builds the dataset with; pv_kwp and annual_kwh
REFERENCE_KWH = 4000.0     # scale the solar and household load columns linearly relative to them

SNAPSHOT = ("soc", "price", "solar", "load", "time", "season", "price_forecast", "solar_forecast")
ACTION_MODES = ("power", "target_soc", "residual")
CONTROLS = ("full", "ev")

class SmartMicrogridEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
            self,
            df: pd.DataFrame | None = None,
            obs_features: tuple[str, ...] | None = None,
            forecast_horizon: int = 24,
            n_distractors: int = 2,
            n_discrete_actions: int | None = None,
            action_mode: str = "power",
            episode_hours: int = 24 * 7,
            capacity_kwh: float = 10.0,
            max_power_kw: float = 5.0,
            efficiency: float = 0.95,
            degradation_cost_per_kwh:float = 0.02,
            buy_markup: float = 0.17,
            feed_in_price: float = 0.077,
            normalize: bool = True,
            record_history: bool = True,
            heat_pump: dict | bool | None = None,
            ev: dict | bool | None = None,
            control: str = "full",
            pv_kwp: float = REFERENCE_KWP,
            annual_kwh: float = REFERENCE_KWH,
    ):
        """obs_features=None: SNAPSHOT, plus 'ev' with an EV, 'temperature' with a heat pump and 'ev_context'
        with control='ev'.

        control   'full': the agent sets the battery (and, with an EV, the wallbox): action (battery, ev).
                  'ev':   the agent sets only the wallbox (1-D action); the battery follows the solar-only rule
                          (baselines.py: store the PV surplus left after the car, cover the house's own deficit),
                          clipped to what is feasible, so it never makes an infeasible request. action_mode does
                          not apply (it is the battery's interface).
        pv_kwp, annual_kwh  PV size and household demand (without heat pump) of this house: the dataset's solar and
                  load columns are scaled linearly from the reference house (10 kWp, 4000 kWh/year; PV output is
                  linear in kWp in data.load_pvgis). Defaults leave the data unchanged."""
        super().__init__()
        self.heat_pump = make_device(HeatPump, heat_pump)
        self.ev = make_device(EV, ev)
        if control not in CONTROLS:
            raise ValueError(f"unknown control {control!r}; choose from {CONTROLS}")
        if control == "ev" and not self.ev:
            raise ValueError("control='ev' needs a household with an EV")
        self.control = control
        if obs_features is None:
            obs_features = (SNAPSHOT + (("ev",) if self.ev else ()) + (("temperature",) if self.heat_pump else ())
                            + (("ev_context",) if control == "ev" else ()))
        unknown = set(obs_features) - set(FEATURE_DOC)
        if unknown:
            raise ValueError(f"unkown obs features {unknown}; choose from {list(FEATURE_DOC)}")
        self.obs_features = tuple(obs_features)
        self.forecast_horizon = forecast_horizon
        self.n_distractors = n_distractors
        self.episode_hours = episode_hours
        self._hours = episode_hours
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
        self.pv_kwp, self.annual_kwh = float(pv_kwp), float(annual_kwh)
        if self.pv_kwp != REFERENCE_KWP:
            self.df["solar_kw"] = self.df["solar_kw"] * (self.pv_kwp / REFERENCE_KWP)
        if self.annual_kwh != REFERENCE_KWH:
            self.df["load_kw"] = self.df["load_kw"] * (self.annual_kwh / REFERENCE_KWH)
        self.df["household_kw"] = self.df["load_kw"]
        self.df["hp_kw"] = self.heat_pump.electric_kw(self.df["temp_c"]).values if self.heat_pump else 0.0
        self.df["load_kw"] = self.df["household_kw"] + self.df["hp_kw"]       # everything the agent cannot shift
        if self.ev:
            self.df = self.df.join(self.ev.schedule(self.df.index))
        self.df["buy_price"] = self.df["spot_price"] + buy_markup
        self.df["sell_price"] = feed_in_price
        needed = episode_hours + forecast_horizon + 1
        midnights = np.flatnonzero(self.df.index.hour == 0)
        self._start_positions = midnights[midnights + needed <= len(self.df)]
        if len(self._start_positions) == 0:
            raise ValueError("data too short for episode length + forecast horizon")

        self._scales = {"price": 0.5, "solar": 5.0, "load": 5.0}

        if action_mode not in ACTION_MODES:
            raise ValueError(f"unknown action_mode {action_mode!r}; choose from {ACTION_MODES}")
        self.action_mode = action_mode
        self.n_discrete_actions = n_discrete_actions
        if n_discrete_actions is not None and self.ev:
            raise ValueError("discrete actions are only implemented for the battery-only env")
        self.action_dim = 1 if control == "ev" else (2 if self.ev else 1)
        if n_discrete_actions is None:
            self.action_space = spaces.Box(-1.0, 1.0, shape=(self.action_dim,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(n_discrete_actions)
            self._discrete_levels = np.linspace(-1.0, 1.0, n_discrete_actions)

        n = self._obs_dim()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n,), dtype=np.float32)

        self.history: list[dict] = []

    def reset(self, seed=None, options=None):
        """options (all optional, for deterministic evaluation):
            start_step     row index of the episode start (must be a midnight with room for the episode)
            soc            initial state of charge instead of uniform(0.2, 0.8)
            ev_soc         initial EV state of charge instead of uniform(0.3, 0.9) (ignored without an EV;
                           the car is always home at midnight, where every episode starts)
            episode_hours  episode length for this episode only (e.g. a whole test split as one continuing run)
        """
        super().reset(seed=seed)
        options = options or {}
        self._hours = int(options.get("episode_hours", self.episode_hours))
        needed = self._hours + self.forecast_horizon + 1
        if "start_step" in options:
            start = int(options["start_step"])
            if self.df.index[start].hour != 0 or start + needed > len(self.df):
                raise ValueError(f"start_step {start} is not a midnight with {needed} rows of data after it")
        elif self._hours == self.episode_hours:
            start = int(self.np_random.choice(self._start_positions))
        else:
            start = int(self.np_random.choice(self.episode_starts(stride_hours=24, episode_hours=self._hours)))
        self.start_step = start
        self.current_step = self.start_step
        self.soc = float(options["soc"]) if "soc" in options else float(self.np_random.uniform(0.2, 0.8))
        self._distractor_state = self.np_random.normal(size=self.n_distractors)
        self._distractor_phase = self.np_random.uniform(0, 2 * np.pi, size=self.n_distractors)
        if self.ev:
            ev_soc = float(options["ev_soc"]) if "ev_soc" in options else float(self.np_random.uniform(0.3, 0.9))
            self.ev_energy = ev_soc * self.ev.capacity_kwh
        self.history = []
        return self._get_obs(), self._info()

    def episode_starts(self, stride_hours: int | None = None, episode_hours: int | None = None) -> np.ndarray:
        """Row indices of consecutive evaluation episodes: from the first midnight in the data, every
        'stride_hours' (default: one episode length, i.e. non-overlapping), as long as an episode plus its
        forecast horizon fits. Deterministic, so every policy is scored on exactly the same weeks."""
        hours = self.episode_hours if episode_hours is None else int(episode_hours)
        stride = hours if stride_hours is None else int(stride_hours)
        midnights = np.flatnonzero(self.df.index.hour == 0)
        if stride % 24 == 0:                     # calendar stride: stays on midnight across DST changes
            cand = midnights[::stride // 24]
        else:
            cand = np.arange(midnights[0], len(self.df), stride)
        needed = hours + self.forecast_horizon + 1   # an episode is exactly 'hours' steps, also across a DST change
        return cand[cand + needed <= len(self.df)].astype(int)

    def step(self, action):
        row = self.df.iloc[self.current_step]
        a = self._parse_action(action)
        ev_requested_kw = self._ev_request(a)
        ev_kw = self._feasible_ev(ev_requested_kw)
        requested_kw = self._battery_request(a[0], ev_kw)
        actual_kw = self._feasible_power(requested_kw)
        soc_before = self.soc
        if self.ev:
            ev_soc_before = self.ev_energy / self.ev.capacity_kwh
            self.ev_energy = min(self.ev_energy + ev_kw * self.dt * self.ev.efficiency, self.ev.capacity_kwh)

        if actual_kw >= 0:
            self.soc += actual_kw * self.dt * self.efficiency / self.capacity_kwh
        else:
            self.soc += actual_kw * self.dt / self.efficiency / self.capacity_kwh
        self.soc = float(np.clip(self.soc, 0.0, 1.0))

        grid_kw = row["load_kw"] - row["solar_kw"] + actual_kw + ev_kw
        energy_kwh = grid_kw * self.dt
        if energy_kwh >= 0:
            cost = energy_kwh * row["buy_price"]
        else:
            cost = energy_kwh * row["sell_price"]
        degradation = abs(actual_kw) * self.dt * self.degradation_cost_per_kwh

        self.current_step += 1
        self._advance_distractors()
        shortfall_kwh = self._ev_events() if self.ev else 0.0          # departure / arrival at the start of the next hour
        ev_public_cost = shortfall_kwh * self.ev.public_price if self.ev else 0.0
        cost += ev_public_cost
        reward = -(cost + degradation)

        if self.record_history:
            self.history.append(dict(
                time=row.name, 
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
                household_kw=row["household_kw"],
                hp_kw=row["hp_kw"],
            ))
            if self.ev:
                self.history[-1].update(
                    ev_home=bool(row["ev_home"]), ev_requested_kw=ev_requested_kw, ev_kw=ev_kw,
                    ev_soc_before=ev_soc_before, ev_soc=self.ev_energy / self.ev.capacity_kwh,
                    ev_shortfall_kwh=shortfall_kwh, ev_public_cost=ev_public_cost)

        terminated = False
        truncated = (self.current_step - self.start_step) >= self._hours
        return self._get_obs(), float(reward), terminated, truncated, self._info()

    def _parse_action(self, action) -> np.ndarray:
        if self.n_discrete_actions is not None:
            return np.array([float(self._discrete_levels[int(action)])])
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        if len(a) != self.action_dim:
            raise ValueError(f"expected an action of size {self.action_dim}, got {len(a)}")
        return a

    def _ev_request(self, a) -> float:
        return float((a[-1] + 1.0) / 2.0 * self.ev.max_power_kw) if self.ev else 0.0

    def _battery_request(self, a0: float, ev_kw: float = 0.0) -> float:
        """Requested grid-side battery power for the battery part of an action, before feasibility clipping.
        With control='ev' the battery is not part of the action: the solar-only rule, already feasible."""
        if self.control == "ev":
            from baselines import _bat_solar_only
            return self._feasible_power(_bat_solar_only(self, ev_kw))
        if self.action_mode == "power":
            return a0 * self.max_power_kw
        if self.action_mode == "residual":
            row = self.df.iloc[self.current_step]
            return (row["solar_kw"] - row["load_kw"] - ev_kw) + a0 * self.max_power_kw
        delta_kwh = ((a0 + 1.0) / 2.0 - self.soc) * self.capacity_kwh          # target_soc
        return delta_kwh / self.efficiency / self.dt if delta_kwh > 0 else delta_kwh * self.efficiency / self.dt

    def _action_to_kw(self, action) -> float:
        """Requested grid-side battery power for an action, before feasibility clipping (TrainingReward uses it)."""
        a = self._parse_action(action)
        return self._battery_request(a[0], self._feasible_ev(self._ev_request(a)))

    def kw_to_action(self, kw: float, ev_kw: float = 0.0) -> np.ndarray:
        """Inverse map for the current state (continuous action space only), clipped to [-1, 1]: battery kW
        (and, with an EV, the requested wallbox kW) -> action. With control='ev' the battery kW is ignored."""
        ev_a = float(np.clip(2.0 * ev_kw / self.ev.max_power_kw - 1.0, -1.0, 1.0)) if self.ev else None
        if self.control == "ev":
            return np.array([ev_a])
        if self.action_mode == "power":
            a = kw / self.max_power_kw
        elif self.action_mode == "residual":
            row = self.df.iloc[self.current_step]
            a = (kw - (row["solar_kw"] - row["load_kw"] - self._feasible_ev(ev_kw))) / self.max_power_kw
        else:
            delta_kwh = kw * self.dt * self.efficiency if kw > 0 else kw * self.dt / self.efficiency
            a = 2.0 * (self.soc + delta_kwh / self.capacity_kwh) - 1.0
        out = [float(np.clip(a, -1.0, 1.0))]
        if self.ev:
            out.append(ev_a)
        return np.array(out)

    def _feasible_ev(self, requested_kw: float) -> float:
        """Wallbox power the car can take this hour: 0 while away, never more than max power or the headroom."""
        if not self.ev or not self.df["ev_home"].iat[self.current_step]:
            return 0.0
        headroom_kw = (self.ev.capacity_kwh - self.ev_energy) / (self.ev.efficiency * self.dt)
        return float(max(0.0, min(requested_kw, self.ev.max_power_kw, headroom_kw)))

    def _ev_events(self) -> float:
        """Events at the start of the (new) current hour: departure tops a short car up to target_soc (bought at
        the public price; returns that kWh), arrival removes the trip's energy."""
        row = self.df.iloc[self.current_step]
        shortfall = 0.0
        if row["ev_departs"]:
            shortfall = max(0.0, self.ev.target_soc * self.ev.capacity_kwh - self.ev_energy)
            self.ev_energy += shortfall
        self.ev_energy -= row["ev_trip_kwh"]
        return shortfall

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
            "distractor": self.n_distractors,
            "ev": 3,
            "temperature": 1,
            "ev_context": 4,
            "house_context": 5,
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
                case "ev":
                    if self.ev:
                        home = float(row["ev_home"])
                        blocks.append([home, self.ev_energy / self.ev.capacity_kwh,
                                       home * min(row["ev_hours_to_departure"], 24.0) / 24.0])
                    else:
                        blocks.append([0.0, 0.0, 0.0])
                case "temperature":
                    blocks.append([row["temp_c"] / 20.0])
                case "ev_context":
                    car = self.ev
                    blocks.append([car.capacity_kwh / 100.0, car.max_power_kw / 11.0, car.kwh_per_km / 0.2,
                                   car.target_soc] if car else [0.0, 0.0, 0.0, 0.0])
                case "house_context":
                    hp = self.heat_pump.heat_loss_kw_per_k / 0.22 if self.heat_pump else 0.0
                    blocks.append([self.pv_kwp / REFERENCE_KWP, self.annual_kwh / REFERENCE_KWH,
                                   self.capacity_kwh / 10.0, self.max_power_kw / 5.0, hp])
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
        info = {"step": self.current_step - self.start_step, "soc": self.soc}
        if self.ev:
            info["ev_soc"] = self.ev_energy / self.ev.capacity_kwh
        return info

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