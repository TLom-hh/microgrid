"""Loading data from raw SMARD / PVGIS / OPSD files into one hourly DataFrame

Output columns:
    spot_price  EUR/kWh     day-ahead price
    solar_kw    kW          PV output
    load_kw     kW          household load, OPSD household scaled to 'annual_kwh'

"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

TIMEZONE = "Europe/Berlin"
INPUT = Path("input")
SMARD_FILE = INPUT / "Gro_handelspreise_202001010000_202601010000_Viertelstunde.csv"
PVGIS_FILE = INPUT / "Timeseries_53.036_10.290_SA3_35deg_0deg_2005_2023.csv"
OPSD_FILE = INPUT / "household_data_15min_singleindex_filtered (2).csv"

# SMARD prices
def load_smard(path: Path = SMARD_FILE) -> pd.Series:
    """Returns hourly price in EUR/kWh"""
    df = pd.read_csv(path, sep=";", decimal=",", thousands=".", na_values=["-"], encoding="utf-8-sig")
    df.columns = [c.replace(" [€/MWh] Originalauflösungen", "") for c in df.columns]
    t = pd.to_datetime(df["Datum von"], format="%d.%m.%Y %H:%M")
    prices = df["DE/AT/LU"].where(df["Deutschland/Luxemburg"].isna(), df["Deutschland/Luxemburg"])
    s = pd.Series(prices.values, index=t, name="spot_price")
    s.index = s.index.tz_localize(TIMEZONE, ambiguous="infer", nonexistent="shift_forward")
    s = s[~s.index.duplicated()].sort_index()
    return s.resample("1h").mean().div(1000.0).rename("spot_price")

# PVGIS solar energy
def load_pvgis(path: Path = PVGIS_FILE, peak_kwp: float = 10.0, system_loss: float = 0.14, gamma: float = -0.004, noct: float = 45.0) -> pd.Series:
    """PV power in kW 
    Data in PVGIS:
        G(i): Global irradiance on the inclined plane (plane of the array) (W/m2)
        H_sun: Sun height (degree)
        T2m: 2-m air temperature (degree Celsius)
        WS10m: 10-m total wind speed (m/s)
        Int: 1 means solar radiation values are reconstructed
   
    Standard crystalline silicon cells:
        theoretical peak kW as 'peak_kwp'
        energy lost in system as 'system_loss'
        power temperature coefficient of crytalline silicion (-0.4%/K) as 'gamma'
        nominal operating cell temperature of crystalline silicon as 'noct'
        
        T_cell = T2m + (noct - 20) / 800 * G
        P_dc = peak_kwp * G / 1000 * (1 + gamma * (T_cell - 25))
        P_ac = P_dc * (1 - system_loss)
    """

    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("time,"))
    end = next(i for i, l in enumerate(lines) if i > start and not (l[:1].isdigit()))
    df = pd.read_csv(path, skiprows=start, nrows=end - start - 1)
    t = pd.to_datetime(df["time"], format="%Y%m%d:%H%M").dt.floor("h")
    g, temp = df["G(i)"].values, df["T2m"].values
    t_cell = temp + (noct - 20.0) / 800 * g
    p_ac = peak_kwp * g / 1000.0 * (1 + gamma * (t_cell - 25.0)) * (1 - system_loss)
    s = pd.Series(np.clip(p_ac, 0 , None), index=t.dt.tz_localize("UTC"), name="solar_kw")
    return s.tz_convert(TIMEZONE)

# OPSD household data
def load_opsd_household(path: Path = OPSD_FILE, household: str = "residential5", annual_kwh: float | None = 4000.0) -> pd.Series:
    """Hourly household load (kW) from OPSD cumulative meter readings.
    
    grid_import used as total consumption (only valid for household without PV (residental2, residential5))
    optionally rescaled to 'annual_kwh'
    """

    col = f"DE_KN_{household}_grid_import"
    df = pd.read_csv(path, usecols=["utc_timestamp", col], parse_dates=["utc_timestamp"], index_col=["utc_timestamp"])
    cum = df[col].dropna()
    kwh_15min = cum.diff().clip(lower=0)
    kwh_15min.index = kwh_15min.index.tz_convert(TIMEZONE)
    hourly_kwh = kwh_15min.resample("1h").sum(min_count=4)
    load_kw = hourly_kwh.rename("load_kw")
    if annual_kwh is not None:
        observed = load_kw.mean() * 8760
        load_kw = load_kw * (annual_kwh / observed)
    return load_kw

def align_load_index(load: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """Reuse measured loads for timeframes the data does not cover
    household load without electric heating is independent of same-day weather, so shifting to different years is acceptable
    """

    src = load.dropna()
    src_key = pd.DataFrame({"y": src.index.isocalendar().year.values, "w": src.index.isocalendar().week.values, "d": src.index.weekday, "h": src.index.hour, "v": src.values})
    src_key = src_key.drop_duplicates(["y", "w", "d", "h"])
    src_key["y"] = src_key["y"].astype(int)
    years = sorted(src_key.y.unique())
    out = np.full(len(target_index), np.nan)
    tgt = pd.DataFrame({"y": target_index.isocalendar().year.values.astype(int), "w": target_index.isocalendar().week.values.astype(int), "d": target_index.weekday, "h": target_index.hour})
    for ty in tgt.y.unique():
        mask = (tgt.y == ty).values
        for sy in sorted(years, key=lambda y: abs(y - ty)):
            block = src_key[src_key.y == sy].set_index(["w", "d", "h"])["v"]
            keys = pd.MultiIndex.from_frame(tgt.loc[mask, ["w", "d", "h"]])
            vals = block.reindex(keys).values
            fill = mask.copy(); fill[mask] = np.isnan(out[mask])
            out[fill] = vals[np.isnan(out[mask])]
            if not np.isnan(out[mask]).any():
                break
    return pd.Series(out, index=target_index, name="load_kw").interpolate(limit=6)


# Combine Series
def build_dataset(peak_kwp: float = 10.0, household: str = "residential5", annual_kwh: float | None = 4000.0, max_gap_hours: int = 6, reuse_load: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Join three sources to single output on comon window
    
    reuse_load=True: window is only set by prices and solar, load is re-aligned with 'align_load_index' to uncovered years
    """

    price, solar, load = load_smard(), load_pvgis(peak_kwp=peak_kwp), load_opsd_household(household=household, annual_kwh=annual_kwh)
    series = [price, solar] if reuse_load else [price, solar, load]
    start = max(x.first_valid_index() for x in series)
    end = min(x.last_valid_index() for x in series)
    idx = pd.date_range(start, end, freq="1h", tz=TIMEZONE)
    if reuse_load:
        load = align_load_index(load, idx)
    df = pd.concat([price.reindex(idx), solar.reindex(idx), load.reindex(idx)], axis=1)

    report =  {}
    for c in df.columns:
        na = df[c].isna()
        runs = (na != na.shift()).cumsum()[na]
        run_len = runs.value_counts()
        report[c] = dict(missing=int(na.sum()), long_gaps=int((run_len > max_gap_hours).sum()), longest=int(run_len.max()) if len(run_len) else 0)

    df = df.interpolate(limit=max_gap_hours, limit_area="inside")
    long_missing = df.isna().any(axis=1)
    if verbose:
        print(f"window {start:%Y-%m-%d} -> {end:%Y%m%d} ({len(idx)} h, {len(idx)/8760:.2f} yr)")
        for c, r in report.items():
            print(f" {c:11s} missing {r['missing']:5d} h, gaps > {max_gap_hours} h: {r['long_gaps']:3d} (longest {r['longest']} h)")
        print(f"  hours still NaN after interpolation: {int(long_missing.sum())}")
    df.attrs["long_gap_hours"] = int(long_missing.sum())
    return df

def summary(df: pd.DataFrame) -> pd.DataFrame:
    yearly = df.groupby(df.index.year).agg(
        hours=("spot_price", "size"),
        price_mean=("spot_price", "mean"), price_min=("spot_price", "min"), price_max=("spot_price", "max"),
        solar_kwh=("solar_kw", "sum"), load_kwh=("load_kw", "sum"))
    yearly["solar_kwh_per_kwp_annualised"] = yearly["solar_kwh"] / yearly["hours"] * 8760 / 10.0
    return yearly.round(3)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/microgrid_hourly.csv")
    ap.add_argument("--peak-kwp", type=float, default=10.0)
    ap.add_argument("--household", default="residential5")
    ap.add_argument("--annual-kwh", type=float, default=4000.0)
    ap.add_argument("--reuse-load", action="store_true", help="align OPSD load onto the price/solar window by weekday")
    args = ap.parse_args()

    df = build_dataset(peak_kwp=args.peak_kwp, household=args.household, annual_kwh=args.annual_kwh, reuse_load=args.reuse_load)
    print(summary(df).to_string())
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index_label="time")
    print(f"wrote {out} ({len(df)} rows)")
