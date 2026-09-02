import numpy as np
from matplotlib import pyplot as plt

C = {
    "price": "#2a78d6",   # blue
    "load": "#eb6834",    # orange
    "battery": "#1baf7a", # aqua
    "solar": "#eda100",   # yellow
    "soc": "#4a3aa7",     # violet
    "import": "#e34948",  # red  (diverging pole: buying)
    "export": "#2a78d6",  # blue (diverging pole: selling)
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#9a9891", "surface": "#fcfcfb", "grid": "#e6e5e1",
}

def _style(ax, ylabel: str):
    ax.set_facecolor(C["surface"])
    ax.grid(True, axis="y", color=C["grid"], linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C["grid"])
    ax.tick_params(colors=C["ink2"], labelsize=8)
    ax.set_ylabel(ylabel, color=C["ink2"], fontsize=9)

def plot_episode(hist, title: str = "Episode"):
    """Stacked time-series view of one episode.

    hist: DataFrame from SmartMicrogridEnv.history_frame() (index = timestamps).
    Panels: prices | solar & load | battery power | SoC | grid exchange | cumulative cost.
    """
    t = hist.index
    fig, axs = plt.subplots(6, 1, figsize=(12, 12), sharex=True)
    fig.patch.set_facecolor(C["surface"])

    ax = axs[0]
    ax.plot(t, hist["buy_price"], color=C["price"], lw=2, label="buy price")
    ax.plot(t, hist["sell_price"], color=C["muted"], lw=1.5, ls="--", label="feed-in price")
    _style(ax, "EUR/kWh")
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)

    ax = axs[1]
    ax.plot(t, hist["solar_kw"], color=C["solar"], lw=2, label="solar")
    ax.plot(t, hist["load_kw"], color=C["load"], lw=2, label="load")
    _style(ax, "kW")
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)

    ax = axs[2]
    ax.step(t, hist["requested_kw"], where="post", color=C["muted"], lw=1.5, ls="--", label="requested")
    ax.step(t, hist["battery_kw"], where="post", color=C["battery"], lw=2, label="actual")
    ax.axhline(0, color=C["grid"], lw=1)
    _style(ax, "battery kW\n(+ charge)")
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)

    ax = axs[3]
    ax.plot(t, hist["soc"], color=C["soc"], lw=2)
    ax.set_ylim(0, 1)
    _style(ax, "SoC")

    ax = axs[4]
    g = hist["grid_kw"].values
    ax.fill_between(t, 0, np.where(g > 0, g, 0), step="post", color=C["import"], alpha=0.35, lw=0, label="import (buy)")
    ax.fill_between(t, 0, np.where(g < 0, g, 0), step="post", color=C["export"], alpha=0.35, lw=0, label="export (sell)")
    ax.step(t, g, where="post", color=C["ink2"], lw=1)
    ax.axhline(0, color=C["grid"], lw=1)
    _style(ax, "grid kW\n(+ import)")
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)

    ax = axs[5]
    cum = (hist["cost"] + hist["degradation"]).cumsum()
    ax.plot(t, cum, color=C["ink"], lw=2)
    ax.annotate(f"{cum.iloc[-1]:.2f} EUR", xy=(t[-1], cum.iloc[-1]), xytext=(-4, 4),
                textcoords="offset points", ha="right", fontsize=9, color=C["ink"])
    _style(ax, "cumulative\ncost EUR")
    ax.set_xlabel("time", color=C["ink2"], fontsize=9)

    fig.suptitle(title, color=C["ink"], fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    return fig

def plot_daily_profiles(df, cols=("buy_price", "solar_kw", "load_kw"), labels=None):
    """Small multiples: mean over hour of day with a 10-90 % band, one panel per column.

    Shows the daily structure the agent has to learn (and what a snapshot hides).
    """
    labels = labels or {"buy_price": "buy price (EUR/kWh)", "solar_kw": "solar (kW)", "load_kw": "load (kW)"}
    colors = {"buy_price": C["price"], "solar_kw": C["solar"], "load_kw": C["load"]}
    hour = df.index.hour
    fig, axs = plt.subplots(1, len(cols), figsize=(4 * len(cols), 3.2))
    fig.patch.set_facecolor(C["surface"])
    for ax, col in zip(np.atleast_1d(axs), cols):
        grp = df[col].groupby(hour)
        mean, lo, hi = grp.mean(), grp.quantile(0.1), grp.quantile(0.9)
        ax.fill_between(mean.index, lo, hi, color=colors.get(col, C["price"]), alpha=0.18, lw=0)
        ax.plot(mean.index, mean, color=colors.get(col, C["price"]), lw=2)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xlabel("hour of day", color=C["ink2"], fontsize=9)
        _style(ax, labels.get(col, col))
        ax.set_title(labels.get(col, col), color=C["ink"], fontsize=10, loc="left")
        ax.set_ylabel("")
    fig.tight_layout()
    return fig
