"""Draw the README figures from the stored backtest. Nothing here is simulated.

Inputs are data/cache hourly candles and logs/agent.sqlite. Re-run after a backtest:

    python scripts/render_visuals.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitoring.store import Store

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
FEE = 0.001
SLIP = 0.0005

plt.rcParams.update(
    {
        "figure.facecolor": "#0b1020",
        "savefig.facecolor": "#0b1020",
        "axes.facecolor": "#0b1020",
        "axes.edgecolor": "#334155",
        "axes.labelcolor": "#e2e8f0",
        "axes.titlecolor": "#67e8f9",
        "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8",
        "text.color": "#e2e8f0",
        "grid.color": "#1e293b",
        "font.size": 11,
    }
)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    equity = pd.DataFrame(Store(f"sqlite:///{ROOT / 'logs' / 'agent.sqlite'}").equity_points(), columns=["ts", "equity"])
    equity["ts"] = pd.to_datetime(equity["ts"], utc=True)
    frames = []
    for symbol, name in (("BTC/USDT", "btc"), ("ETH/USDT", "eth")):
        path = ROOT / "data" / "cache" / f"{symbol.replace('/', '_')}_1h.csv"
        frame = pd.read_csv(path, parse_dates=["timestamp"])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.rename(columns={"open": f"{name}_open", "close": f"{name}_close"})
        frames.append(frame[["timestamp", f"{name}_open", f"{name}_close"]])
    prices = frames[0].merge(frames[1], on="timestamp", how="inner").sort_values("timestamp")
    merged = pd.merge_asof(equity.sort_values("ts"), prices, left_on="ts", right_on="timestamp", direction="backward")
    gap = (merged["ts"] - merged["timestamp"]).dt.total_seconds().max()
    if merged[["btc_close", "eth_close"]].isna().any().any() or gap > 2 * 3600:
        raise SystemExit("Equity timestamps do not line up with the candle cache.")
    start = merged.iloc[0]
    btc_qty = 5_000 / (float(start.btc_open) * (1 + SLIP) * (1 + FEE))
    eth_qty = 5_000 / (float(start.eth_open) * (1 + SLIP) * (1 + FEE))
    merged["bh"] = btc_qty * merged["btc_close"] + eth_qty * merged["eth_close"]
    # The published buy-and-hold figure charges the exit fee and slippage once, at the end.
    last = len(merged) - 1
    merged.loc[merged.index[last], "bh"] = (
        btc_qty * float(merged.iloc[last].btc_close) * (1 - SLIP) * (1 - FEE)
        + eth_qty * float(merged.iloc[last].eth_close) * (1 - SLIP) * (1 - FEE)
    )
    merged["strategy_ret"] = merged["equity"] / merged["equity"].iloc[0] * 100 - 100
    merged["bh_ret"] = merged["bh"] / 10_000 * 100 - 100
    merged["btc_ret"] = merged["btc_close"] / float(start.btc_open) * 100 - 100
    merged["eth_ret"] = merged["eth_close"] / float(start.eth_open) * 100 - 100
    peak = merged["equity"].cummax()
    merged["drawdown"] = (merged["equity"] - peak) / peak * 100
    bh_peak = merged["bh"].cummax()
    merged["bh_drawdown"] = (merged["bh"] - bh_peak) / bh_peak * 100
    return merged, _trades()


def _trades() -> pd.DataFrame:
    rows = []
    for trade in Store(f"sqlite:///{ROOT / 'logs' / 'agent.sqlite'}").closed_trades():
        held = (trade.closed_at - trade.opened_at).total_seconds() / 3600
        rows.append(
            {
                "symbol": trade.symbol,
                "held_hours": held,
                "return_pct": (trade.exit_price / trade.entry_price - 1) * 100,
                "pnl": trade.realized_pnl,
                "opened": trade.opened_at,
            }
        )
    return pd.DataFrame(rows)


def _style_3d(ax) -> None:
    pane = (0.06, 0.09, 0.16, 0.55)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(pane)
        axis.pane.set_edgecolor("#334155")
        axis.line.set_color("#334155")


def _grab(fig) -> Image.Image:
    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(buffer.copy())


def _save_gif(frames: list[Image.Image], path: Path, duration: int = 80) -> None:
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True,
        disposal=2,
    )


def equity_gif(data: pd.DataFrame) -> None:
    steps = np.linspace(8, len(data) - 1, 42, dtype=int)
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    frames = []
    for end in steps:
        ax.clear()
        view = data.iloc[: end + 1]
        ax.plot(view["ts"], view["bh_ret"], color="#94a3b8", linewidth=1.3, label="Buy and hold")
        ax.plot(view["ts"], view["strategy_ret"], color="#22d3ee", linewidth=2.0, label="MOM-03 baseline")
        ax.axhline(0, color="#334155", linewidth=0.8)
        ax.scatter(view["ts"].iloc[-1], view["strategy_ret"].iloc[-1], color="#fbbf24", s=28, zorder=3)
        ax.set_title("Equity replay, 23 Jun 2026 to the latest bar")
        ax.set_ylabel("Return from 10,000 (%)")
        ax.grid(True, linewidth=0.4)
        ax.legend(loc="upper left", frameon=False)
        ax.set_xlim(data["ts"].iloc[0], data["ts"].iloc[-1])
        low = min(data["strategy_ret"].min(), data["bh_ret"].min()) - 2
        high = max(data["strategy_ret"].max(), data["bh_ret"].max()) + 2
        ax.set_ylim(low, high)
        fig.tight_layout()
        frames.append(_grab(fig))
    plt.close(fig)
    _save_gif(frames, ASSETS / "equity_replay.gif", duration=90)


def path_3d(data: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(10.2, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0b1020")
    steps = np.linspace(12, len(data) - 1, 28, dtype=int)
    frames = []
    x_all, y_all, z_all = data["btc_ret"], data["eth_ret"], data["strategy_ret"]
    for index, end in enumerate(steps):
        ax.cla()
        ax.set_facecolor("#0b1020")
        _style_3d(ax)
        view = data.iloc[: end + 1]
        ax.plot(view["btc_ret"], view["eth_ret"], view["strategy_ret"], color="#22d3ee", linewidth=1.8)
        ax.scatter(view["btc_ret"].iloc[-1], view["eth_ret"].iloc[-1], view["strategy_ret"].iloc[-1], color="#fbbf24", s=30)
        ax.set_xlim(x_all.min() - 2, x_all.max() + 2)
        ax.set_ylim(y_all.min() - 2, y_all.max() + 2)
        ax.set_zlim(z_all.min() - 1, z_all.max() + 1)
        ax.set_xlabel("BTC return %", labelpad=8)
        ax.set_ylabel("ETH return %", labelpad=8)
        ax.set_zlabel("Strategy return %", labelpad=8)
        ax.set_title("3D path: the two markets versus the baseline")
        ax.view_init(elev=22, azim=28 + index * 4)
        fig.tight_layout()
        frames.append(_grab(fig))
    fig.savefig(ASSETS / "path_3d.png", dpi=140)
    plt.close(fig)
    _save_gif(frames, ASSETS / "path_3d.gif", duration=110)


def trades_3d(trades: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(10.2, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0b1020")
    _style_3d(ax)
    for symbol, color in (("BTC/USDT", "#f59e0b"), ("ETH/USDT", "#22d3ee")):
        part = trades[trades["symbol"] == symbol]
        ax.scatter(part["held_hours"], part["return_pct"], part["pnl"], color=color, s=36, depthshade=True, label=symbol)
    ax.set_xlabel("Hours held", labelpad=8)
    ax.set_ylabel("Price change %", labelpad=8)
    ax.set_zlabel("Realized PnL", labelpad=8)
    ax.set_title("Closed trades in 3D")
    ax.legend(loc="upper left", frameon=False)
    ax.view_init(elev=24, azim=-58)
    fig.tight_layout()
    fig.savefig(ASSETS / "trades_3d.png", dpi=140)
    # A short orbit so the same cloud can be read from more than one side.
    frames = []
    for azim in range(-70, 50, 8):
        ax.view_init(elev=24, azim=azim)
        fig.canvas.draw()
        frames.append(_grab(fig))
    plt.close(fig)
    _save_gif(frames, ASSETS / "trades_3d.gif", duration=120)


def performance(data: pd.DataFrame, trades: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 7.2), sharex=False)
    axes[0].plot(data["ts"], data["bh_ret"], color="#94a3b8", linewidth=1.2, label="Buy and hold")
    axes[0].plot(data["ts"], data["strategy_ret"], color="#22d3ee", linewidth=1.8, label="MOM-03 baseline")
    axes[0].axhline(0, color="#334155", linewidth=0.8)
    axes[0].set_title("Cumulative return")
    axes[0].set_ylabel("Percent")
    axes[0].grid(True, linewidth=0.4)
    axes[0].legend(frameon=False)
    axes[1].fill_between(data["ts"], data["drawdown"], 0, color="#22d3ee", alpha=0.35, label="Baseline drawdown")
    axes[1].plot(data["ts"], data["bh_drawdown"], color="#94a3b8", linewidth=1.0, label="Buy-and-hold drawdown")
    axes[1].set_title("Drawdown from the running peak")
    axes[1].set_ylabel("Percent")
    axes[1].grid(True, linewidth=0.4)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "performance.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    colors = ["#34d399" if value > 0 else "#fb7185" for value in trades["pnl"]]
    ax.bar(range(len(trades)), trades["pnl"], color=colors, width=0.8)
    ax.axhline(0, color="#334155", linewidth=0.8)
    ax.set_title("Realized PnL of each closed trade, in order")
    ax.set_xlabel("Trade number")
    ax.set_ylabel("Quote currency")
    ax.grid(True, axis="y", linewidth=0.4)
    fig.tight_layout()
    fig.savefig(ASSETS / "trade_pnl.png", dpi=140)
    plt.close(fig)


def summarize(data: pd.DataFrame, trades: pd.DataFrame) -> None:
    daily = data.set_index("ts")["equity"].resample("1D").last().pct_change().dropna()
    sharpe = float(daily.mean() / daily.std() * np.sqrt(365)) if daily.std() else 0.0
    bh_daily = data.set_index("ts")["bh"].resample("1D").last().pct_change().dropna()
    bh_sharpe = float(bh_daily.mean() / bh_daily.std() * np.sqrt(365)) if bh_daily.std() else 0.0
    wins = int((trades["pnl"] > 0).sum())
    print(f"bars {len(data)}")
    print(f"strategy_return {data['strategy_ret'].iloc[-1]:.2f}")
    print(f"bh_return {data['bh_ret'].iloc[-1]:.2f}")
    print(f"strategy_dd {data['drawdown'].min():.2f}")
    print(f"bh_dd {data['bh_drawdown'].min():.2f}")
    print(f"sharpe {sharpe:.2f}")
    print(f"bh_sharpe {bh_sharpe:.2f}")
    print(f"wins {wins} of {len(trades)}")


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    data, trades = load()
    equity_gif(data)
    path_3d(data)
    trades_3d(trades)
    performance(data, trades)
    summarize(data, trades)
    for name in (
        "3d_preview.gif",
        "3d_simulation.mp4",
        "3d_surface.png",
        "performance_metrics.png",
        "results_heatmap.png",
    ):
        stale = ASSETS / name
        if stale.exists():
            stale.unlink()
            print(f"removed {name}")


if __name__ == "__main__":
    main()
