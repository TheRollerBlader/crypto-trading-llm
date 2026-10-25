"""Streamlit view of the paper-trading log.

Run: streamlit run monitoring/dashboard.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.config import project_root
from monitoring.store import Store


def load_stats(database: str) -> dict:
    path = Path(database)
    if not path.is_absolute():
        path = project_root() / path
    url = f"sqlite:///{path}"
    return Store(url).stats()


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Paper trading monitor", layout="wide")
    st.title("Paper trading monitor")
    st.caption("Decisions are proposed by the model. Orders exist only after the risk manager and the paper broker accept them.")
    database = st.text_input("SQLite log", "logs/agent.sqlite")
    path = Path(database)
    if not path.is_absolute():
        path = project_root() / path
    if not path.exists():
        st.info("No database yet. Run a paper or backtest session first. This page updates when you rerun it.")
        return
    stats = load_stats(str(path))
    equity = stats.get("equity")
    win_rate = stats.get("win_rate")
    columns = st.columns(4)
    columns[0].metric("Equity", "n/a" if equity is None else f"{equity:,.2f}")
    columns[1].metric("Realized PnL", f"{stats['realized_pnl']:,.2f}")
    columns[2].metric("Win rate", "n/a" if win_rate is None else f"{win_rate * 100:.1f}%")
    columns[3].metric("Closed trades", str(stats["closed_trades"]))
    curve = stats["equity_curve"]
    if curve:
        frame = pd.DataFrame(curve, columns=["time", "equity"])
        st.subheader("Equity curve")
        st.line_chart(frame.set_index("time"))
    else:
        st.subheader("Equity curve")
        st.write("No equity points recorded.")
    st.subheader("Open positions")
    positions = stats["positions"]
    st.dataframe(pd.DataFrame(positions) if positions else pd.DataFrame(columns=["symbol", "qty", "entry_price"]), use_container_width=True)
    st.subheader("Closed trades")
    trades = stats["trades"]
    st.dataframe(pd.DataFrame(trades) if trades else pd.DataFrame(columns=["symbol", "realized_pnl", "reason"]), use_container_width=True)
    assets = project_root() / "assets"
    clips = [
        ("Equity replay", assets / "equity_replay.gif"),
        ("3D market path", assets / "path_3d.gif"),
        ("3D trades", assets / "trades_3d.gif"),
    ]
    available = [(title, path) for title, path in clips if path.exists()]
    if available:
        st.subheader("Replays")
        st.caption("These are rendered from the stored backtest. Regenerate them with python scripts/render_visuals.py.")
        for title, path in available:
            st.markdown(f"**{title}**")
            st.image(str(path))


if __name__ == "__main__":
    main()
