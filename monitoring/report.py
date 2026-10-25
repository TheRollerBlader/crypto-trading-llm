"""Markdown performance reports."""

from __future__ import annotations


def performance_markdown(title: str, metrics: dict, notes: list[str]) -> str:
    win_rate = metrics.get("win_rate")
    win_text = "n/a" if win_rate is None else f"{win_rate * 100:.1f}%"
    lines = [
        f"# {title}",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Total return | {_pct(metrics.get('total_return_pct'))} |",
        f"| Buy and hold return | {_pct(metrics.get('buy_hold_return_pct'))} |",
        f"| Sharpe (daily, 365-day) | {_num(metrics.get('sharpe'))} |",
        f"| Max drawdown | {_pct(metrics.get('max_drawdown_pct'))} |",
        f"| Win rate | {win_text} |",
        f"| Closed trades | {metrics.get('num_trades', 0)} |",
        f"| Fees paid | {_num(metrics.get('fees_paid'))} |",
        f"| Ending equity | {_num(metrics.get('ending_equity'))} |",
        "",
    ]
    if notes:
        lines.append("## Notes")
        lines.append("")
        lines.extend(f"- {note}" for note in notes)
        lines.append("")
    return "\n".join(lines)


def _pct(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}%"


def _num(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"
