"""Live trading cannot start unless every independent gate is open.

Paper mode checks the endpoints the order client will actually call. Marketing
and backup URLs are ignored. A read-only historical downloader is a separate
object and is not an order client.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

from core.config import AppConfig

CONFIRMATION_PHRASE = "ENABLE LIVE TRADING"

PRODUCTION_HOSTS = {
    "api.binance.com",
    "api.binance.us",
    "api.binance.me",
    "fapi.binance.com",
    "dapi.binance.com",
    "eapi.binance.com",
    "papi.binance.com",
    "api.alpaca.markets",
    "api.kraken.com",
    "futures.kraken.com",
}

SANDBOX_MARKERS = ("testnet", "sandbox", "paper-api", "demo")


class SafetyError(RuntimeError):
    """Startup or runtime condition that must stop the process."""


@dataclass
class RuntimeOptions:
    mode: str = "paper"
    dry_run: bool = False
    symbols: list[str] | None = None
    cycles: int | None = None
    decision: str = "llm"
    enable_live: bool = False
    interactive: bool = False
    confirmation: str | None = None
    exchange_id: str | None = None


def env_flag(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() == "true"


def paper_credentials() -> tuple[str, str]:
    return os.getenv("PAPER_API_KEY", "").strip(), os.getenv("PAPER_API_SECRET", "").strip()


def live_credentials() -> tuple[str, str]:
    return os.getenv("LIVE_API_KEY", "").strip(), os.getenv("LIVE_API_SECRET", "").strip()


def anthropic_api_key() -> str:
    return os.getenv("ANTHROPIC_API_KEY", "").strip()


def hostname(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def is_production_host(url: str) -> bool:
    return hostname(url) in PRODUCTION_HOSTS


def is_sandbox_host(url: str) -> bool:
    host = hostname(url)
    return any(marker in host for marker in SANDBOX_MARKERS)


def active_trading_urls(urls: dict) -> list[str]:
    """Return the public and private REST hosts ccxt will trade through."""

    api = urls.get("api", {}) if isinstance(urls, dict) else {}
    found: list[str] = []
    if isinstance(api, dict):
        for key in ("public", "private"):
            value = api.get(key)
            if isinstance(value, str):
                found.append(value)
    return found


def trading_urls_are_sandbox(urls: list[str]) -> bool:
    if not urls:
        return False
    return all(is_sandbox_host(url) and not is_production_host(url) for url in urls)


def trading_urls_are_production(urls: list[str]) -> bool:
    if not urls:
        return False
    return all(is_production_host(url) and not is_sandbox_host(url) for url in urls)


def _any_live_signal(config: AppConfig, options: RuntimeOptions) -> bool:
    return bool(config.live_trading or env_flag("LIVE_TRADING") or options.enable_live)


def _all_live_flags(config: AppConfig, options: RuntimeOptions) -> bool:
    return bool(config.live_trading and env_flag("LIVE_TRADING") and options.enable_live)


def authorize(
    config: AppConfig,
    options: RuntimeOptions,
    trading_urls: list[str] | None = None,
    check_urls: bool = True,
) -> str:
    """Return 'paper', 'paper-simulated', or 'live'. Raise SafetyError otherwise.

    Call this before create_order. For ccxt, call it again with the real URLs
    after set_sandbox_mode and close the client if this raises.
    """

    exchange_id = config.exchange.id.lower()
    if exchange_id == "simulated":
        if _any_live_signal(config, options):
            raise SafetyError("The simulated exchange cannot be used for live trading.")
        if not config.exchange.sandbox:
            raise SafetyError("The simulated exchange must keep sandbox: true.")
        return "paper-simulated"

    if options.mode == "backtest":
        if _any_live_signal(config, options):
            raise SafetyError("Backtests cannot run with any live-trading switch set.")
        return "backtest"

    if not _all_live_flags(config, options):
        if _any_live_signal(config, options):
            raise SafetyError(
                "Refusing to start. A live-trading switch is on, but live mode needs "
                "all of: live_trading: true in config.yaml, LIVE_TRADING=true, "
                "--enable-live, and the confirmation phrase. Set every switch off "
                "to run paper mode."
            )
        if not config.exchange.sandbox:
            raise SafetyError("Paper mode requires exchange.sandbox: true.")
        if check_urls:
            if not trading_urls_are_sandbox(trading_urls or []):
                raise SafetyError(
                    "Paper mode detected a production trading endpoint and will not start."
                )
        return "paper"

    if options.mode != "paper":
        raise SafetyError("Live trading cannot be combined with this mode.")
    if config.exchange.sandbox:
        raise SafetyError("Live mode requires exchange.sandbox: false.")
    live_key, live_secret = live_credentials()
    if not live_key or not live_secret:
        raise SafetyError("Live mode requires LIVE_API_KEY and LIVE_API_SECRET.")
    paper_key, _paper_secret = paper_credentials()
    if paper_key and live_key == paper_key:
        raise SafetyError("LIVE_API_KEY must be different from PAPER_API_KEY.")
    if check_urls and not trading_urls_are_production(trading_urls or []):
        raise SafetyError("Live mode is not pointed at a production trading endpoint.")
    if not options.interactive:
        raise SafetyError("Live trading requires an interactive terminal confirmation.")
    if options.confirmation != CONFIRMATION_PHRASE:
        raise SafetyError("Live confirmation phrase did not match. Nothing was started.")
    return "live"


def require_paper_keys(config: AppConfig, options: RuntimeOptions) -> None:
    if config.exchange.id.lower() == "simulated" or options.dry_run:
        return
    key, secret = paper_credentials()
    if not key or not secret:
        raise SafetyError(
            "Paper orders need PAPER_API_KEY and PAPER_API_SECRET from the exchange "
            "sandbox. Use --dry-run, or set exchange.id to simulated, to run without keys."
        )
