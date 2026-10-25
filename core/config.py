"""Load config.yaml and overlay non-secret environment settings."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


class ExchangeConfig(BaseModel):
    id: str = "binance"
    sandbox: bool = True
    market_type: str = "spot"
    quote_currency: str = "USDT"
    initial_quote: float = 10_000.0


class RiskConfig(BaseModel):
    max_position_size_pct: float = 10
    max_total_exposure_pct: float = 40
    max_daily_loss_pct: float = 3
    max_drawdown_pct: float = 15
    max_trades_per_day: int = 8
    min_stop_distance_pct: float = 0.4
    max_stop_distance_pct: float = 8
    default_stop_atr_mult: float = 1.8
    default_take_profit_atr_mult: float = 3.0
    cooldown_after_loss_minutes: int = 60
    slippage_bps: float = 5
    fee_bps: float = 10
    min_confidence: float = 0.55
    min_trade_pct: float = 1
    allow_short: bool = False


class LLMConfig(BaseModel):
    model: str = "claude-sonnet-4-5"
    max_tokens: int = 2500
    max_retries: int = 2
    timeout_seconds: float = 45
    prompt_cache_min_chars: int = 4000


class SocialConfig(BaseModel):
    enabled_sources: list[str] = Field(default_factory=lambda: ["twitter", "reddit", "instagram"])
    cashtags: list[str] = Field(default_factory=lambda: ["$BTC", "$ETH"])
    accounts: list[str] = Field(default_factory=list)
    subreddits: list[str] = Field(default_factory=lambda: ["cryptocurrency", "bitcoin", "ethereum"])
    cache_ttl_seconds: int = 300
    max_posts_per_source: int = 25
    request_timeout_seconds: float = 20
    min_interval_seconds: float = 2
    retries: int = 2
    sentiment_mode: str = "lexicon"
    spike_post_count: int = 25
    prompt_post_limit: int = 12


class StrategyConfig(BaseModel):
    directory: str = "strategy_context"
    mode: str = "auto"
    full_max_chars: int = 120_000
    chunk_chars: int = 2500
    top_k: int = 6


class MonitoringConfig(BaseModel):
    database: str = "logs/agent.sqlite"
    jsonl_path: str = "logs/cycles.jsonl"
    log_path: str = "logs/agent.jsonl"
    kill_switch_file: str = "KILL_SWITCH"


class BacktestConfig(BaseModel):
    timeframe: str = "1h"
    warmup_bars: int = 1440
    decision: str = "baseline"


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    live_trading: bool = False
    symbols: list[str] = Field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    cycle_interval_minutes: int = 15
    primary_timeframe: str = "1h"
    timeframes: list[str] = Field(default_factory=lambda: ["15m", "1h", "4h", "1d"])
    stale_candle_multiple: float = 2.0
    candle_limit: int = 300
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    social: SocialConfig = Field(default_factory=SocialConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)


def project_root() -> Path:
    override = os.environ.get("TRADING_BOT_ROOT")
    if override:
        return Path(override).resolve()
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "config.yaml").exists():
            return candidate
    return here


def load_config(path: Path | None = None) -> AppConfig:
    root = project_root()
    load_dotenv(root / ".env", override=False)
    config_path = path or (root / "config.yaml")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = AppConfig.model_validate(raw)
    model = os.getenv("ANTHROPIC_MODEL")
    if model:
        config.llm.model = model
    return config


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path
