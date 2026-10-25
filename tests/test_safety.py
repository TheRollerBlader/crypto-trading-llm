import ccxt
import pytest

from core.safety import (
    CONFIRMATION_PHRASE,
    RuntimeOptions,
    SafetyError,
    active_trading_urls,
    authorize,
    trading_urls_are_sandbox,
)
from tests.helpers import config


def test_default_paper_config_does_not_request_live(monkeypatch):
    monkeypatch.delenv("LIVE_TRADING", raising=False)
    monkeypatch.setenv("LIVE_TRADING", "false")
    label = authorize(config(), RuntimeOptions(mode="paper"), check_urls=False)
    assert label == "paper"


def test_partial_live_switch_refuses_to_start(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING", "true")
    with pytest.raises(SafetyError, match="Refusing to start"):
        authorize(config(), RuntimeOptions(mode="paper"), check_urls=False)


def test_live_mode_rejects_a_wrong_phrase_and_sandbox_urls(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING", "true")
    monkeypatch.setenv("LIVE_API_KEY", "live-key-value")
    monkeypatch.setenv("LIVE_API_SECRET", "live-secret-value")
    monkeypatch.setenv("PAPER_API_KEY", "paper-key-value")
    cfg = config()
    cfg.live_trading = True
    cfg.exchange.sandbox = False
    options = RuntimeOptions(mode="paper", enable_live=True, interactive=True, confirmation="yes")
    with pytest.raises(SafetyError):
        authorize(cfg, options, ["https://api.binance.com/api/v3"], check_urls=True)
    options.confirmation = CONFIRMATION_PHRASE
    with pytest.raises(SafetyError):
        authorize(cfg, options, ["https://testnet.binance.vision/api/v3"], check_urls=True)
    assert authorize(cfg, options, ["https://api.binance.com/api/v3"], check_urls=True) == "live"


def test_paper_mode_rejects_production_urls_and_accepts_testnet(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING", "false")
    cfg = config()
    with pytest.raises(SafetyError, match="production"):
        authorize(cfg, RuntimeOptions(mode="paper"), ["https://api.binance.com/api/v3"], check_urls=True)
    assert (
        authorize(cfg, RuntimeOptions(mode="paper"), ["https://testnet.binance.vision/api/v3"], check_urls=True)
        == "paper"
    )


def test_simulated_exchange_cannot_be_marked_live(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING", "false")
    cfg = config()
    cfg.exchange.id = "simulated"
    assert authorize(cfg, RuntimeOptions(mode="paper"), check_urls=False) == "paper-simulated"
    cfg.live_trading = True
    monkeypatch.setenv("LIVE_TRADING", "true")
    with pytest.raises(SafetyError):
        authorize(cfg, RuntimeOptions(mode="paper", enable_live=True), check_urls=False)


def test_ccxt_binance_sandbox_rewrites_the_trading_hosts():
    exchange = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    assert not trading_urls_are_sandbox(active_trading_urls(exchange.urls))
    exchange.set_sandbox_mode(True)
    urls = active_trading_urls(exchange.urls)
    assert trading_urls_are_sandbox(urls)
    assert all("testnet.binance.vision" in url for url in urls)
