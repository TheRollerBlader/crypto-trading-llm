import pytest
from pydantic import ValidationError

from core.models import Action, TradeDecision, hold_decision
from core.redact import redact


def test_valid_decision_and_hold_helper():
    decision = TradeDecision(
        symbol="ETH/USDT",
        action="buy",
        size_pct=4,
        entry_type="market",
        stop_loss=90,
        take_profit=120,
        confidence=0.7,
        reasoning="trend intact",
        strategy_referenced="MOM-03",
        extra_field="ignored",
    )
    assert decision.action is Action.BUY
    held = hold_decision("BTC/USDT", "no edge")
    assert held.action is Action.HOLD
    assert held.size_pct == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"action": "MAYBE"},
        {"size_pct": 140},
        {"confidence": 1.4},
        {"reasoning": ""},
        {"strategy_referenced": ""},
        {"size_pct": float("nan")},
    ],
)
def test_invalid_decisions_are_rejected(payload):
    base = dict(
        symbol="BTC/USDT",
        action="BUY",
        size_pct=5,
        entry_type="MARKET",
        stop_loss=90,
        take_profit=110,
        confidence=0.6,
        reasoning="because",
        strategy_referenced="MOM-03",
    )
    base.update(payload)
    with pytest.raises(ValidationError):
        TradeDecision.model_validate(base)


def test_buy_requires_size_and_limit_requires_price():
    with pytest.raises(ValidationError):
        TradeDecision(
            symbol="BTC/USDT",
            action="BUY",
            size_pct=0,
            entry_type="MARKET",
            confidence=0.6,
            reasoning="empty",
            strategy_referenced="MOM-03",
        )
    with pytest.raises(ValidationError):
        TradeDecision(
            symbol="BTC/USDT",
            action="BUY",
            size_pct=5,
            entry_type="LIMIT",
            confidence=0.6,
            reasoning="missing limit",
            strategy_referenced="MOM-03",
        )


def test_secrets_are_redacted():
    text = "request failed api_key=SUPERSECRET123456 and sk-ant-abc_def"
    cleaned = redact(text)
    assert "SUPERSECRET123456" not in cleaned
    assert "sk-ant-abc" not in cleaned
    assert "[REDACTED]" in cleaned
