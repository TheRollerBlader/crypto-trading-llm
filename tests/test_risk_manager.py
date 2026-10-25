from datetime import timedelta

from core.models import Action, Position
from core.timeframes import utcnow
from risk.manager import RiskManager
from tests.helpers import config, decision, portfolio, snapshot


def _review(cfg, decisions, port, markets=None, now=None):
    markets = markets or [snapshot()]
    return RiskManager(cfg).review(decisions, port, markets, now or utcnow())


def test_position_size_is_resized_not_blindly_accepted():
    cfg = config()
    verdict = _review(cfg, [decision(size=50)], portfolio())[0]
    assert verdict.approved
    assert verdict.action is Action.BUY
    assert verdict.size_pct == pytest_approx_size(cfg.risk.max_position_size_pct)
    assert any("max position" in reason for reason in verdict.reasons)


def pytest_approx_size(value):
    return value


def test_total_exposure_room_resizes_the_order():
    cfg = config()
    cfg.risk.max_position_size_pct = 20
    cfg.risk.max_total_exposure_pct = 40
    held = Position(symbol="ETH/USDT", qty=1, entry_price=3500, opened_at=utcnow(), stop_loss=3400)
    port = portfolio(positions=[held])
    markets = [snapshot("BTC/USDT", 100, atr=2), snapshot("ETH/USDT", 3500, atr=20)]
    verdicts = _review(cfg, [decision("BTC/USDT", size=20, stop=95)], port, markets)
    btc = verdicts[0]
    assert btc.approved
    assert btc.size_pct == pytest_approx_size(5)


def test_daily_loss_circuit_breaker_blocks_entries_and_allows_close():
    cfg = config()
    port = portfolio(equity=9600, positions=[Position(symbol="BTC/USDT", qty=0.1, entry_price=100, opened_at=utcnow(), stop_loss=90)])
    verdicts = _review(
        cfg,
        [decision(size=5, stop=95), decision(action=Action.CLOSE, stop=90)],
        port,
    )
    assert verdicts[0].action is Action.HOLD
    assert verdicts[0].halt
    assert verdicts[0].halt_reason == "max_daily_loss"
    assert verdicts[1].approved
    assert verdicts[1].action is Action.CLOSE


def test_drawdown_circuit_breaker():
    cfg = config()
    port = portfolio(equity=8000, peak_equity=10_000, day_start_equity=8000)
    verdict = _review(cfg, [decision()], port)[0]
    assert verdict.halt_reason == "max_drawdown"
    assert verdict.action is Action.HOLD
    assert not verdict.approved


def test_max_trades_per_day():
    cfg = config()
    port = portfolio(trades_today=cfg.risk.max_trades_per_day)
    verdict = _review(cfg, [decision()], port)[0]
    assert verdict.action is Action.HOLD
    assert any("max trades" in reason for reason in verdict.reasons)


def test_missing_stop_is_attached_and_missing_price_is_rejected():
    cfg = config()
    attached = _review(cfg, [decision(stop=None)], portfolio(), [snapshot(price=100, atr=2)])[0]
    assert attached.approved
    assert attached.stop_loss == 100 - 1.8 * 2
    assert any("stop attached" in reason for reason in attached.reasons)
    naked = snapshot(price=100, atr=None)
    naked.primary.indicators.atr = None
    rejected = _review(cfg, [decision(stop=None)], portfolio(), [naked])[0]
    assert rejected.action is Action.HOLD
    assert not rejected.approved


def test_wrong_side_stop_is_rejected():
    cfg = config()
    verdict = _review(cfg, [decision(stop=120)], portfolio(), [snapshot(price=100)])[0]
    assert verdict.action is Action.HOLD
    assert any("wrong side" in reason for reason in verdict.reasons)


def test_cooldown_after_loss_blocks_entries():
    cfg = config()
    port = portfolio(last_loss_at=utcnow() - timedelta(minutes=10))
    blocked = _review(cfg, [decision()], port)[0]
    assert blocked.action is Action.HOLD
    assert any("cooldown" in reason for reason in blocked.reasons)
    port = portfolio(
        last_loss_at=utcnow() - timedelta(minutes=10),
        positions=[Position(symbol="BTC/USDT", qty=1, entry_price=100, opened_at=utcnow(), stop_loss=90)],
    )
    closing = _review(cfg, [decision(action=Action.CLOSE, stop=90)], port)[0]
    assert closing.approved
    assert closing.action is Action.CLOSE


def test_stop_tighter_than_fees_and_slippage_is_rejected():
    cfg = config()
    cfg.risk.min_stop_distance_pct = 0
    cfg.risk.fee_bps = 50
    cfg.risk.slippage_bps = 50
    verdict = _review(cfg, [decision(stop=99.9)], portfolio(), [snapshot(price=100, atr=1)])[0]
    assert verdict.action is Action.HOLD
    assert any("fees and slippage" in reason for reason in verdict.reasons)


def test_stop_wider_than_the_maximum_is_tightened():
    cfg = config()
    verdict = _review(cfg, [decision(stop=50)], portfolio(), [snapshot(price=100, atr=1)])[0]
    assert verdict.approved
    assert verdict.stop_loss == 100 * (1 - cfg.risk.max_stop_distance_pct / 100)


def test_low_confidence_and_disabled_short():
    cfg = config()
    quiet = _review(cfg, [decision(confidence=0.1)], portfolio())[0]
    assert quiet.action is Action.HOLD
    assert any("confidence" in reason for reason in quiet.reasons)
    short = _review(cfg, [decision(action=Action.SELL, size=5, stop=105)], portfolio())[0]
    assert short.action is Action.HOLD
    assert any("shorting" in reason for reason in short.reasons)


def test_two_buys_respect_the_same_exposure_budget():
    cfg = config()
    cfg.risk.max_position_size_pct = 10
    cfg.risk.max_total_exposure_pct = 15
    markets = [snapshot("BTC/USDT", 100), snapshot("ETH/USDT", 50)]
    verdicts = _review(
        cfg,
        [decision("BTC/USDT", size=10, stop=95), decision("ETH/USDT", size=10, stop=47)],
        portfolio(),
        markets,
    )
    assert verdicts[0].approved and verdicts[0].size_pct == 10
    assert verdicts[1].approved and verdicts[1].size_pct == 5
