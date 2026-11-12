import pytest
from data.tick_candle_aggregator import TickCandleAggregator, TradeTick


class TestTickCandleAggregator:
    def test_single_minute_aggregation(self):
        agg = TickCandleAggregator(symbol="BTC-USD", bucket_size_sec=60)

        # Aligned minute 1 base
        base_ms = (1742000000000 // 60000) * 60000

        agg.ingest_tick(TradeTick(timestamp_ms=base_ms + 1000, symbol="BTC-USD", price=65000.0, volume=1.0, side="buy"))
        agg.ingest_tick(TradeTick(timestamp_ms=base_ms + 5000, symbol="BTC-USD", price=65200.0, volume=2.0, side="buy"))
        agg.ingest_tick(TradeTick(timestamp_ms=base_ms + 15000, symbol="BTC-USD", price=64900.0, volume=0.5, side="sell"))
        agg.ingest_tick(TradeTick(timestamp_ms=base_ms + 45000, symbol="BTC-USD", price=65100.0, volume=1.5, side="buy"))

        candle = agg.get_latest_candle()
        assert candle is not None
        assert candle.open == 65000.0
        assert candle.high == 65200.0
        assert candle.low == 64900.0
        assert candle.close == 65100.0
        assert candle.volume == 5.0
        assert candle.tick_count == 4
        assert candle.is_sealed is False

        # Expected VWAP: (65000*1 + 65200*2 + 64900*0.5 + 65100*1.5) / 5.0
        expected_vwap = (65000 + 130400 + 32450 + 97650) / 5.0  # 325500 / 5 = 65100.0
        assert candle.vwap == pytest.approx(expected_vwap, abs=0.1)

    def test_minute_boundary_sealing(self):
        sealed_list = []
        agg = TickCandleAggregator(
            symbol="ETH-USD",
            bucket_size_sec=60,
            on_candle_sealed=lambda c: sealed_list.append(c),
        )

        base_ms = (1742000000000 // 60000) * 60000
        # Tick in minute 1
        agg.ingest_tick(TradeTick(timestamp_ms=base_ms + 10000, symbol="ETH-USD", price=3400.0, volume=10.0, side="buy"))

        # Tick in minute 2 (triggers sealing of minute 1)
        sealed = agg.ingest_tick(TradeTick(timestamp_ms=base_ms + 62000, symbol="ETH-USD", price=3420.0, volume=5.0, side="buy"))

        assert sealed is not None
        assert sealed.symbol == "ETH-USD"
        assert sealed.is_sealed is True
        assert sealed.open == 3400.0
        assert sealed.close == 3400.0
        assert len(sealed_list) == 1

        # Current candle is now minute 2
        current = agg.get_latest_candle()
        assert current is not None
        assert current.bucket_start_ms == base_ms + 60000
        assert current.open == 3420.0
        assert current.is_sealed is False

    def test_ignores_foreign_symbols(self):
        agg = TickCandleAggregator(symbol="SOL-USD")
        agg.ingest_tick(TradeTick(timestamp_ms=1000, symbol="BTC-USD", price=65000, volume=1, side="buy"))
        assert agg.get_latest_candle() is None
