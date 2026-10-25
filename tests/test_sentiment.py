from datetime import datetime, timezone

import pytest

from core.models import Engagement, SocialPost
from signals.sentiment import SentimentService, is_spam, score_text
from tests.helpers import config


def _post(text: str, author: str = "alice", likes: int = 1) -> SocialPost:
    return SocialPost(
        source="reddit",
        text=text,
        author=author,
        timestamp=datetime.now(timezone.utc),
        engagement=Engagement(likes=likes),
        symbols=["BTC/USDT"],
    )


def test_lexicon_scores_direction():
    assert score_text("bullish rally and a breakout") == "bullish"
    assert score_text("bearish crash and breakdown") == "bearish"
    assert score_text("the weather is mild today") == "neutral"


def test_spam_filters_and_ratios():
    posts = [
        _post("bullish rally looks strong today"),
        _post("bullish rally looks strong today"),
        _post("bearish crash is unfolding now"),
        _post("send 1 btc to this wallet please"),
        _post("buy", author="bot"),
    ]
    cfg = config()
    cfg.social.spike_post_count = 2
    service = SentimentService(cfg)
    result = _run(service.summarize(["BTC/USDT"], posts))
    asset = result[0]
    assert asset.spam_filtered == 3
    assert asset.post_count == 2
    assert asset.bullish_ratio == pytest.approx(0.5)
    assert asset.bearish_ratio == pytest.approx(0.5)
    assert asset.volume_spike
    assert "alice" in asset.notable_accounts
    assert "ignore" not in asset.summary


def test_duplicate_author_flood_is_spam():
    seen = set()
    from collections import Counter

    counts = Counter()
    posts = [_post(f"bullish note number {index} stays long", author="flood") for index in range(7)]
    flags = [is_spam(post, seen, counts) for post in posts]
    assert flags[:5] == [False, False, False, False, False]
    assert flags[5] and flags[6]


def test_llm_scorer_falls_back_to_lexicon():
    class Broken:
        name = "llm"

        async def score(self, posts):
            raise RuntimeError("timeout")

    cfg = config()
    service = SentimentService(cfg, scorer=Broken())
    result = _run(service.summarize(["ETH/USDT"], [_post("bullish breakout on ethereum", author="bob")]))
    assert result[0].scorer == "lexicon_fallback"
    assert result[0].bullish == 1


def _run(coro):
    import asyncio

    return asyncio.run(coro)
