import asyncio
from pathlib import Path

import pytest

from core.config import SocialConfig
from data.social import InstagramSource, JsonCache, NotConfigured, RateLimiter, RedditSource, SocialCollector, SocialRequest, TwitterSource


def test_missing_credentials_are_skipped_and_outages_are_isolated(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("X_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("REDDIT_USER_AGENT", raising=False)
    cache = JsonCache(tmp_path)
    social = SocialConfig()

    class Outage:
        name = "reddit"

        async def fetch(self, request):
            raise RuntimeError("429 too many requests")

    collector = SocialCollector(
        [
            TwitterSource(social, cache),
            Outage(),
            InstagramSource(social, cache),
        ]
    )
    request = SocialRequest(["BTC/USDT"], ["$BTC"], [], ["cryptocurrency"], 10)
    result = asyncio.run(collector.collect(request))
    assert result.posts == []
    assert "twitter" in result.skipped
    assert result.failed == ["reddit"]


def test_instagram_stub_returns_nothing(tmp_path: Path):
    source = InstagramSource(SocialConfig(), JsonCache(tmp_path))
    posts = asyncio.run(source.fetch(SocialRequest(["BTC/USDT"], ["$BTC"], [], [], 5)))
    assert posts == []


def test_reddit_and_twitter_classes_raise_when_unconfigured(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("X_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    cache = JsonCache(tmp_path)
    with pytest.raises(NotConfigured):
        asyncio.run(TwitterSource(SocialConfig(retries=1), cache)._fetch(SocialRequest(["BTC/USDT"], ["$BTC"], [], [], 5)))
    with pytest.raises(NotConfigured):
        asyncio.run(RedditSource(SocialConfig(retries=1), cache)._fetch(SocialRequest(["BTC/USDT"], ["$BTC"], [], [], 5)))


def test_rate_limiter_spaces_calls():
    async def scenario():
        limiter = RateLimiter(0.02)
        await limiter.wait()
        await limiter.wait()

    import time

    started = time.monotonic()
    asyncio.run(scenario())
    assert time.monotonic() - started >= 0.02
