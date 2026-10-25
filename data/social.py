"""Official-API social collectors.

Twitter uses the X API through tweepy. Reddit uses PRAW. Instagram is a stub:
the official API cannot search cashtags, and this project does not scrape.
A source that is not configured is skipped. A source that errors is logged
and the cycle continues with whatever the other sources returned.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

from core.config import AppConfig, SocialConfig
from core.models import Engagement, SocialPost
from core.redact import redact

logger = logging.getLogger(__name__)


class NotConfigured(RuntimeError):
    """The source has no credentials. This is not an outage."""


class SocialRequest:
    def __init__(self, symbols: list[str], cashtags: list[str], accounts: list[str], subreddits: list[str], limit: int):
        self.symbols = symbols
        self.cashtags = cashtags
        self.accounts = accounts
        self.subreddits = subreddits
        self.limit = limit


class SocialCollectResult:
    def __init__(self, posts: list[SocialPost], failed: list[str], skipped: list[str]):
        self.posts = posts
        self.failed = failed
        self.skipped = skipped


class RateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            delay = self.min_interval - (now - self._last)
            if delay > 0:
                await asyncio.sleep(delay)
            self._last = time.monotonic()


class JsonCache:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.directory / f"{digest}.json"

    def get(self, key: str):
        path = self._path(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("expires", 0) < time.time():
            return None
        return payload.get("value")

    def set(self, key: str, value, ttl: int) -> None:
        path = self._path(key)
        path.write_text(
            json.dumps({"expires": time.time() + ttl, "value": value}),
            encoding="utf-8",
        )


async def with_retries(fn, attempts: int, base_delay: float, retry_on: tuple[type[Exception], ...] = (Exception,)):
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return await fn()
        except NotConfigured:
            raise
        except retry_on as exc:
            last = exc
            if attempt == attempts - 1:
                break
            await asyncio.sleep(base_delay * (2**attempt))
    assert last is not None
    raise last


class SocialSource(ABC):
    name: str

    def __init__(self, config: SocialConfig, cache: JsonCache):
        self.config = config
        self.cache = cache
        self.limiter = RateLimiter(config.min_interval_seconds)

    async def fetch(self, request: SocialRequest) -> list[SocialPost]:
        key = f"{self.name}|{','.join(request.symbols)}|{','.join(request.cashtags)}|{','.join(request.accounts)}"

        async def attempt() -> list[SocialPost]:
            await self.limiter.wait()
            cached = self.cache.get(key)
            if cached is not None:
                return [SocialPost.model_validate(item) for item in cached]
            posts = await self._fetch(request)
            self.cache.set(key, [post.model_dump(mode="json") for post in posts], self.config.cache_ttl_seconds)
            return posts

        return await with_retries(attempt, attempts=max(1, self.config.retries), base_delay=0.05)

    @abstractmethod
    async def _fetch(self, request: SocialRequest) -> list[SocialPost]:
        raise NotImplementedError


class TwitterSource(SocialSource):
    name = "twitter"

    async def _fetch(self, request: SocialRequest) -> list[SocialPost]:
        import os

        token = os.getenv("X_BEARER_TOKEN", "").strip()
        if not token:
            raise NotConfigured("X_BEARER_TOKEN is not set")
        return await asyncio.to_thread(self._search, token, request)

    def _search(self, token: str, request: SocialRequest) -> list[SocialPost]:
        import tweepy

        client = tweepy.Client(bearer_token=token, wait_on_rate_limit=False)
        terms = [tag if tag.startswith("$") else f"${tag}" for tag in request.cashtags]
        query = " OR ".join(terms) if terms else " OR ".join(request.symbols)
        query = f"({query}) -is:retweet lang:en"
        posts: list[SocialPost] = []
        response = client.search_recent_tweets(
            query=query[:512],
            max_results=max(10, min(request.limit, 100)),
            tweet_fields=["created_at", "public_metrics", "author_id"],
            expansions=["author_id"],
            user_fields=["username"],
        )
        users = {}
        includes = getattr(response, "includes", None) or {}
        for user in includes.get("users", []) or []:
            users[str(user.id)] = user.username
        for tweet in response.data or []:
            metrics = tweet.public_metrics or {}
            created = tweet.created_at or datetime.now(timezone.utc)
            posts.append(
                SocialPost(
                    source=self.name,
                    text=tweet.text or "",
                    author=users.get(str(tweet.author_id), str(tweet.author_id)),
                    timestamp=created if created.tzinfo else created.replace(tzinfo=timezone.utc),
                    engagement=Engagement(
                        likes=int(metrics.get("like_count", 0)),
                        reposts=int(metrics.get("retweet_count", 0)),
                        comments=int(metrics.get("reply_count", 0)),
                    ),
                    symbols=_mentioned(tweet.text or "", request.symbols),
                    url=f"https://x.com/i/web/status/{tweet.id}",
                )
            )
        for account in request.accounts[:5]:
            user_query = f"from:{account.lstrip('@')} -is:retweet"
            user_response = client.search_recent_tweets(
                query=user_query,
                max_results=10,
                tweet_fields=["created_at", "public_metrics", "author_id"],
            )
            for tweet in user_response.data or []:
                metrics = tweet.public_metrics or {}
                created = tweet.created_at or datetime.now(timezone.utc)
                posts.append(
                    SocialPost(
                        source=self.name,
                        text=tweet.text or "",
                        author=account.lstrip("@"),
                        timestamp=created if created.tzinfo else created.replace(tzinfo=timezone.utc),
                        engagement=Engagement(
                            likes=int(metrics.get("like_count", 0)),
                            reposts=int(metrics.get("retweet_count", 0)),
                            comments=int(metrics.get("reply_count", 0)),
                        ),
                        symbols=_mentioned(tweet.text or "", request.symbols),
                    )
                )
        return posts[: request.limit]


class RedditSource(SocialSource):
    name = "reddit"

    async def _fetch(self, request: SocialRequest) -> list[SocialPost]:
        import os

        client_id = os.getenv("REDDIT_CLIENT_ID", "").strip()
        secret = os.getenv("REDDIT_CLIENT_SECRET", "").strip()
        agent = os.getenv("REDDIT_USER_AGENT", "").strip()
        if not client_id or not secret or not agent:
            raise NotConfigured("Reddit credentials are not set")
        return await asyncio.to_thread(self._search, client_id, secret, agent, request)

    def _search(self, client_id: str, secret: str, agent: str, request: SocialRequest) -> list[SocialPost]:
        import praw

        reddit = praw.Reddit(client_id=client_id, client_secret=secret, user_agent=agent)
        subreddit = reddit.subreddit("+".join(request.subreddits or ["cryptocurrency"]))
        posts: list[SocialPost] = []
        queries = request.cashtags or request.symbols
        per_query = max(1, request.limit // max(1, len(queries)))
        for query in queries:
            for submission in subreddit.search(query, sort="new", time_filter="day", limit=per_query):
                created = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
                posts.append(
                    SocialPost(
                        source=self.name,
                        text=f"{submission.title}\n{submission.selftext or ''}".strip(),
                        author=str(submission.author) if submission.author else "unknown",
                        timestamp=created,
                        engagement=Engagement(
                            likes=int(getattr(submission, "score", 0) or 0),
                            comments=int(getattr(submission, "num_comments", 0) or 0),
                        ),
                        symbols=_mentioned(submission.title or "", request.symbols),
                        url=f"https://www.reddit.com{submission.permalink}",
                    )
                )
        for account in request.accounts[:5]:
            try:
                redditor = reddit.redditor(account)
                for submission in redditor.submissions.new(limit=5):
                    created = datetime.fromtimestamp(submission.created_utc, tz=timezone.utc)
                    posts.append(
                        SocialPost(
                            source=self.name,
                            text=f"{submission.title}\n{submission.selftext or ''}".strip(),
                            author=account,
                            timestamp=created,
                            engagement=Engagement(likes=int(submission.score or 0), comments=int(submission.num_comments or 0)),
                            symbols=_mentioned(submission.title or "", request.symbols),
                            url=f"https://www.reddit.com{submission.permalink}",
                        )
                    )
            except Exception as exc:
                logger.info("reddit account %s skipped: %s", account, redact(str(exc)))
        return posts[: request.limit]


class InstagramSource(SocialSource):
    """Stub. Official Instagram API access does not offer cashtag search.

    This class never scrapes and never returns posts. It exists so the
    collector has a stable plugin point if a compliant integration is added later.
    """

    name = "instagram"

    async def _fetch(self, request: SocialRequest) -> list[SocialPost]:
        logger.info(
            "Instagram source is a stub. Official API access is limited to approved "
            "business use and cannot search cashtags. No Instagram data was collected."
        )
        return []


def _mentioned(text: str, symbols: list[str]) -> list[str]:
    upper = text.upper()
    found = []
    for symbol in symbols:
        base = symbol.split("/")[0].upper()
        if f"${base}" in upper or base in upper:
            found.append(symbol)
    return found


class SocialCollector:
    def __init__(self, sources: list[SocialSource]):
        self.sources = sources

    async def collect(self, request: SocialRequest) -> SocialCollectResult:
        posts: list[SocialPost] = []
        failed: list[str] = []
        skipped: list[str] = []
        for source in self.sources:
            try:
                posts.extend(await source.fetch(request))
            except NotConfigured as exc:
                skipped.append(source.name)
                logger.info("%s skipped: %s", source.name, exc)
            except Exception as exc:
                failed.append(source.name)
                logger.warning("%s failed and will be ignored this cycle: %s", source.name, redact(str(exc)))
        return SocialCollectResult(posts=posts, failed=failed, skipped=skipped)


def build_sources(config: AppConfig, cache_dir: Path) -> list[SocialSource]:
    cache = JsonCache(cache_dir)
    mapping = {
        "twitter": TwitterSource,
        "reddit": RedditSource,
        "instagram": InstagramSource,
    }
    sources: list[SocialSource] = []
    for name in config.social.enabled_sources:
        klass = mapping.get(name.lower())
        if klass is None:
            logger.warning("unknown social source %s", name)
            continue
        sources.append(klass(config.social, cache))
    return sources
