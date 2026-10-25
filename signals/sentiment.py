"""Turn social posts into a per-asset summary.

The default scorer is a small lexicon. It is not a trading instruction.
Optional LLM scoring is batched and, if it fails, the lexicon is used.
Spam and bot-like posts are dropped before ratios are computed.
"""

from __future__ import annotations

import inspect
import logging
import re
from collections import Counter

from core.config import AppConfig
from core.models import AssetSentiment, SocialPost

logger = logging.getLogger(__name__)

BULLISH = {
    "bullish",
    "breakout",
    "rally",
    "undervalued",
    "accumulate",
    "support",
    "buy",
    "long",
    "ath",
    "moon",
    "bounce",
    "uptrend",
}
BEARISH = {
    "bearish",
    "dump",
    "crash",
    "sell",
    "short",
    "resistance",
    "overvalued",
    "scam",
    "rug",
    "breakdown",
    "liquidation",
    "downtrend",
}
SPAM_PHRASES = (
    "giveaway",
    "send 1 btc",
    "airdrop claim",
    "dm me",
    "guaranteed profit",
    "double your",
)


def score_text(text: str) -> str:
    tokens = re.findall(r"[a-z]+", text.lower())
    bullish = sum(token in BULLISH for token in tokens)
    bearish = sum(token in BEARISH for token in tokens)
    if bullish > bearish:
        return "bullish"
    if bearish > bullish:
        return "bearish"
    return "neutral"


def is_spam(post: SocialPost, seen: set[str], author_counts: Counter) -> bool:
    text = re.sub(r"\s+", " ", post.text.strip().lower())
    if len(text) < 8:
        return True
    if len(re.findall(r"\$[a-z]{2,}", text)) > 5:
        return True
    if re.search(r"(.)\1{7,}", text):
        return True
    if any(phrase in text for phrase in SPAM_PHRASES):
        return True
    if text in seen:
        return True
    seen.add(text)
    author_counts[post.author] += 1
    if author_counts[post.author] > 5:
        return True
    return False


def _matches(post: SocialPost, symbol: str) -> bool:
    if symbol in post.symbols:
        return True
    base = symbol.split("/")[0].lower()
    text = post.text.lower()
    return f"${base}" in text or base in text


class LexiconScorer:
    name = "lexicon"

    def score(self, posts: list[SocialPost]) -> list[str]:
        return [score_text(post.text) for post in posts]


class BatchSentimentScorer:
    """Optional LLM scorer. `complete` returns a list of labels or raises."""

    name = "llm"

    def __init__(self, complete):
        self._complete = complete

    async def score(self, posts: list[SocialPost]) -> list[str]:
        if not posts:
            return []
        labels = await self._complete(posts)
        if not isinstance(labels, list) or len(labels) != len(posts):
            raise ValueError("sentiment scorer returned the wrong shape")
        cleaned = []
        for label in labels:
            value = str(label).lower().strip()
            if value not in {"bullish", "bearish", "neutral"}:
                raise ValueError("unknown sentiment label")
            cleaned.append(value)
        return cleaned


class SentimentService:
    def __init__(self, config: AppConfig, scorer=None):
        self.config = config
        self.scorer = scorer or LexiconScorer()

    async def summarize(self, symbols: list[str], posts: list[SocialPost]) -> list[AssetSentiment]:
        seen: set[str] = set()
        authors: Counter = Counter()
        kept: list[SocialPost] = []
        spam = 0
        for post in posts:
            if is_spam(post, seen, authors):
                spam += 1
                continue
            kept.append(post)
        labels: list[str]
        scorer_name = getattr(self.scorer, "name", "lexicon")
        try:
            if inspect.iscoroutinefunction(self.scorer.score):
                labels = await self.scorer.score(kept)
            else:
                labels = self.scorer.score(kept)
        except Exception as exc:
            logger.warning("sentiment scorer failed, using lexicon: %s", exc)
            labels = LexiconScorer().score(kept)
            scorer_name = "lexicon_fallback"
        summaries = []
        for symbol in symbols:
            indexes = [index for index, post in enumerate(kept) if _matches(post, symbol)]
            chosen_posts = [kept[index] for index in indexes]
            chosen_labels = [labels[index] for index in indexes]
            summaries.append(self._one(symbol, chosen_posts, chosen_labels, spam, scorer_name))
        return summaries

    def _one(self, symbol: str, posts: list[SocialPost], labels: list[str], spam: int, scorer_name: str) -> AssetSentiment:
        bullish = labels.count("bullish")
        bearish = labels.count("bearish")
        neutral = labels.count("neutral")
        total = max(len(labels), 1) if labels else 1
        if not labels:
            bullish_ratio = bearish_ratio = neutral_ratio = 0.0
            total_posts = 0
        else:
            bullish_ratio = bullish / len(labels)
            bearish_ratio = bearish / len(labels)
            neutral_ratio = neutral / len(labels)
            total_posts = len(labels)
        ranked = sorted(posts, key=lambda post: post.engagement.total, reverse=True)
        notable = []
        for post in ranked:
            if post.author not in notable:
                notable.append(post.author)
            if len(notable) == 3:
                break
        spike = total_posts >= self.config.social.spike_post_count
        summary = (
            f"{symbol}: posts={total_posts} bullish_ratio={bullish_ratio:.2f} "
            f"bearish_ratio={bearish_ratio:.2f} volume_spike={spike} "
            f"notable={','.join(notable) if notable else 'none'} spam_filtered={spam} scorer={scorer_name}"
        )
        return AssetSentiment(
            symbol=symbol,
            post_count=total_posts,
            bullish=bullish,
            bearish=bearish,
            neutral=neutral,
            bullish_ratio=bullish_ratio,
            bearish_ratio=bearish_ratio,
            neutral_ratio=neutral_ratio,
            volume_spike=spike,
            notable_accounts=notable,
            spam_filtered=spam,
            summary=summary,
            scorer=scorer_name,
        )

