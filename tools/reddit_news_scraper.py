"""
Data-gathering tools used by Team C (Data Puller, Sentiment Analyst):
Yahoo Finance price history, Reddit (PRAW), and news headlines via RSS.
"""

import os

import numpy as np
from dotenv import load_dotenv

try:
    import feedparser
except ImportError:
    feedparser = None

try:
    import praw
except ImportError:
    praw = None

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))


def get_vader_analyzer():
    """Lazily load or initialize the NLTK VADER SentimentIntensityAnalyzer."""
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        try:
            return SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            return SentimentIntensityAnalyzer()
    except Exception:
        return None


def calculate_sentiment_metrics(texts: list[str], weights: list[float] | None = None) -> dict:
    """Compute detailed sentiment statistics using NLTK VADER."""
    analyzer = get_vader_analyzer()
    if not analyzer or not texts:
        return {
            "mean_compound": 0.0,
            "std_compound": 0.0,
            "positive_ratio": 0.0,
            "negative_ratio": 0.0,
            "neutral_ratio": 1.0,
            "weighted_compound": 0.0,
            "sample_count": 0,
            "available": False,
        }

    compounds = []
    pos_count = 0
    neg_count = 0
    neu_count = 0

    valid_weights = []
    if weights and len(weights) == len(texts):
        valid_weights = [max(0.1, float(w)) for w in weights]
    else:
        valid_weights = [1.0] * len(texts)

    for text in texts:
        if not text or not text.strip():
            compounds.append(0.0)
            neu_count += 1
            continue
        scores = analyzer.polarity_scores(text)
        cmp = scores["compound"]
        compounds.append(cmp)
        if cmp >= 0.05:
            pos_count += 1
        elif cmp <= -0.05:
            neg_count += 1
        else:
            neu_count += 1

    total = len(texts)
    mean_cmp = float(np.mean(compounds)) if compounds else 0.0
    std_cmp = float(np.std(compounds)) if len(compounds) > 1 else 0.0
    w_sum = sum(valid_weights)
    weighted_cmp = float(sum(c * w for c, w in zip(compounds, valid_weights)) / max(w_sum, 1e-9))

    return {
        "mean_compound": round(mean_cmp, 4),
        "std_compound": round(std_cmp, 4),
        "positive_ratio": round(pos_count / max(total, 1), 4),
        "negative_ratio": round(neg_count / max(total, 1), 4),
        "neutral_ratio": round(neu_count / max(total, 1), 4),
        "weighted_compound": round(weighted_cmp, 4),
        "sample_count": total,
        "available": True,
    }


def get_reddit_client():
    """Build an authenticated read-only PRAW client from environment variables."""
    if praw is None:
        raise RuntimeError("praw is not installed")
    return praw.Reddit(
        # .strip() guards against stray whitespace/tab characters that can
        # sneak in when a secret is copy-pasted into a CI provider's UI --
        # Reddit's OAuth endpoint rejects a corrupted secret with a plain 401.
        client_id=os.environ["REDDIT_CLIENT_ID"].strip(),
        client_secret=os.environ["REDDIT_CLIENT_SECRET"].strip(),
        user_agent=os.environ.get("REDDIT_USER_AGENT", "trading-system-sentiment-bot/1.0").strip(),
    )


def scrape_reddit_mentions(symbol: str, subreddits: list[str] | None = None, limit: int = 50) -> list[dict]:
    """
    Pull recent posts mentioning `symbol` from the given subreddits
    (defaults to wallstreetbets, stocks, investing) for sentiment scoring.
    """
    subreddits = subreddits or ["wallstreetbets", "stocks", "investing"]
    reddit = get_reddit_client()
    posts = []
    for sub_name in subreddits:
        subreddit = reddit.subreddit(sub_name)
        for submission in subreddit.search(symbol, sort="new", time_filter="week", limit=limit):
            posts.append(
                {
                    "subreddit": sub_name,
                    "title": submission.title,
                    "selftext": submission.selftext,
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "created_utc": submission.created_utc,
                    "url": submission.url,
                }
            )
    return posts


DEFAULT_NEWS_RSS_FEEDS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US",
    "https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en",
]


def scrape_news_headlines(symbol: str, feed_templates: list[str] | None = None, limit: int = 50) -> list[dict]:
    """
    Pull recent news headlines mentioning `symbol` from RSS feeds (Yahoo
    Finance and Google News by default). No API key required.
    """
    if feedparser is None:
        print("[Market Intelligence] feedparser not installed -- skipping RSS headlines.")
        return []
    feed_templates = feed_templates or DEFAULT_NEWS_RSS_FEEDS
    articles = []
    for template in feed_templates:
        feed_url = template.format(symbol=symbol)
        parsed = feedparser.parse(feed_url)
        for entry in parsed.entries[:limit]:
            articles.append(
                {
                    "source": parsed.feed.get("title", feed_url),
                    "title": entry.get("title", ""),
                    "description": entry.get("summary", ""),
                    "published_at": entry.get("published", ""),
                    "url": entry.get("link", ""),
                }
            )
    return articles


def get_yahoo_price_history(symbol: str, period: str = "2y", interval: str = "1d"):
    """
    Pull OHLCV price history from Yahoo Finance as a pandas DataFrame.
    Uses the `yfinance` library, which wraps Yahoo's public chart API.
    """
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    return ticker.history(period=period, interval=interval)
