"""
Real news data collector using Google News RSS feeds.
No API key required for news collection.
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import xml.etree.ElementTree as ET
import re
from urllib.parse import quote
import time


class NewsCollector:
    """Collect real news headlines from Google News RSS."""

    def __init__(self):
        """Initialize news collector."""
        self.base_url = "https://news.google.com/rss/search"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }

    def search_news(
        self,
        query: str,
        days_back: int = 30,
        max_results: int = 100
    ) -> pd.DataFrame:
        """
        Search Google News for articles matching query.

        Args:
            query: Search query (e.g., "Biden election", "inflation CPI")
            days_back: How many days back to search
            max_results: Maximum number of results

        Returns:
            DataFrame with news articles
        """
        print(f"Searching Google News for: '{query}'...")

        # Build URL with query
        encoded_query = quote(query)
        url = f"{self.base_url}?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            # Parse RSS XML
            root = ET.fromstring(response.content)

            articles = []
            cutoff_date = datetime.now() - timedelta(days=days_back)

            for item in root.findall('.//item'):
                title = item.find('title')
                pub_date = item.find('pubDate')
                source = item.find('source')
                link = item.find('link')

                if title is not None and pub_date is not None:
                    # Parse date
                    try:
                        date_str = pub_date.text
                        # Format: "Sat, 30 Nov 2024 12:00:00 GMT"
                        parsed_date = datetime.strptime(
                            date_str.replace(' GMT', ''),
                            '%a, %d %b %Y %H:%M:%S'
                        )

                        if parsed_date >= cutoff_date:
                            articles.append({
                                'date': parsed_date.date(),
                                'datetime': parsed_date,
                                'text': title.text,
                                'source': source.text if source is not None else 'Unknown',
                                'url': link.text if link is not None else '',
                                'query': query
                            })
                    except (ValueError, AttributeError):
                        continue

                if len(articles) >= max_results:
                    break

            print(f"Found {len(articles)} articles")
            return pd.DataFrame(articles)

        except Exception as e:
            print(f"Error fetching news: {e}")
            return pd.DataFrame()

    def collect_market_news(
        self,
        market_topic: str,
        keywords: Optional[List[str]] = None,
        days_back: int = 30
    ) -> pd.DataFrame:
        """
        Collect news for a specific market topic.

        Args:
            market_topic: Main topic (e.g., "Biden 2024 election")
            keywords: Additional keywords to search
            days_back: Days of history to collect

        Returns:
            Combined DataFrame of all news
        """
        all_news = []

        # Search main topic
        df = self.search_news(market_topic, days_back)
        if not df.empty:
            all_news.append(df)

        # Search additional keywords
        if keywords:
            for keyword in keywords:
                time.sleep(0.5)  # Rate limiting
                df = self.search_news(keyword, days_back)
                if not df.empty:
                    all_news.append(df)

        if all_news:
            combined = pd.concat(all_news, ignore_index=True)
            # Remove duplicates by title
            combined = combined.drop_duplicates(subset=['text'])
            # Sort by date
            combined = combined.sort_values('datetime', ascending=False)
            print(f"Total unique articles: {len(combined)}")
            return combined

        return pd.DataFrame()


class RedditCollector:
    """Collect posts from Reddit (no API key needed for public data)."""

    def __init__(self):
        """Initialize Reddit collector."""
        self.base_url = "https://www.reddit.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (research project)'
        }

    def search_subreddit(
        self,
        subreddit: str,
        query: str,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Search a subreddit for posts.

        Args:
            subreddit: Subreddit name (e.g., "wallstreetbets")
            query: Search query
            limit: Max posts to return

        Returns:
            DataFrame with posts
        """
        print(f"Searching r/{subreddit} for: '{query}'...")

        url = f"{self.base_url}/r/{subreddit}/search.json"
        params = {
            'q': query,
            'restrict_sr': 'on',
            'sort': 'new',
            'limit': min(limit, 100)
        }

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            posts = []
            for post in data.get('data', {}).get('children', []):
                post_data = post.get('data', {})
                created = post_data.get('created_utc', 0)

                posts.append({
                    'date': datetime.fromtimestamp(created).date(),
                    'datetime': datetime.fromtimestamp(created),
                    'text': post_data.get('title', ''),
                    'body': post_data.get('selftext', '')[:500],  # Truncate
                    'source': f"reddit/r/{subreddit}",
                    'score': post_data.get('score', 0),
                    'num_comments': post_data.get('num_comments', 0),
                    'url': f"https://reddit.com{post_data.get('permalink', '')}"
                })

            print(f"Found {len(posts)} posts")
            return pd.DataFrame(posts)

        except Exception as e:
            print(f"Error fetching Reddit data: {e}")
            return pd.DataFrame()

    def collect_from_subreddits(
        self,
        subreddits: List[str],
        query: str,
        limit_per_sub: int = 50
    ) -> pd.DataFrame:
        """Collect from multiple subreddits."""
        all_posts = []

        for sub in subreddits:
            time.sleep(1)  # Rate limiting
            df = self.search_subreddit(sub, query, limit_per_sub)
            if not df.empty:
                all_posts.append(df)

        if all_posts:
            combined = pd.concat(all_posts, ignore_index=True)
            combined = combined.drop_duplicates(subset=['text'])
            combined = combined.sort_values('datetime', ascending=False)
            return combined

        return pd.DataFrame()


def collect_real_sentiment_data(
    market_name: str,
    days_back: int = 30,
    include_reddit: bool = True
) -> pd.DataFrame:
    """
    Collect real sentiment data for a market.

    Args:
        market_name: Name of the market to analyze
        days_back: Days of history
        include_reddit: Whether to include Reddit data

    Returns:
        DataFrame with text data ready for sentiment analysis
    """
    all_data = []

    # Determine search terms based on market name
    market_lower = market_name.lower()

    if "biden" in market_lower or "trump" in market_lower or "election" in market_lower:
        news_queries = ["Biden election 2024", "Trump election 2024", "presidential polls"]
        reddit_subs = ["politics", "news", "PoliticalDiscussion"]
        reddit_query = "Biden OR Trump election"
    elif "inflation" in market_lower or "cpi" in market_lower:
        news_queries = ["inflation CPI", "consumer prices", "Fed inflation"]
        reddit_subs = ["economics", "finance", "wallstreetbets"]
        reddit_query = "inflation CPI prices"
    elif "fed" in market_lower or "rate" in market_lower:
        news_queries = ["Federal Reserve rate cut", "Fed interest rates", "FOMC meeting"]
        reddit_subs = ["economics", "finance", "wallstreetbets"]
        reddit_query = "Fed rate cut interest"
    else:
        # Generic search
        news_queries = [market_name]
        reddit_subs = ["news", "finance"]
        reddit_query = market_name

    # Collect news
    news_collector = NewsCollector()
    for query in news_queries:
        df = news_collector.search_news(query, days_back)
        if not df.empty:
            df['data_source'] = 'google_news'
            all_data.append(df)
        time.sleep(0.5)

    # Collect Reddit
    if include_reddit:
        reddit_collector = RedditCollector()
        df = reddit_collector.collect_from_subreddits(reddit_subs, reddit_query)
        if not df.empty:
            df['data_source'] = 'reddit'
            all_data.append(df)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined = combined.drop_duplicates(subset=['text'])
        combined['market_name'] = market_name

        # Ensure we have required columns
        if 'date' not in combined.columns:
            combined['date'] = datetime.now().date()

        print(f"\nTotal collected: {len(combined)} texts")
        print(f"  - News: {len(combined[combined['data_source'] == 'google_news'])}")
        if include_reddit:
            print(f"  - Reddit: {len(combined[combined['data_source'] == 'reddit'])}")

        return combined

    print("No data collected")
    return pd.DataFrame()


if __name__ == "__main__":
    # Test the collectors
    print("Testing News Collector...")
    print("=" * 50)

    df = collect_real_sentiment_data("Biden 2024 Election", days_back=7)

    if not df.empty:
        print("\nSample data:")
        print(df[['date', 'text', 'source']].head(10))
    else:
        print("No data collected - check internet connection")
