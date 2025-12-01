"""
Sentiment analysis module using pre-trained transformer models.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm
import warnings

# Disable tokenizer parallelism to prevent mutex issues
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

warnings.filterwarnings('ignore')

# Lazy imports to speed up module loading
_transformers_loaded = False
_torch_loaded = False


class SimpleSentimentAnalyzer:
    """
    Simple keyword-based sentiment analyzer that doesn't require ML models.
    Use this when transformers models are unavailable or too slow.
    """

    # Sentiment word lists
    POSITIVE_WORDS = {
        'good', 'great', 'excellent', 'positive', 'strong', 'better', 'best',
        'win', 'winning', 'success', 'successful', 'gain', 'gains', 'up',
        'rise', 'rising', 'increase', 'growth', 'momentum', 'leading', 'lead',
        'support', 'optimistic', 'confidence', 'confident', 'bullish', 'favor',
        'advantage', 'ahead', 'outperform', 'surge', 'rally', 'boost', 'improve'
    }

    NEGATIVE_WORDS = {
        'bad', 'poor', 'negative', 'weak', 'worse', 'worst', 'fail', 'failing', 'failed',
        'loss', 'lose', 'losing', 'down', 'fall', 'falling', 'decrease', 'decline',
        'concern', 'concerned', 'worry', 'worried', 'bearish', 'trouble', 'problem',
        'challenge', 'challenging', 'behind', 'underperform', 'drop', 'crash', 'risk',
        'uncertain', 'uncertainty', 'doubt', 'struggle', 'struggling', 'trail', 'trails',
        'terrible', 'awful', 'horrible', 'disaster', 'catastrophe', 'crisis', 'threat'
    }

    def __init__(self):
        print("Using SimpleSentimentAnalyzer (keyword-based)")
        self.pipeline = None  # For compatibility

    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using keyword matching."""
        if not text or len(text.strip()) == 0:
            return {"label": "NEUTRAL", "score": 0.5, "normalized_score": 0.0}

        words = text.lower().split()

        pos_count = sum(1 for w in words if w in self.POSITIVE_WORDS)
        neg_count = sum(1 for w in words if w in self.NEGATIVE_WORDS)
        total = pos_count + neg_count

        if total == 0:
            return {"label": "NEUTRAL", "score": 0.5, "normalized_score": 0.0}

        # Calculate normalized score from -1 to +1
        normalized_score = (pos_count - neg_count) / total

        if normalized_score > 0.1:
            label = "POSITIVE"
            score = min(0.5 + normalized_score * 0.5, 1.0)
        elif normalized_score < -0.1:
            label = "NEGATIVE"
            score = min(0.5 + abs(normalized_score) * 0.5, 1.0)
        else:
            label = "NEUTRAL"
            score = 0.5

        return {
            "label": label,
            "score": score,
            "normalized_score": normalized_score
        }

    def analyze_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, float]]:
        """Analyze multiple texts."""
        return [self.analyze_text(t) for t in tqdm(texts, desc="Analyzing sentiment")]

    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """Add sentiment analysis to a DataFrame."""
        print(f"Analyzing sentiment for {len(df)} texts...")
        texts = df[text_column].fillna("").tolist()
        results = self.analyze_batch(texts)

        df['sentiment_label'] = [r['label'] for r in results]
        df['sentiment_score'] = [r['score'] for r in results]
        df['sentiment_normalized'] = [r['normalized_score'] for r in results]

        return df


class SentimentAnalyzer:
    """Analyze sentiment of text using pre-trained models."""

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english", use_simple: bool = False):
        """
        Initialize sentiment analyzer.

        Args:
            model_name: HuggingFace model name. Options:
                - "cardiffnlp/twitter-roberta-base-sentiment-latest" (Twitter/social media)
                - "ProsusAI/finbert" (Financial text)
                - "distilbert-base-uncased-finetuned-sst-2-english" (General - default, fastest)
            use_simple: If True, use simple keyword-based analyzer instead of ML model
        """
        if use_simple:
            print("Using simple keyword-based sentiment analyzer")
            self._simple = SimpleSentimentAnalyzer()
            self.pipeline = None
            self.model_name = "simple"
            return

        # Import here to avoid slow module loading
        try:
            from transformers import pipeline
            import torch
        except ImportError as e:
            print(f"Transformers not available: {e}")
            print("Falling back to simple analyzer...")
            self._simple = SimpleSentimentAnalyzer()
            self.pipeline = None
            self.model_name = "simple"
            return

        print(f"Loading sentiment model: {model_name}")
        self.model_name = model_name
        self._simple = None

        try:
            # Check if GPU is available
            device = 0 if torch.cuda.is_available() else -1

            # Load the sentiment analysis pipeline
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=device,
                truncation=True,
                max_length=512
            )

            print(f"Model loaded successfully on {'GPU' if device == 0 else 'CPU'}")

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to simple analyzer...")
            self._simple = SimpleSentimentAnalyzer()
            self.pipeline = None

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment label and score
        """
        # Use simple analyzer fallback if pipeline not available
        if self._simple is not None:
            return self._simple.analyze_text(text)

        if not text or len(text.strip()) == 0:
            return {"label": "NEUTRAL", "score": 0.0, "normalized_score": 0.0}

        try:
            result = self.pipeline(text[:512])[0]  # Truncate to max length

            # Normalize score to -1 to 1 scale
            label = result['label'].upper()
            score = result['score']

            if 'NEGATIVE' in label or label == 'LABEL_0':
                normalized_score = -score
            elif 'POSITIVE' in label or label == 'LABEL_2' or label == 'LABEL_1':
                normalized_score = score
            else:  # NEUTRAL or unknown
                normalized_score = 0.0

            return {
                "label": label,
                "score": score,
                "normalized_score": normalized_score
            }

        except Exception as e:
            print(f"Error analyzing text: {e}")
            return {"label": "ERROR", "score": 0.0, "normalized_score": 0.0}

    def analyze_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, float]]:
        """
        Analyze sentiment for a batch of texts.

        Args:
            texts: List of texts to analyze
            batch_size: Number of texts to process at once

        Returns:
            List of sentiment dictionaries
        """
        # Use simple analyzer fallback if pipeline not available
        if self._simple is not None:
            return self._simple.analyze_batch(texts, batch_size)

        results = []

        # Process in batches with progress bar
        for i in tqdm(range(0, len(texts), batch_size), desc="Analyzing sentiment"):
            batch = texts[i:i + batch_size]

            # Filter out empty texts
            valid_texts = [t[:512] if t and len(t.strip()) > 0 else "neutral text" for t in batch]

            try:
                batch_results = self.pipeline(valid_texts)

                for result in batch_results:
                    label = result['label'].upper()
                    score = result['score']

                    if 'NEGATIVE' in label or label == 'LABEL_0':
                        normalized_score = -score
                    elif 'POSITIVE' in label or label == 'LABEL_2' or label == 'LABEL_1':
                        normalized_score = score
                    else:
                        normalized_score = 0.0

                    results.append({
                        "label": label,
                        "score": score,
                        "normalized_score": normalized_score
                    })

            except Exception as e:
                print(f"Error processing batch: {e}")
                # Add neutral results for failed batch
                results.extend([{"label": "ERROR", "score": 0.0, "normalized_score": 0.0}] * len(batch))

        return results

    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Add sentiment analysis to a DataFrame.

        Args:
            df: DataFrame with text data
            text_column: Name of column containing text

        Returns:
            DataFrame with sentiment columns added
        """
        # Use simple analyzer fallback if pipeline not available
        if self._simple is not None:
            return self._simple.analyze_dataframe(df, text_column)

        print(f"Analyzing sentiment for {len(df)} texts...")

        # Get texts
        texts = df[text_column].fillna("").tolist()

        # Analyze in batches
        results = self.analyze_batch(texts)

        # Add results to dataframe
        df['sentiment_label'] = [r['label'] for r in results]
        df['sentiment_score'] = [r['score'] for r in results]
        df['sentiment_normalized'] = [r['normalized_score'] for r in results]

        return df


class APISentimentAnalyzer:
    """
    Sentiment analyzer using LLM APIs (OpenAI or Anthropic).
    Much more accurate than keyword-based, no local model issues.
    """

    def __init__(self, provider: str = "openai", api_key: Optional[str] = None):
        """
        Initialize API-based sentiment analyzer.

        Args:
            provider: "openai" or "anthropic"
            api_key: API key (or set OPENAI_API_KEY / ANTHROPIC_API_KEY env var)
        """
        self.provider = provider.lower()
        self.api_key = api_key or os.environ.get(
            "OPENAI_API_KEY" if self.provider == "openai" else "ANTHROPIC_API_KEY"
        )

        if not self.api_key:
            raise ValueError(f"No API key provided. Set {self.provider.upper()}_API_KEY environment variable.")

        print(f"Using {self.provider.upper()} API for sentiment analysis")

    def _call_openai(self, texts: List[str]) -> List[Dict[str, float]]:
        """Call OpenAI API for sentiment analysis."""
        import requests
        import json
        import time
        import re

        results = []

        for text in texts:
            # Process one at a time to avoid rate limits and parsing issues
            prompt = f"""Analyze the sentiment of this text. Respond with ONLY valid JSON (no markdown):
{{"label": "POSITIVE" or "NEGATIVE" or "NEUTRAL", "score": 0.0 to 1.0, "normalized_score": -1.0 to 1.0}}

Text: {text[:500]}"""

            max_retries = 2
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "gpt-4o-mini",
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0,
                            "max_tokens": 100
                        },
                        timeout=30
                    )

                    if response.status_code == 429:  # Rate limited
                        time.sleep(2)
                        continue

                    response.raise_for_status()
                    content = response.json()["choices"][0]["message"]["content"]

                    # Clean up response - remove markdown code blocks if present
                    content = content.strip()
                    if content.startswith("```"):
                        content = re.sub(r'^```json?\n?', '', content)
                        content = re.sub(r'\n?```$', '', content)

                    result = json.loads(content.strip())
                    results.append(result)
                    break

                except Exception as e:
                    if attempt == max_retries - 1:
                        # Fallback to neutral
                        results.append({"label": "NEUTRAL", "score": 0.5, "normalized_score": 0.0})
                    else:
                        time.sleep(1)

            # Small delay to avoid rate limits
            time.sleep(0.1)

        return results

    def _call_anthropic(self, texts: List[str]) -> List[Dict[str, float]]:
        """Call Anthropic API for sentiment analysis."""
        import requests

        results = []

        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            texts_formatted = "\n".join([f"{j+1}. {t}" for j, t in enumerate(batch)])
            prompt = f"""Analyze the sentiment of each text below. For each, respond with ONLY a JSON array where each element has:
- "label": "POSITIVE", "NEGATIVE", or "NEUTRAL"
- "score": confidence from 0.0 to 1.0
- "normalized_score": -1.0 (very negative) to 1.0 (very positive)

Texts:
{texts_formatted}

Respond with ONLY valid JSON array, no other text."""

            try:
                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "claude-3-haiku-20240307",
                        "max_tokens": 1024,
                        "messages": [{"role": "user", "content": prompt}]
                    },
                    timeout=30
                )
                response.raise_for_status()

                content = response.json()["content"][0]["text"]
                import json
                batch_results = json.loads(content.strip())
                results.extend(batch_results)

            except Exception as e:
                print(f"API error: {e}")
                results.extend([{"label": "NEUTRAL", "score": 0.5, "normalized_score": 0.0}] * len(batch))

        return results

    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of a single text."""
        if not text or len(text.strip()) == 0:
            return {"label": "NEUTRAL", "score": 0.5, "normalized_score": 0.0}

        results = self.analyze_batch([text])
        return results[0] if results else {"label": "NEUTRAL", "score": 0.5, "normalized_score": 0.0}

    def analyze_batch(self, texts: List[str], batch_size: int = 10) -> List[Dict[str, float]]:
        """Analyze sentiment for multiple texts."""
        if self.provider == "openai":
            return self._call_openai(texts)
        else:
            return self._call_anthropic(texts)

    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """Add sentiment analysis to a DataFrame."""
        print(f"Analyzing sentiment for {len(df)} texts using {self.provider.upper()} API...")

        texts = df[text_column].fillna("").tolist()

        # Process all texts (the _call methods handle individual processing)
        results = []
        for i in tqdm(range(len(texts)), desc="Analyzing sentiment"):
            text = texts[i]
            if text and len(text.strip()) > 0:
                result = self.analyze_batch([text])
                results.extend(result)
            else:
                results.append({"label": "NEUTRAL", "score": 0.5, "normalized_score": 0.0})

        df['sentiment_label'] = [r['label'] for r in results]
        df['sentiment_score'] = [r['score'] for r in results]
        df['sentiment_normalized'] = [r['normalized_score'] for r in results]

        return df


class VADERSentimentAnalyzer:
    """Simple lexicon-based sentiment analyzer using VADER."""

    def __init__(self):
        """Initialize VADER sentiment analyzer."""
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
            import nltk

            # Download VADER lexicon
            try:
                nltk.download('vader_lexicon', quiet=True)
            except:
                pass

            self.sia = SentimentIntensityAnalyzer()
            print("VADER sentiment analyzer loaded")

        except Exception as e:
            print(f"Error loading VADER: {e}")
            print("Install with: pip install nltk")
            self.sia = None

    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER."""
        if not self.sia or not text:
            return {"compound": 0.0, "pos": 0.0, "neu": 0.0, "neg": 0.0}

        scores = self.sia.polarity_scores(text)
        return scores

    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """Add VADER sentiment to DataFrame."""
        if not self.sia:
            print("VADER not available")
            return df

        print(f"Analyzing sentiment with VADER for {len(df)} texts...")

        results = [self.analyze_text(text) for text in tqdm(df[text_column].fillna(""))]

        df['vader_compound'] = [r['compound'] for r in results]
        df['vader_pos'] = [r['pos'] for r in results]
        df['vader_neu'] = [r['neu'] for r in results]
        df['vader_neg'] = [r['neg'] for r in results]

        return df


def create_sample_text_data(market_name: str, days: int = 30) -> pd.DataFrame:
    """
    Create sample social media/news text data for testing.

    Args:
        market_name: Name of the market
        days: Number of days of data

    Returns:
        DataFrame with sample text data
    """
    from datetime import datetime, timedelta

    # Sample texts based on market topic
    if "election" in market_name.lower():
        positive_texts = [
            "Biden's campaign is gaining strong momentum in key swing states",
            "Latest polls show Biden with growing support among voters",
            "Biden's economic policies receiving positive feedback",
            "Strong turnout expected to favor Biden in the election",
            "Biden leading in multiple state polls"
        ]
        negative_texts = [
            "Biden facing challenges in latest polling data",
            "Concerns raised about Biden's campaign strategy",
            "Biden trails in several key battleground states",
            "Questions about Biden's performance in debates",
            "Biden's approval ratings showing decline"
        ]
    elif "inflation" in market_name.lower():
        positive_texts = [
            "Inflation data shows signs of cooling",
            "Economists optimistic about inflation trajectory",
            "Latest CPI report better than expected",
            "Inflation pressures appearing to ease",
            "Positive outlook for inflation in coming months"
        ]
        negative_texts = [
            "Inflation remains stubbornly high",
            "Concerns about persistent inflationary pressures",
            "Inflation numbers exceed expectations",
            "Rising prices continue to worry consumers",
            "Inflation showing no signs of significant decline"
        ]
    else:  # Fed rate cuts
        positive_texts = [
            "Fed signals willingness to cut rates soon",
            "Economic data supports case for rate cuts",
            "Markets anticipate Fed rate reduction",
            "Fed officials hint at easing monetary policy",
            "Rate cut expected in upcoming Fed meeting"
        ]
        negative_texts = [
            "Fed remains cautious about cutting rates",
            "No immediate rate cuts expected from Fed",
            "Fed maintains hawkish stance on rates",
            "Economic conditions don't support rate cuts yet",
            "Fed likely to keep rates unchanged"
        ]

    # Generate dates
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    # Create random text data
    texts = []
    sentiment_trend = np.cumsum(np.random.normal(0, 0.1, days))  # Random walk

    for i, date in enumerate(dates):
        # Number of posts per day (random)
        n_posts = np.random.randint(3, 10)

        for _ in range(n_posts):
            # Choose positive or negative based on trend
            if sentiment_trend[i] + np.random.normal(0, 0.3) > 0:
                text = np.random.choice(positive_texts)
            else:
                text = np.random.choice(negative_texts)

            texts.append({
                'date': date.date(),
                'text': text,
                'source': np.random.choice(['twitter', 'reddit', 'news']),
                'market_name': market_name
            })

    return pd.DataFrame(texts)


if __name__ == "__main__":
    # Example usage
    print("Testing sentiment analyzer...")

    # Create sample data
    df = create_sample_text_data("Biden 2024 Election", days=30)
    print(f"\nCreated {len(df)} sample texts")
    print(df.head())

    # Analyze sentiment
    analyzer = SentimentAnalyzer()
    df = analyzer.analyze_dataframe(df)

    print("\nSentiment analysis results:")
    print(df[['date', 'text', 'sentiment_label', 'sentiment_normalized']].head(10))

    # Show average sentiment by date
    daily_sentiment = df.groupby('date')['sentiment_normalized'].mean()
    print("\nDaily average sentiment:")
    print(daily_sentiment.head(10))
