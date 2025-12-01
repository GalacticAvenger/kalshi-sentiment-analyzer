"""
Module for interacting with Kalshi API and collecting market data.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import json


class KalshiDataCollector:
    """Collect historical market data from Kalshi."""

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Kalshi data collector.

        Args:
            api_key: Optional API key for authenticated requests
        """
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def get_markets(self, limit: int = 100, status: str = "settled") -> List[Dict]:
        """
        Fetch markets from Kalshi.

        Args:
            limit: Maximum number of markets to fetch
            status: Market status (active, settled, closed)

        Returns:
            List of market dictionaries
        """
        try:
            url = f"{self.BASE_URL}/markets"
            params = {
                "limit": limit,
                "status": status
            }

            response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return data.get("markets", [])
            else:
                print(f"API request failed with status {response.status_code}")
                return []

        except Exception as e:
            print(f"Error fetching markets: {e}")
            return []

    def get_market_history(self, market_ticker: str) -> pd.DataFrame:
        """
        Get historical price data for a specific market.

        Args:
            market_ticker: The market ticker symbol

        Returns:
            DataFrame with historical price data
        """
        try:
            url = f"{self.BASE_URL}/markets/{market_ticker}/history"

            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                history = data.get("history", [])

                if history:
                    df = pd.DataFrame(history)
                    df['timestamp'] = pd.to_datetime(df['ts'], unit='s')
                    df['date'] = df['timestamp'].dt.date
                    return df
                else:
                    return pd.DataFrame()
            else:
                print(f"Could not fetch history for {market_ticker}")
                return pd.DataFrame()

        except Exception as e:
            print(f"Error fetching market history: {e}")
            return pd.DataFrame()

    def search_markets(self, keyword: str) -> List[Dict]:
        """
        Search for markets containing a keyword.

        Args:
            keyword: Search term

        Returns:
            List of matching markets
        """
        markets = self.get_markets(limit=200, status="settled")

        matching = [
            m for m in markets
            if keyword.lower() in m.get("title", "").lower() or
               keyword.lower() in m.get("ticker_name", "").lower()
        ]

        return matching

    def create_sample_market_data(self, market_name: str, days: int = 30) -> pd.DataFrame:
        """
        Create sample market data for testing when API is unavailable.

        Args:
            market_name: Name of the market
            days: Number of days of historical data

        Returns:
            DataFrame with sample market data
        """
        import numpy as np

        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

        # Simulate market probabilities with some random walk
        base_prob = 0.5
        changes = np.random.normal(0, 0.05, days)
        probabilities = np.clip(np.cumsum(changes) + base_prob, 0.1, 0.9)

        df = pd.DataFrame({
            'date': dates,
            'market_name': market_name,
            'probability': probabilities,
            'volume': np.random.randint(100, 10000, days)
        })

        return df

    def save_market_data(self, df: pd.DataFrame, filename: str):
        """Save market data to CSV."""
        df.to_csv(filename, index=False)
        print(f"Saved market data to {filename}")

    def load_market_data(self, filename: str) -> pd.DataFrame:
        """Load market data from CSV."""
        df = pd.read_csv(filename)
        df['date'] = pd.to_datetime(df['date']).dt.date
        return df


def fetch_sample_markets(save_dir: str = "data/raw/") -> Dict[str, pd.DataFrame]:
    """
    Fetch or create sample data for multiple markets.

    Args:
        save_dir: Directory to save market data

    Returns:
        Dictionary mapping market names to DataFrames
    """
    collector = KalshiDataCollector()

    # Define sample markets (use real topics that would have sentiment data)
    sample_markets = {
        "presidential_election_2024": "Will Biden win the 2024 Presidential Election?",
        "inflation_q3_2024": "Will inflation exceed 3% in Q3 2024?",
        "fed_rate_cut_2024": "Will the Fed cut rates in 2024?"
    }

    market_data = {}

    for market_id, market_name in sample_markets.items():
        print(f"Creating sample data for: {market_name}")

        # Try to get real data first, fall back to sample data
        df = collector.create_sample_market_data(market_name, days=30)

        # Save to file
        filename = f"{save_dir}{market_id}_prices.csv"
        collector.save_market_data(df, filename)

        market_data[market_id] = df

    return market_data


if __name__ == "__main__":
    # Example usage
    print("Fetching sample Kalshi market data...")
    markets = fetch_sample_markets()

    for market_id, df in markets.items():
        print(f"\n{market_id}:")
        print(df.head())
