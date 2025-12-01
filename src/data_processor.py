"""
Data processing and alignment module for combining market and sentiment data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


class DataProcessor:
    """Process and align market price and sentiment data."""

    def __init__(self):
        """Initialize data processor."""
        pass

    def align_data(
        self,
        market_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        date_column: str = 'date'
    ) -> pd.DataFrame:
        """
        Align market prices with sentiment data by date.

        Args:
            market_df: DataFrame with market price data
            sentiment_df: DataFrame with sentiment analysis results
            date_column: Name of date column

        Returns:
            Combined DataFrame with aligned data
        """
        print("Aligning market and sentiment data...")

        # Ensure date columns are in date format
        market_df[date_column] = pd.to_datetime(market_df[date_column]).dt.date
        sentiment_df[date_column] = pd.to_datetime(sentiment_df[date_column]).dt.date

        # Aggregate sentiment by date
        daily_sentiment = self.aggregate_sentiment_by_date(sentiment_df, date_column)

        # Merge with market data
        combined = market_df.merge(
            daily_sentiment,
            on=date_column,
            how='left'
        )

        # Fill missing sentiment values with 0 (neutral)
        sentiment_columns = [col for col in combined.columns if 'sentiment' in col.lower()]
        combined[sentiment_columns] = combined[sentiment_columns].fillna(0)

        print(f"Combined dataset has {len(combined)} rows")

        return combined

    def aggregate_sentiment_by_date(
        self,
        sentiment_df: pd.DataFrame,
        date_column: str = 'date'
    ) -> pd.DataFrame:
        """
        Aggregate sentiment scores by date.

        Args:
            sentiment_df: DataFrame with sentiment data
            date_column: Name of date column

        Returns:
            DataFrame with daily aggregated sentiment
        """
        # Group by date and calculate statistics
        agg_dict = {}

        if 'sentiment_normalized' in sentiment_df.columns:
            agg_dict['sentiment_normalized'] = ['mean', 'std', 'count']

        if 'vader_compound' in sentiment_df.columns:
            agg_dict['vader_compound'] = ['mean', 'std']

        if not agg_dict:
            print("Warning: No sentiment columns found")
            return pd.DataFrame()

        aggregated = sentiment_df.groupby(date_column).agg(agg_dict).reset_index()

        # Flatten column names
        aggregated.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                              for col in aggregated.columns.values]

        # Rename for clarity
        rename_dict = {}
        if 'sentiment_normalized_mean' in aggregated.columns:
            rename_dict.update({
                'sentiment_normalized_mean': 'avg_sentiment',
                'sentiment_normalized_std': 'sentiment_volatility',
                'sentiment_normalized_count': 'mention_count'
            })

        if 'vader_compound_mean' in aggregated.columns:
            rename_dict.update({
                'vader_compound_mean': 'vader_sentiment',
                'vader_compound_std': 'vader_volatility'
            })

        aggregated = aggregated.rename(columns=rename_dict)

        return aggregated

    def create_lagged_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        lags: List[int] = [1, 2, 3]
    ) -> pd.DataFrame:
        """
        Create lagged features for time series analysis.

        Args:
            df: DataFrame with time series data
            columns: Columns to create lags for
            lags: List of lag periods

        Returns:
            DataFrame with lagged features
        """
        df = df.copy()

        for col in columns:
            for lag in lags:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)

        return df

    def create_moving_averages(
        self,
        df: pd.DataFrame,
        columns: List[str],
        windows: List[int] = [3, 7, 14]
    ) -> pd.DataFrame:
        """
        Create moving average features.

        Args:
            df: DataFrame with time series data
            columns: Columns to create moving averages for
            windows: Window sizes for moving averages

        Returns:
            DataFrame with moving average features
        """
        df = df.copy()

        for col in columns:
            for window in windows:
                df[f'{col}_ma{window}'] = df[col].rolling(window=window, min_periods=1).mean()

        return df

    def calculate_changes(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """
        Calculate day-over-day changes.

        Args:
            df: DataFrame with time series data
            columns: Columns to calculate changes for

        Returns:
            DataFrame with change columns
        """
        df = df.copy()

        for col in columns:
            df[f'{col}_change'] = df[col].diff()
            df[f'{col}_pct_change'] = df[col].pct_change()

        return df

    def normalize_columns(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'minmax'
    ) -> pd.DataFrame:
        """
        Normalize columns to 0-1 range or standardize.

        Args:
            df: DataFrame
            columns: Columns to normalize
            method: 'minmax' or 'standard'

        Returns:
            DataFrame with normalized columns
        """
        df = df.copy()

        for col in columns:
            if method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[f'{col}_normalized'] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df[f'{col}_normalized'] = 0
            elif method == 'standard':
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df[f'{col}_normalized'] = (df[col] - mean_val) / std_val
                else:
                    df[f'{col}_normalized'] = 0

        return df

    def prepare_analysis_dataset(
        self,
        market_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        create_features: bool = True
    ) -> pd.DataFrame:
        """
        Prepare complete dataset for analysis.

        Args:
            market_df: Market price data
            sentiment_df: Sentiment data
            create_features: Whether to create lagged and MA features

        Returns:
            Prepared DataFrame ready for analysis
        """
        # Align data
        df = self.align_data(market_df, sentiment_df)

        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)

        # Create additional features if requested
        if create_features:
            # Calculate changes
            df = self.calculate_changes(df, ['probability', 'avg_sentiment'])

            # Create moving averages
            df = self.create_moving_averages(
                df,
                ['probability', 'avg_sentiment'],
                windows=[3, 7]
            )

            # Create lagged features
            df = self.create_lagged_features(
                df,
                ['avg_sentiment'],
                lags=[1, 2, 3]
            )

        # Fill NaN values
        df = df.fillna(method='bfill').fillna(0)

        print(f"Prepared dataset with {len(df)} rows and {len(df.columns)} columns")

        return df

    def split_train_test(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets (chronologically).

        Args:
            df: DataFrame to split
            test_size: Fraction of data for testing

        Returns:
            Tuple of (train_df, test_df)
        """
        split_idx = int(len(df) * (1 - test_size))

        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        print(f"Train set: {len(train_df)} rows")
        print(f"Test set: {len(test_df)} rows")

        return train_df, test_df

    def get_summary_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get summary statistics for the dataset."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        summary = df[numeric_cols].describe().T
        summary['missing'] = df[numeric_cols].isnull().sum()
        summary['missing_pct'] = (summary['missing'] / len(df) * 100).round(2)

        return summary


def load_and_combine_data(
    market_file: str,
    sentiment_file: Optional[str] = None,
    create_sample_sentiment: bool = True
) -> pd.DataFrame:
    """
    Load and combine market and sentiment data.

    Args:
        market_file: Path to market data CSV
        sentiment_file: Path to sentiment data CSV (optional)
        create_sample_sentiment: Create sample sentiment data if file not provided

    Returns:
        Combined DataFrame
    """
    processor = DataProcessor()

    # Load market data
    print(f"Loading market data from {market_file}...")
    market_df = pd.read_csv(market_file)

    # Load or create sentiment data
    if sentiment_file:
        print(f"Loading sentiment data from {sentiment_file}...")
        sentiment_df = pd.read_csv(sentiment_file)
    elif create_sample_sentiment:
        print("Creating sample sentiment data...")
        from sentiment_analyzer import create_sample_text_data, SentimentAnalyzer

        # Get market name
        market_name = market_df['market_name'].iloc[0] if 'market_name' in market_df.columns else "Sample Market"

        # Create sample text data
        text_df = create_sample_text_data(market_name, days=len(market_df))

        # Analyze sentiment
        analyzer = SentimentAnalyzer()
        sentiment_df = analyzer.analyze_dataframe(text_df)
    else:
        print("No sentiment data provided")
        return market_df

    # Combine data
    combined_df = processor.prepare_analysis_dataset(market_df, sentiment_df)

    return combined_df


if __name__ == "__main__":
    # Example usage
    print("Testing data processor...")

    # Create sample data
    from kalshi_api import KalshiDataCollector
    from sentiment_analyzer import create_sample_text_data, SentimentAnalyzer

    collector = KalshiDataCollector()
    market_df = collector.create_sample_market_data("Sample Market", days=30)

    text_df = create_sample_text_data("Sample Market", days=30)
    analyzer = SentimentAnalyzer()
    sentiment_df = analyzer.analyze_dataframe(text_df)

    # Process and combine
    processor = DataProcessor()
    combined_df = processor.prepare_analysis_dataset(market_df, sentiment_df)

    print("\nCombined dataset:")
    print(combined_df.head())
    print("\nColumns:", combined_df.columns.tolist())
    print("\nSummary statistics:")
    print(processor.get_summary_statistics(combined_df))
