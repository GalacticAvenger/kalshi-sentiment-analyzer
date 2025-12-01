"""
Statistical analysis module for correlation and causality testing.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class StatisticalAnalyzer:
    """Perform statistical analysis on market and sentiment data."""

    def __init__(self):
        """Initialize statistical analyzer."""
        pass

    def calculate_correlation(
        self,
        df: pd.DataFrame,
        col1: str,
        col2: str,
        method: str = 'pearson'
    ) -> Dict[str, float]:
        """
        Calculate correlation between two columns.

        Args:
            df: DataFrame with data
            col1: First column name
            col2: Second column name
            method: 'pearson', 'spearman', or 'kendall'

        Returns:
            Dictionary with correlation coefficient and p-value
        """
        # Remove NaN values
        data = df[[col1, col2]].dropna()

        if len(data) < 3:
            return {'correlation': 0.0, 'p_value': 1.0, 'n_samples': len(data)}

        if method == 'pearson':
            corr, p_value = stats.pearsonr(data[col1], data[col2])
        elif method == 'spearman':
            corr, p_value = stats.spearmanr(data[col1], data[col2])
        elif method == 'kendall':
            corr, p_value = stats.kendalltau(data[col1], data[col2])
        else:
            raise ValueError(f"Unknown method: {method}")

        return {
            'correlation': corr,
            'p_value': p_value,
            'n_samples': len(data),
            'significant': p_value < 0.05
        }

    def correlation_matrix(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'pearson'
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for multiple columns.

        Args:
            df: DataFrame with data
            columns: List of column names (uses all numeric if None)
            method: Correlation method

        Returns:
            Correlation matrix DataFrame
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        data = df[columns].dropna()

        if method == 'pearson':
            corr_matrix = data.corr(method='pearson')
        elif method == 'spearman':
            corr_matrix = data.corr(method='spearman')
        elif method == 'kendall':
            corr_matrix = data.corr(method='kendall')
        else:
            raise ValueError(f"Unknown method: {method}")

        return corr_matrix

    def lead_lag_analysis(
        self,
        df: pd.DataFrame,
        col1: str,
        col2: str,
        max_lag: int = 5
    ) -> pd.DataFrame:
        """
        Analyze lead-lag relationships between two variables.

        Args:
            df: DataFrame with time series data
            col1: First column (potential leading indicator)
            col2: Second column (potential lagging indicator)
            max_lag: Maximum number of lags to test

        Returns:
            DataFrame with correlations at different lags
        """
        results = []

        # Test negative lags (col1 leads col2)
        for lag in range(-max_lag, 0):
            shifted = df[[col1, col2]].copy()
            shifted[f'{col2}_shifted'] = shifted[col2].shift(-lag)

            corr_result = self.calculate_correlation(
                shifted,
                col1,
                f'{col2}_shifted'
            )

            results.append({
                'lag': lag,
                'correlation': corr_result['correlation'],
                'p_value': corr_result['p_value'],
                'interpretation': f'{col1} leads {col2} by {abs(lag)} periods'
            })

        # Test zero lag (contemporaneous)
        corr_result = self.calculate_correlation(df, col1, col2)
        results.append({
            'lag': 0,
            'correlation': corr_result['correlation'],
            'p_value': corr_result['p_value'],
            'interpretation': 'Contemporaneous correlation'
        })

        # Test positive lags (col2 leads col1)
        for lag in range(1, max_lag + 1):
            shifted = df[[col1, col2]].copy()
            shifted[f'{col1}_shifted'] = shifted[col1].shift(lag)

            corr_result = self.calculate_correlation(
                shifted,
                f'{col1}_shifted',
                col2
            )

            results.append({
                'lag': lag,
                'correlation': corr_result['correlation'],
                'p_value': corr_result['p_value'],
                'interpretation': f'{col2} leads {col1} by {lag} periods'
            })

        results_df = pd.DataFrame(results)
        return results_df

    def granger_causality_test(
        self,
        df: pd.DataFrame,
        col1: str,
        col2: str,
        max_lag: int = 3
    ) -> Dict[str, any]:
        """
        Perform Granger causality test.

        Args:
            df: DataFrame with time series data
            col1: First column
            col2: Second column
            max_lag: Maximum lag to test

        Returns:
            Dictionary with test results
        """
        try:
            from statsmodels.tsa.stattools import grangercausalitytests

            # Prepare data
            data = df[[col2, col1]].dropna()

            if len(data) < max_lag + 10:
                return {
                    'error': 'Insufficient data for Granger causality test',
                    'col1_causes_col2': None,
                    'min_p_value': None
                }

            # Test if col1 Granger-causes col2
            results = grangercausalitytests(data, max_lag, verbose=False)

            # Extract p-values
            p_values = []
            for lag in range(1, max_lag + 1):
                # Get F-test p-value
                p_value = results[lag][0]['ssr_ftest'][1]
                p_values.append(p_value)

            min_p_value = min(p_values)
            causes = min_p_value < 0.05

            return {
                'col1_causes_col2': causes,
                'min_p_value': min_p_value,
                'p_values_by_lag': {i + 1: p for i, p in enumerate(p_values)},
                'interpretation': f"{col1} {'does' if causes else 'does not'} Granger-cause {col2} (p={min_p_value:.4f})"
            }

        except Exception as e:
            return {
                'error': str(e),
                'col1_causes_col2': None,
                'min_p_value': None
            }

    def regression_analysis(
        self,
        df: pd.DataFrame,
        dependent_var: str,
        independent_vars: List[str]
    ) -> Dict[str, any]:
        """
        Perform linear regression analysis.

        Args:
            df: DataFrame with data
            dependent_var: Dependent variable (y)
            independent_vars: Independent variables (X)

        Returns:
            Dictionary with regression results
        """
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

            # Prepare data
            data = df[[dependent_var] + independent_vars].dropna()

            if len(data) < 10:
                return {'error': 'Insufficient data for regression'}

            X = data[independent_vars].values
            y = data[dependent_var].values

            # Fit model
            model = LinearRegression()
            model.fit(X, y)

            # Predictions
            y_pred = model.predict(X)

            # Calculate metrics
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)

            # Coefficients
            coefficients = {
                var: coef
                for var, coef in zip(independent_vars, model.coef_)
            }

            return {
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'intercept': model.intercept_,
                'coefficients': coefficients,
                'n_samples': len(data)
            }

        except Exception as e:
            return {'error': str(e)}

    def calculate_metrics(
        self,
        df: pd.DataFrame,
        price_col: str = 'probability',
        sentiment_col: str = 'avg_sentiment'
    ) -> Dict[str, any]:
        """
        Calculate comprehensive metrics for market-sentiment relationship.

        Args:
            df: Combined market and sentiment data
            price_col: Market price/probability column
            sentiment_col: Sentiment score column

        Returns:
            Dictionary with all metrics
        """
        results = {}

        # 1. Basic correlation
        results['correlation'] = self.calculate_correlation(
            df, price_col, sentiment_col
        )

        # 2. Lead-lag analysis
        results['lead_lag'] = self.lead_lag_analysis(
            df, sentiment_col, price_col, max_lag=5
        )

        # 3. Find strongest lead-lag relationship
        lead_lag_df = results['lead_lag']
        # Handle NaN values in correlation
        valid_corr = lead_lag_df['correlation'].dropna()
        if len(valid_corr) > 0:
            strongest_idx = valid_corr.abs().idxmax()
            strongest = lead_lag_df.loc[strongest_idx]
        else:
            # Default if all correlations are NaN
            strongest = lead_lag_df.iloc[len(lead_lag_df)//2]  # middle row (lag 0)
        results['strongest_relationship'] = {
            'lag': int(strongest['lag']),
            'correlation': strongest['correlation'],
            'p_value': strongest['p_value'],
            'interpretation': strongest['interpretation']
        }

        # 4. Granger causality (both directions)
        results['sentiment_causes_price'] = self.granger_causality_test(
            df, sentiment_col, price_col
        )

        results['price_causes_sentiment'] = self.granger_causality_test(
            df, price_col, sentiment_col
        )

        # 5. Regression analysis
        if f'{sentiment_col}_lag1' in df.columns:
            independent_vars = [sentiment_col, f'{sentiment_col}_lag1', f'{sentiment_col}_lag2']
            independent_vars = [v for v in independent_vars if v in df.columns]
        else:
            independent_vars = [sentiment_col]

        results['regression'] = self.regression_analysis(
            df, price_col, independent_vars
        )

        # 6. Summary statistics
        results['summary_stats'] = {
            'price_mean': df[price_col].mean(),
            'price_std': df[price_col].std(),
            'sentiment_mean': df[sentiment_col].mean(),
            'sentiment_std': df[sentiment_col].std(),
            'n_observations': len(df)
        }

        return results

    def generate_report(self, results: Dict[str, any]) -> str:
        """
        Generate a text report from analysis results.

        Args:
            results: Results dictionary from calculate_metrics

        Returns:
            Formatted text report
        """
        report = []
        report.append("=" * 70)
        report.append("KALSHI SENTIMENT ANALYSIS REPORT")
        report.append("=" * 70)

        # Summary statistics
        report.append("\n1. SUMMARY STATISTICS")
        report.append("-" * 70)
        stats = results['summary_stats']
        report.append(f"Number of observations: {stats['n_observations']}")
        report.append(f"Average market probability: {stats['price_mean']:.3f} (±{stats['price_std']:.3f})")
        report.append(f"Average sentiment score: {stats['sentiment_mean']:.3f} (±{stats['sentiment_std']:.3f})")

        # Correlation
        report.append("\n2. CORRELATION ANALYSIS")
        report.append("-" * 70)
        corr = results['correlation']
        report.append(f"Pearson correlation: {corr['correlation']:.3f}")
        report.append(f"P-value: {corr['p_value']:.4f}")
        report.append(f"Statistically significant: {'Yes' if corr['significant'] else 'No'} (α=0.05)")

        # Lead-lag
        report.append("\n3. LEAD-LAG ANALYSIS")
        report.append("-" * 70)
        strongest = results['strongest_relationship']
        report.append(f"Strongest relationship: {strongest['interpretation']}")
        report.append(f"Correlation: {strongest['correlation']:.3f}")
        report.append(f"P-value: {strongest['p_value']:.4f}")

        # Granger causality
        report.append("\n4. GRANGER CAUSALITY TESTS")
        report.append("-" * 70)

        sent_causes = results['sentiment_causes_price']
        if 'error' not in sent_causes:
            report.append(f"Sentiment → Price: {sent_causes['interpretation']}")
        else:
            report.append(f"Sentiment → Price: {sent_causes['error']}")

        price_causes = results['price_causes_sentiment']
        if 'error' not in price_causes:
            report.append(f"Price → Sentiment: {price_causes['interpretation']}")
        else:
            report.append(f"Price → Sentiment: {price_causes['error']}")

        # Regression
        report.append("\n5. REGRESSION ANALYSIS")
        report.append("-" * 70)
        reg = results['regression']
        if 'error' not in reg:
            report.append(f"R² score: {reg['r2']:.3f}")
            report.append(f"RMSE: {reg['rmse']:.4f}")
            report.append(f"MAE: {reg['mae']:.4f}")
            report.append("Coefficients:")
            for var, coef in reg['coefficients'].items():
                report.append(f"  {var}: {coef:.4f}")
        else:
            report.append(f"Error: {reg['error']}")

        # Conclusions
        report.append("\n6. CONCLUSIONS")
        report.append("-" * 70)

        if corr['significant']:
            if corr['correlation'] > 0:
                report.append("✓ Positive correlation between sentiment and market prices")
            else:
                report.append("✓ Negative correlation between sentiment and market prices")
        else:
            report.append("✗ No significant correlation found")

        if strongest['lag'] < 0:
            report.append(f"✓ Sentiment appears to LEAD price by {abs(strongest['lag'])} periods")
        elif strongest['lag'] > 0:
            report.append(f"✓ Price appears to LEAD sentiment by {strongest['lag']} periods")
        else:
            report.append("✓ Sentiment and price move contemporaneously")

        report.append("=" * 70)

        return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    print("Testing statistical analyzer...")

    # Create sample combined data
    from data_processor import DataProcessor
    from kalshi_api import KalshiDataCollector
    from sentiment_analyzer import create_sample_text_data, SentimentAnalyzer

    collector = KalshiDataCollector()
    market_df = collector.create_sample_market_data("Sample Market", days=30)

    text_df = create_sample_text_data("Sample Market", days=30)
    analyzer = SentimentAnalyzer()
    sentiment_df = analyzer.analyze_dataframe(text_df)

    processor = DataProcessor()
    combined_df = processor.prepare_analysis_dataset(market_df, sentiment_df)

    # Run statistical analysis
    stat_analyzer = StatisticalAnalyzer()
    results = stat_analyzer.calculate_metrics(combined_df)

    # Generate report
    report = stat_analyzer.generate_report(results)
    print("\n" + report)
