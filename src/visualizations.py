"""
Visualization module for creating charts and plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class Visualizer:
    """Create visualizations for market and sentiment analysis."""

    def __init__(self, output_dir: str = "outputs/"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = output_dir

    def plot_time_series(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        price_col: str = 'probability',
        sentiment_col: str = 'avg_sentiment',
        title: str = "Market Price vs Sentiment Over Time",
        save_name: Optional[str] = None
    ):
        """
        Plot market price and sentiment over time.

        Args:
            df: DataFrame with time series data
            date_col: Date column name
            price_col: Price/probability column name
            sentiment_col: Sentiment column name
            title: Plot title
            save_name: Filename to save (optional)
        """
        fig, ax1 = plt.subplots(figsize=(14, 6))

        # Plot price on left axis
        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Market Probability', color=color)
        ax1.plot(df[date_col], df[price_col], color=color, linewidth=2, label='Market Price', marker='o')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim([0, 1])

        # Create second y-axis for sentiment
        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('Sentiment Score', color=color)
        ax2.plot(df[date_col], df[sentiment_col], color=color, linewidth=2, label='Sentiment', marker='s', alpha=0.7)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # Title and layout
        plt.title(title, fontsize=14, fontweight='bold')
        fig.tight_layout()

        # Rotate x-axis labels
        plt.xticks(rotation=45)

        # Add grid
        ax1.grid(True, alpha=0.3)

        if save_name:
            plt.savefig(f"{self.output_dir}{save_name}", dpi=300, bbox_inches='tight')
            print(f"Saved plot to {self.output_dir}{save_name}")

        plt.show()

    def plot_scatter(
        self,
        df: pd.DataFrame,
        x_col: str = 'avg_sentiment',
        y_col: str = 'probability',
        title: str = "Sentiment vs Market Price Correlation",
        save_name: Optional[str] = None
    ):
        """
        Create scatter plot with regression line.

        Args:
            df: DataFrame with data
            x_col: X-axis column
            y_col: Y-axis column
            title: Plot title
            save_name: Filename to save (optional)
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create scatter plot with regression line
        sns.regplot(
            data=df,
            x=x_col,
            y=y_col,
            scatter_kws={'alpha': 0.6, 's': 50},
            line_kws={'color': 'red', 'linewidth': 2},
            ax=ax
        )

        # Calculate correlation
        from scipy import stats
        data = df[[x_col, y_col]].dropna()
        if len(data) > 2:
            corr, p_value = stats.pearsonr(data[x_col], data[y_col])
            ax.text(
                0.05, 0.95,
                f'Correlation: {corr:.3f}\np-value: {p_value:.4f}',
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )

        ax.set_xlabel('Sentiment Score', fontsize=12)
        ax.set_ylabel('Market Probability', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if save_name:
            plt.savefig(f"{self.output_dir}{save_name}", dpi=300, bbox_inches='tight')
            print(f"Saved plot to {self.output_dir}{save_name}")

        plt.show()

    def plot_lead_lag(
        self,
        lead_lag_df: pd.DataFrame,
        title: str = "Lead-Lag Correlation Analysis",
        save_name: Optional[str] = None
    ):
        """
        Plot lead-lag correlation results.

        Args:
            lead_lag_df: DataFrame with lead-lag results
            title: Plot title
            save_name: Filename to save (optional)
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Create bar plot
        colors = ['red' if x < 0 else 'blue' if x > 0 else 'gray'
                  for x in lead_lag_df['lag']]

        ax.bar(
            lead_lag_df['lag'],
            lead_lag_df['correlation'],
            color=colors,
            alpha=0.7,
            edgecolor='black'
        )

        # Add horizontal line at 0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # Labels and title
        ax.set_xlabel('Lag (Negative = Sentiment Leads, Positive = Price Leads)', fontsize=12)
        ax.set_ylabel('Correlation Coefficient', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Add grid
        ax.grid(True, alpha=0.3, axis='y')

        # Annotate strongest correlation
        max_idx = lead_lag_df['correlation'].abs().idxmax()
        max_row = lead_lag_df.loc[max_idx]
        ax.annotate(
            f'Strongest: {max_row["correlation"]:.3f}',
            xy=(max_row['lag'], max_row['correlation']),
            xytext=(10, 10),
            textcoords='offset points',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
        )

        if save_name:
            plt.savefig(f"{self.output_dir}{save_name}", dpi=300, bbox_inches='tight')
            print(f"Saved plot to {self.output_dir}{save_name}")

        plt.show()

    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        title: str = "Correlation Matrix",
        save_name: Optional[str] = None
    ):
        """
        Plot correlation matrix heatmap.

        Args:
            df: DataFrame with data
            columns: Columns to include (uses all numeric if None)
            title: Plot title
            save_name: Filename to save (optional)
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # Calculate correlation matrix
        corr_matrix = df[columns].corr()

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )

        ax.set_title(title, fontsize=14, fontweight='bold')

        if save_name:
            plt.savefig(f"{self.output_dir}{save_name}", dpi=300, bbox_inches='tight')
            print(f"Saved plot to {self.output_dir}{save_name}")

        plt.show()

    def plot_distribution(
        self,
        df: pd.DataFrame,
        column: str,
        title: Optional[str] = None,
        save_name: Optional[str] = None
    ):
        """
        Plot distribution of a variable.

        Args:
            df: DataFrame with data
            column: Column to plot
            title: Plot title
            save_name: Filename to save (optional)
        """
        if title is None:
            title = f"Distribution of {column}"

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax1.hist(df[column].dropna(), bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.set_xlabel(column, fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Histogram', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2.boxplot(df[column].dropna(), vert=True)
        ax2.set_ylabel(column, fontsize=12)
        ax2.set_title('Box Plot', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight='bold')
        fig.tight_layout()

        if save_name:
            plt.savefig(f"{self.output_dir}{save_name}", dpi=300, bbox_inches='tight')
            print(f"Saved plot to {self.output_dir}{save_name}")

        plt.show()

    def plot_moving_averages(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        value_col: str = 'probability',
        title: str = "Price with Moving Averages",
        save_name: Optional[str] = None
    ):
        """
        Plot value with moving averages.

        Args:
            df: DataFrame with data
            date_col: Date column
            value_col: Value column
            title: Plot title
            save_name: Filename to save (optional)
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot actual values
        ax.plot(df[date_col], df[value_col], label='Actual', linewidth=2, marker='o', markersize=4)

        # Plot moving averages if they exist
        ma_cols = [col for col in df.columns if col.startswith(f'{value_col}_ma')]

        for ma_col in ma_cols:
            window = ma_col.split('ma')[-1]
            ax.plot(df[date_col], df[ma_col], label=f'MA-{window}', linewidth=2, alpha=0.7)

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(value_col, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.xticks(rotation=45)
        fig.tight_layout()

        if save_name:
            plt.savefig(f"{self.output_dir}{save_name}", dpi=300, bbox_inches='tight')
            print(f"Saved plot to {self.output_dir}{save_name}")

        plt.show()

    def create_dashboard(
        self,
        df: pd.DataFrame,
        lead_lag_df: pd.DataFrame,
        results: Dict,
        market_name: str = "Market",
        save_name: Optional[str] = None
    ):
        """
        Create comprehensive dashboard with multiple plots.

        Args:
            df: Combined DataFrame
            lead_lag_df: Lead-lag analysis results
            results: Statistical analysis results
            market_name: Name of the market
            save_name: Filename to save (optional)
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. Time series plot
        ax1 = fig.add_subplot(gs[0, :])
        ax1_twin = ax1.twinx()

        ax1.plot(df['date'], df['probability'], color='blue', linewidth=2, label='Market Price', marker='o')
        ax1.set_ylabel('Market Probability', color='blue', fontsize=11)
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_ylim([0, 1])

        ax1_twin.plot(df['date'], df['avg_sentiment'], color='orange', linewidth=2, label='Sentiment', marker='s', alpha=0.7)
        ax1_twin.set_ylabel('Sentiment Score', color='orange', fontsize=11)
        ax1_twin.tick_params(axis='y', labelcolor='orange')
        ax1_twin.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        ax1.set_title(f'{market_name}: Price vs Sentiment Over Time', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # 2. Scatter plot
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.scatter(df['avg_sentiment'], df['probability'], alpha=0.6, s=50)

        # Add regression line
        from scipy import stats
        data = df[['avg_sentiment', 'probability']].dropna()
        if len(data) > 2:
            z = np.polyfit(data['avg_sentiment'], data['probability'], 1)
            p = np.poly1d(z)
            ax2.plot(data['avg_sentiment'], p(data['avg_sentiment']), "r-", linewidth=2)

            corr = results['correlation']['correlation']
            p_val = results['correlation']['p_value']
            ax2.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.4f}',
                    transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax2.set_xlabel('Sentiment Score', fontsize=11)
        ax2.set_ylabel('Market Probability', fontsize=11)
        ax2.set_title('Correlation Analysis', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. Lead-lag plot
        ax3 = fig.add_subplot(gs[1, 1])
        colors = ['red' if x < 0 else 'blue' if x > 0 else 'gray' for x in lead_lag_df['lag']]
        ax3.bar(lead_lag_df['lag'], lead_lag_df['correlation'], color=colors, alpha=0.7, edgecolor='black')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax3.set_xlabel('Lag', fontsize=11)
        ax3.set_ylabel('Correlation', fontsize=11)
        ax3.set_title('Lead-Lag Analysis', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Summary statistics table
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.axis('off')

        summary_text = f"""
        SUMMARY STATISTICS
        ──────────────────────────────────
        Observations: {results['summary_stats']['n_observations']}

        Market Price:
          Mean: {results['summary_stats']['price_mean']:.3f}
          Std: {results['summary_stats']['price_std']:.3f}

        Sentiment:
          Mean: {results['summary_stats']['sentiment_mean']:.3f}
          Std: {results['summary_stats']['sentiment_std']:.3f}

        Correlation: {results['correlation']['correlation']:.3f}
        Significant: {'Yes' if results['correlation']['significant'] else 'No'}
        """

        ax4.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')

        # 5. Key findings
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')

        strongest = results['strongest_relationship']
        findings_text = f"""
        KEY FINDINGS
        ──────────────────────────────────
        Strongest Relationship:
          {strongest['interpretation']}
          Correlation: {strongest['correlation']:.3f}

        Granger Causality:
        """

        if 'error' not in results['sentiment_causes_price']:
            sent_causes = results['sentiment_causes_price']
            findings_text += f"\n  Sentiment → Price: "
            findings_text += "Yes" if sent_causes['col1_causes_col2'] else "No"

        findings_text += f"\n\nRegression R²: "
        if 'error' not in results['regression']:
            findings_text += f"{results['regression']['r2']:.3f}"
        else:
            findings_text += "N/A"

        ax5.text(0.1, 0.5, findings_text, fontsize=10, family='monospace',
                verticalalignment='center')

        # Overall title
        fig.suptitle(f'Kalshi Sentiment Analysis Dashboard: {market_name}',
                    fontsize=16, fontweight='bold', y=0.98)

        if save_name:
            plt.savefig(f"{self.output_dir}{save_name}", dpi=300, bbox_inches='tight')
            print(f"Saved dashboard to {self.output_dir}{save_name}")

        plt.show()


if __name__ == "__main__":
    # Example usage
    print("Testing visualizations...")

    # Create sample data
    from data_processor import DataProcessor
    from kalshi_api import KalshiDataCollector
    from sentiment_analyzer import create_sample_text_data, SentimentAnalyzer
    from statistical_analysis import StatisticalAnalyzer

    collector = KalshiDataCollector()
    market_df = collector.create_sample_market_data("Sample Market", days=30)

    text_df = create_sample_text_data("Sample Market", days=30)
    analyzer = SentimentAnalyzer()
    sentiment_df = analyzer.analyze_dataframe(text_df)

    processor = DataProcessor()
    combined_df = processor.prepare_analysis_dataset(market_df, sentiment_df)

    stat_analyzer = StatisticalAnalyzer()
    results = stat_analyzer.calculate_metrics(combined_df)

    # Create visualizations
    viz = Visualizer()

    print("\n1. Time series plot")
    viz.plot_time_series(combined_df)

    print("\n2. Scatter plot")
    viz.plot_scatter(combined_df)

    print("\n3. Lead-lag plot")
    viz.plot_lead_lag(results['lead_lag'])

    print("\n4. Dashboard")
    viz.create_dashboard(combined_df, results['lead_lag'], results, "Sample Market")
