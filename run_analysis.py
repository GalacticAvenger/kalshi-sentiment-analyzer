#!/usr/bin/env python3
"""
Main script to run Kalshi sentiment analysis.

Usage:
    python run_analysis.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Load environment variables from .env file if it exists
from dotenv import load_dotenv
load_dotenv()

from kalshi_api import KalshiDataCollector
from sentiment_analyzer import SentimentAnalyzer, APISentimentAnalyzer, create_sample_text_data
from news_collector import collect_real_sentiment_data
from data_processor import DataProcessor
from statistical_analysis import StatisticalAnalyzer
from visualizations import Visualizer


def main():
    """Run complete analysis pipeline."""

    print("=" * 70)
    print("KALSHI SENTIMENT ANALYZER")
    print("=" * 70)

    # Configuration
    USE_REAL_DATA = True  # Set to False to use sample data
    market_name = "Will Biden win the 2024 Presidential Election?"

    # Step 1: Collect market data
    print("\n[Step 1/6] Collecting market data...")
    collector = KalshiDataCollector()

    # Using sample market price data (could integrate real Kalshi API later)
    market_df = collector.create_sample_market_data(market_name, days=30)
    print(f"✓ Collected {len(market_df)} days of market data")

    # Step 2: Collect text data
    print("\n[Step 2/6] Collecting text data for sentiment analysis...")

    if USE_REAL_DATA:
        print("Fetching REAL news and social media data...")
        text_df = collect_real_sentiment_data(market_name, days_back=30, include_reddit=True)

        if text_df.empty:
            print("Warning: No real data collected, falling back to sample data")
            text_df = create_sample_text_data(market_name, days=30)
    else:
        text_df = create_sample_text_data(market_name, days=30)

    print(f"✓ Collected {len(text_df)} text samples")

    # Step 3: Analyze sentiment
    print("\n[Step 3/6] Analyzing sentiment...")

    # Use transformer-based sentiment analysis (DistilBERT)
    # Note: Requires Python 3.11 due to tokenizers compatibility issues with 3.13
    analyzer = SentimentAnalyzer(use_simple=False)
    sentiment_df = analyzer.analyze_dataframe(text_df)
    print(f"✓ Analyzed sentiment for {len(sentiment_df)} texts")

    # Step 4: Process and combine data
    print("\n[Step 4/6] Processing and aligning data...")
    processor = DataProcessor()
    combined_df = processor.prepare_analysis_dataset(market_df, sentiment_df)

    print(f"✓ Created combined dataset with {len(combined_df)} rows")

    # Save combined data
    os.makedirs('data/processed', exist_ok=True)
    combined_df.to_csv('data/processed/combined_analysis.csv', index=False)
    print(f"✓ Saved to data/processed/combined_analysis.csv")

    # Also save raw sentiment data
    sentiment_df.to_csv('data/processed/sentiment_data.csv', index=False)
    print(f"✓ Saved sentiment data to data/processed/sentiment_data.csv")

    # Step 5: Statistical analysis
    print("\n[Step 5/6] Running statistical analysis...")
    stat_analyzer = StatisticalAnalyzer()
    results = stat_analyzer.calculate_metrics(combined_df)

    print(f"✓ Completed statistical analysis")

    # Generate report
    report = stat_analyzer.generate_report(results)

    # Save report
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/analysis_report.txt', 'w') as f:
        f.write(report)

    print(f"✓ Saved report to outputs/analysis_report.txt")

    # Display report
    print("\n" + report)

    # Step 6: Create visualizations
    print("\n[Step 6/6] Creating visualizations...")
    viz = Visualizer(output_dir='outputs/')

    # Time series
    print("  - Time series plot...")
    viz.plot_time_series(
        combined_df,
        title=f"{market_name}: Price vs Sentiment",
        save_name="time_series.png"
    )

    # Scatter
    print("  - Scatter plot...")
    viz.plot_scatter(
        combined_df,
        title="Sentiment vs Market Price Correlation",
        save_name="scatter.png"
    )

    # Lead-lag
    print("  - Lead-lag analysis plot...")
    viz.plot_lead_lag(
        results['lead_lag'],
        title="Lead-Lag Correlation Analysis",
        save_name="lead_lag.png"
    )

    # Dashboard
    print("  - Creating dashboard...")
    viz.create_dashboard(
        combined_df,
        results['lead_lag'],
        results,
        market_name=market_name,
        save_name="dashboard.png"
    )

    print(f"\n✓ All visualizations saved to outputs/")

    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - data/processed/combined_analysis.csv")
    print("  - data/processed/sentiment_data.csv")
    print("  - outputs/analysis_report.txt")
    print("  - outputs/time_series.png")
    print("  - outputs/scatter.png")
    print("  - outputs/lead_lag.png")
    print("  - outputs/dashboard.png")
    print("\nKey Findings:")

    corr = results['correlation']
    strongest = results['strongest_relationship']

    if corr['significant']:
        direction = "positive" if corr['correlation'] > 0 else "negative"
        print(f"  • Significant {direction} correlation: {corr['correlation']:.3f}")
    else:
        print(f"  • No significant correlation found")

    print(f"  • {strongest['interpretation']}")

    if strongest['lag'] < 0:
        print(f"  • Sentiment appears to LEAD price movements")
    elif strongest['lag'] > 0:
        print(f"  • Price appears to LEAD sentiment")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
