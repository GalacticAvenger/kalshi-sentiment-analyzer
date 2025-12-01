#!/usr/bin/env python3
"""
Test script to verify installation and dependencies.

Run this after installing requirements to ensure everything works.
"""

import sys


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")

    required_packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib.pyplot'),
        ('seaborn', 'seaborn'),
        ('scipy', 'scipy.stats'),
        ('sklearn', 'sklearn.linear_model'),
        ('transformers', 'transformers'),
        ('torch', 'torch'),
    ]

    failed = []

    for name, module in required_packages:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name}: {e}")
            failed.append(name)

    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        print("Run: pip install -r requirements.txt")
        return False

    print("\n✅ All required packages imported successfully!")
    return True


def test_modules():
    """Test that custom modules can be imported."""
    print("\nTesting custom modules...")

    sys.path.insert(0, 'src')

    modules = [
        'kalshi_api',
        'sentiment_analyzer',
        'data_processor',
        'statistical_analysis',
        'visualizations'
    ]

    failed = []

    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module}: {e}")
            failed.append(module)

    if failed:
        print(f"\n❌ Failed to import modules: {', '.join(failed)}")
        return False

    print("\n✅ All custom modules imported successfully!")
    return True


def test_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")

    try:
        # Test data collection
        print("  Testing data collection...")
        from kalshi_api import KalshiDataCollector
        collector = KalshiDataCollector()
        market_df = collector.create_sample_market_data("Test Market", days=5)
        assert len(market_df) == 5
        print("    ✓ Data collection works")

        # Test sentiment analysis
        print("  Testing sentiment analysis...")
        from sentiment_analyzer import SentimentAnalyzer, create_sample_text_data
        text_df = create_sample_text_data("Test Market", days=5)
        assert len(text_df) > 0
        print("    ✓ Sample text generation works")

        # Test data processing
        print("  Testing data processing...")
        from data_processor import DataProcessor
        processor = DataProcessor()
        daily_sentiment = processor.aggregate_sentiment_by_date(text_df)
        assert len(daily_sentiment) > 0
        print("    ✓ Data processing works")

        # Test statistical analysis
        print("  Testing statistical analysis...")
        from statistical_analysis import StatisticalAnalyzer
        stat_analyzer = StatisticalAnalyzer()
        correlation = stat_analyzer.calculate_correlation(market_df, 'probability', 'volume')
        assert 'correlation' in correlation
        print("    ✓ Statistical analysis works")

        print("\n✅ All functionality tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sentiment_model():
    """Test sentiment model loading."""
    print("\nTesting sentiment model loading...")
    print("(This may take a minute on first run to download model)")

    try:
        from sentiment_analyzer import SentimentAnalyzer

        analyzer = SentimentAnalyzer()

        # Test simple sentiment
        result = analyzer.analyze_text("This is a great project!")

        assert 'normalized_score' in result
        print(f"  ✓ Model loaded and working")
        print(f"    Test sentiment: {result['normalized_score']:.3f}")

        print("\n✅ Sentiment model test passed!")
        return True

    except Exception as e:
        print(f"\n❌ Sentiment model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("KALSHI SENTIMENT ANALYZER - INSTALLATION TEST")
    print("=" * 70)
    print()

    results = []

    # Test imports
    results.append(("Package Imports", test_imports()))

    # Test custom modules
    results.append(("Custom Modules", test_modules()))

    # Test functionality
    results.append(("Functionality", test_functionality()))

    # Test sentiment model (optional, takes time)
    try_model = input("\nTest sentiment model loading? (takes ~1 min) [y/N]: ").lower()
    if try_model == 'y':
        results.append(("Sentiment Model", test_sentiment_model()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<50} {status}")

    all_passed = all(result[1] for result in results)

    print("=" * 70)

    if all_passed:
        print("\n🎉 All tests passed! Your installation is ready.")
        print("\nNext steps:")
        print("  1. Run the analysis: python run_analysis.py")
        print("  2. Or open the notebook: jupyter notebook notebooks/analysis.ipynb")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("   Try: pip install -r requirements.txt")

    print()


if __name__ == "__main__":
    main()
