# 🚀 Quick Start Guide

Get up and running with Kalshi Sentiment Analyzer in 5 minutes!

## Option 1: Automated Setup (Recommended)

```bash
# Navigate to project directory
cd kalshi-sentiment-analyzer

# Run setup script
./setup.sh

# Activate virtual environment
source venv/bin/activate

# Run analysis
python run_analysis.py
```

## Option 2: Manual Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the analysis
python run_analysis.py
```

## Option 3: Jupyter Notebook

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch Jupyter
jupyter notebook

# 3. Open notebooks/analysis.ipynb
```

---

## What You'll Get

After running the analysis, you'll find:

### In `outputs/` directory:
- **dashboard.png** - Comprehensive analysis dashboard
- **time_series.png** - Price vs sentiment over time
- **scatter.png** - Correlation scatter plot
- **lead_lag.png** - Lead-lag analysis chart
- **analysis_report.txt** - Detailed statistical report

### In `data/processed/` directory:
- **combined_analysis.csv** - Merged dataset for further analysis

---

## Customizing the Analysis

### Using Your Own Data

1. **Market Data** (CSV with columns: date, market_name, probability, volume)
```python
from src.data_processor import load_and_combine_data

df = load_and_combine_data(
    market_file="your_market_data.csv",
    sentiment_file="your_sentiment_data.csv"
)
```

2. **Sentiment Data** (CSV with columns: date, text, source)
```python
from src.sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()
sentiment_df = analyzer.analyze_dataframe(your_text_df)
```

### Changing Sentiment Models

```python
# Use FinBERT for financial markets
analyzer = SentimentAnalyzer(model="ProsusAI/finbert")

# Use Twitter-optimized model (default)
analyzer = SentimentAnalyzer(model="cardiffnlp/twitter-roberta-base-sentiment-latest")
```

### Analyzing Multiple Markets

```python
from src.kalshi_api import fetch_sample_markets

# Fetch multiple markets
markets = fetch_sample_markets(save_dir="data/raw/")

# Analyze each market
for market_id, market_df in markets.items():
    # Run analysis...
```

---

## Troubleshooting

### "Module not found" errors
```bash
# Make sure you're in the project directory
cd kalshi-sentiment-analyzer

# Install dependencies
pip install -r requirements.txt
```

### "CUDA/GPU not found" warnings
This is normal! The project works on CPU. GPU just makes it faster.

### "Insufficient data" errors
- Need at least 30 days of data for Granger causality test
- Some statistical tests require minimum data points

### Slow sentiment analysis
- Batch size can be reduced: `analyzer.analyze_batch(texts, batch_size=16)`
- Or use a smaller model: `SentimentAnalyzer(model="distilbert-base-uncased-finetuned-sst-2-english")`

---

## Next Steps

1. ✅ Run the default analysis
2. 📊 Review the outputs in `outputs/`
3. 📓 Explore the Jupyter notebook
4. 🔧 Customize for your own data
5. 📈 Analyze more markets

---

## Getting Help

- Check [README.md](README.md) for detailed documentation
- Review [PROGRESS_REPORT_TEMPLATE.md](PROGRESS_REPORT_TEMPLATE.md) for project context
- Look at source code comments in `src/` directory
- Examine the Jupyter notebook for examples

---

## Project Structure Reminder

```
kalshi-sentiment-analyzer/
├── run_analysis.py          # ← Start here
├── requirements.txt         # Dependencies
├── README.md               # Full documentation
├── src/                    # Source code
│   ├── kalshi_api.py
│   ├── sentiment_analyzer.py
│   ├── data_processor.py
│   ├── statistical_analysis.py
│   └── visualizations.py
├── notebooks/
│   └── analysis.ipynb      # ← Or start here
├── data/
│   ├── raw/               # Input data
│   └── processed/         # Output data
└── outputs/               # Charts and reports
```

---

**Ready to analyze some markets? Run `python run_analysis.py`!** 🚀
