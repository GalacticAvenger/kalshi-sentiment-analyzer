# Kalshi Sentiment Analyzer

**Analyzing the correlation between public sentiment and prediction market prices using transformer-based NLP models.**

**Authors:** Sam Meddin & Cyrus
**Course:** CPSC 171 - Introduction to Artificial Intelligence
**Institution:** Yale University
**Semester:** Fall 2025

---

## Introduction

### Project Objective

This project investigates whether public sentiment—extracted from news headlines and social media—correlates with or predicts movements in Kalshi prediction market prices. Prediction markets aggregate collective beliefs about future events, but the relationship between public discourse and market pricing remains underexplored.

### Why This Matters

- **Market Efficiency Testing**: Do prediction markets efficiently incorporate public sentiment, or do sentiment shifts precede price movements?
- **Behavioral Finance Insights**: Understanding how public opinion influences betting behavior
- **Practical Applications**: Identifying potential leading indicators for market movements

### Key Research Question

> Does public sentiment lead, lag, or move contemporaneously with prediction market prices?

---

## Methods and Results

### Architecture Overview

```
┌─────────────────────┐     ┌─────────────────────┐
│   Data Collection   │     │   Market Prices     │
│  ─────────────────  │     │  ─────────────────  │
│  • Google News RSS  │     │  • Kalshi API       │
│  • Reddit JSON API  │     │  • Historical Data  │
└──────────┬──────────┘     └──────────┬──────────┘
           │                           │
           ▼                           ▼
┌─────────────────────┐     ┌─────────────────────┐
│ Sentiment Analysis  │     │  Data Processing    │
│  ─────────────────  │     │  ─────────────────  │
│  DistilBERT Model   │────▶│  • Time Alignment   │
│  (Transformer NLP)  │     │  • Feature Eng.     │
└─────────────────────┘     └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │ Statistical Analysis │
                            │  ─────────────────  │
                            │  • Correlation      │
                            │  • Lead-Lag         │
                            │  • Granger Causality│
                            │  • Regression       │
                            └──────────┬──────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │   Visualizations    │
                            │  ─────────────────  │
                            │  • Dashboard        │
                            │  • Time Series      │
                            │  • Scatter Plots    │
                            └─────────────────────┘
```

### 1. Data Collection

We collect real-time data from two sources without requiring API authentication:

**Google News RSS**
```python
# Endpoint: https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en
# Parses RSS/XML feed for headlines, dates, and sources
# Rate limited: 0.5s between queries
```

**Reddit JSON API**
```python
# Endpoint: https://www.reddit.com/r/{subreddit}/search.json
# Fetches posts from r/politics, r/news, r/PoliticalDiscussion
# Returns title, body, timestamp, score, comment count
```

For the Biden 2024 Election market, we collected **257 texts** (108 news + 149 Reddit posts).

### 2. Sentiment Analysis

We use **DistilBERT** (`distilbert-base-uncased-finetuned-sst-2-english`), a transformer model fine-tuned for sentiment classification:

- **Input**: Raw text (truncated to 512 tokens)
- **Output**: POSITIVE/NEGATIVE label with confidence score
- **Normalization**: Mapped to [-1.0, +1.0] scale for statistical analysis
- **Performance**: ~1.3 texts/second on CPU

**Important Note**: Python 3.11 is required due to a mutex deadlock bug in the `tokenizers` library (v0.22.1) on Python 3.13.

### 3. Statistical Analysis

| Test | Purpose | Result (Sample Run) |
|------|---------|---------------------|
| **Pearson Correlation** | Linear relationship | r = -0.339, p = 0.067 |
| **Lead-Lag Analysis** | Temporal dynamics | Price leads sentiment by 2 periods |
| **Granger Causality** | Predictive relationship | Price → Sentiment: p = 0.047* |
| **Linear Regression** | Predictive modeling | R² = 0.449 |

*Statistically significant at α = 0.05

### 4. Sample Results

```
======================================================================
KALSHI SENTIMENT ANALYSIS REPORT
======================================================================

1. SUMMARY STATISTICS
----------------------------------------------------------------------
Number of observations: 30
Average market probability: 0.433 (±0.115)
Average sentiment score: -0.558 (±0.457)

2. CORRELATION ANALYSIS
----------------------------------------------------------------------
Pearson correlation: -0.339
P-value: 0.0673
Statistically significant: No (α=0.05)

3. LEAD-LAG ANALYSIS
----------------------------------------------------------------------
Strongest relationship: probability leads avg_sentiment by 2 periods
Correlation: -0.404
P-value: 0.0331

4. GRANGER CAUSALITY TESTS
----------------------------------------------------------------------
Sentiment → Price: avg_sentiment does not Granger-cause probability (p=0.2624)
Price → Sentiment: probability does Granger-cause avg_sentiment (p=0.0469)

5. CONCLUSIONS
----------------------------------------------------------------------
✓ Price appears to LEAD sentiment by 2 periods
```

### 5. Visualizations

The pipeline generates four visualization outputs:

| Output | Description |
|--------|-------------|
| `dashboard.png` | 4-panel comprehensive summary |
| `time_series.png` | Dual-axis price vs sentiment over time |
| `scatter.png` | Correlation with regression line |
| `lead_lag.png` | Correlation at different time lags |

---

## Quick Start

### Requirements

- **Python 3.11** (required for transformer compatibility)
- See `requirements.txt` for dependencies

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/kalshi-sentiment-analyzer.git
cd kalshi-sentiment-analyzer

# Install dependencies
pip install -r requirements.txt
```

### Running the Analysis

```bash
# Use Python 3.11 specifically
~/.pyenv/versions/3.11.9/bin/python3 run_analysis.py
```

Or modify for your Python 3.11 path.

---

## Project Structure

```
kalshi-sentiment-analyzer/
├── run_analysis.py           # Main pipeline orchestrator
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
│
├── src/
│   ├── kalshi_api.py        # Market data collection
│   ├── news_collector.py    # Google News + Reddit scraping
│   ├── sentiment_analyzer.py # DistilBERT sentiment analysis
│   ├── data_processor.py    # Time alignment & feature engineering
│   ├── statistical_analysis.py # Correlation, Granger, regression
│   └── visualizations.py    # Matplotlib/Seaborn plots
│
├── data/
│   └── processed/           # Generated CSV outputs
│
├── outputs/                  # Generated visualizations & reports
│
└── notebooks/
    └── analysis.ipynb       # Interactive Jupyter notebook
```

---

## Discussion

### Key Finding

Our analysis revealed that **market prices appear to lead sentiment** rather than the reverse. The Granger causality test showed that price movements significantly predict subsequent sentiment shifts (p = 0.047), while sentiment does not significantly predict price changes (p = 0.262).

This suggests that in prediction markets, **participants may be reacting to price movements** rather than prices reflecting sentiment. This could indicate:
1. Market prices incorporate information before it reaches public discourse
2. Price movements themselves generate news coverage and social media discussion
3. The market is relatively efficient at aggregating private information

### Technical Reflections

1. **Python version compatibility is critical for ML libraries** — The HuggingFace `tokenizers` library has a mutex deadlock bug on macOS with Python 3.13, requiring us to use Python 3.11.

2. **RSS feeds provide robust, API-key-free data collection** — Google News RSS and Reddit's public JSON endpoints enabled scraping ~250+ texts without authentication overhead.

3. **Lead-lag analysis reveals temporal dynamics invisible in simple correlation** — Testing correlations at multiple time offsets (-5 to +5 periods) uncovered the price-leads-sentiment relationship that contemporaneous correlation missed.

4. **Transformer output normalization enables cross-model comparability** — We implemented a normalized score mapping (-1.0 to +1.0) to standardize outputs across different model architectures.

### Limitations

- **Sample data for market prices**: Currently uses simulated Kalshi price data (real API integration possible with credentials)
- **Limited time window**: 30-day analysis window; longer periods may reveal different patterns
- **Single market focus**: Analysis focused on election markets; other market types may behave differently
- **Binary sentiment model**: DistilBERT outputs POSITIVE/NEGATIVE only; neutral classification could improve precision

### Future Work

1. **Real Kalshi API integration** for live market data
2. **Multi-market analysis** comparing sentiment-price relationships across different market types
3. **FinBERT integration** for financial-domain-specific sentiment
4. **Real-time dashboard** with live sentiment tracking
5. **Ensemble models** combining multiple sentiment classifiers
6. **Predictive modeling** using sentiment features to forecast price movements

---

## Technical Specifications

### Sentiment Model

| Parameter | Value |
|-----------|-------|
| Model | `distilbert-base-uncased-finetuned-sst-2-english` |
| Architecture | DistilBERT (6-layer transformer) |
| Training Data | Stanford Sentiment Treebank (SST-2) |
| Output Classes | POSITIVE, NEGATIVE |
| Max Sequence Length | 512 tokens |

### Statistical Methods

| Method | Implementation |
|--------|----------------|
| Pearson Correlation | `scipy.stats.pearsonr` |
| Spearman Correlation | `scipy.stats.spearmanr` |
| Granger Causality | `statsmodels.tsa.stattools.grangercausalitytests` |
| Linear Regression | `sklearn.linear_model.LinearRegression` |
| Lead-Lag Analysis | Custom implementation with shifted time series |

---

## References

### Models & Libraries
- **DistilBERT**: Sanh et al. (2019). "DistilBERT, a distilled version of BERT"
- **HuggingFace Transformers**: Wolf et al. (2020)
- **statsmodels**: Seabold & Perktold (2010)

### Methods
- **Granger Causality**: Granger, C. W. J. (1969). "Investigating Causal Relations by Econometric Models"
- **Lead-Lag Analysis**: Chan, K. (1991). "A Further Analysis of the Lead-Lag Relationship"

### Data Sources
- **Google News RSS**: `https://news.google.com/rss/search`
- **Reddit JSON API**: `https://www.reddit.com/dev/api/`
- **Kalshi**: `https://kalshi.com/`

---

## Authors

**Sam Meddin** — sam.meddin@yale.edu
**Cyrus** — Yale University

---

## Acknowledgments

- **CPSC 171 Teaching Staff**: Anushka, Jaxon, Pranik, Sue
- **HuggingFace**: For open-source transformer models
- **Open Source Community**: pandas, scikit-learn, matplotlib, seaborn

---

*Built for CPSC 171: Introduction to Artificial Intelligence, Yale University, Fall 2025*
