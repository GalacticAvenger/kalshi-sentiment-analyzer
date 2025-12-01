#!/bin/bash
# Setup script for Kalshi Sentiment Analyzer

echo "========================================="
echo "Kalshi Sentiment Analyzer - Setup"
echo "========================================="

# Check Python version
echo ""
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✓ Python $python_version detected (>= 3.8 required)"
else
    echo "✗ Python 3.8 or higher required. Found: $python_version"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the analysis:"
echo "  python run_analysis.py"
echo ""
echo "To start Jupyter notebook:"
echo "  jupyter notebook notebooks/analysis.ipynb"
echo ""
echo "========================================="
