# MarketMind – Generative AI Powered Sales & Marketing Intelligence Platform

A minimal Flask-based prototype to upload CSV data, run basic analytics, train a simple predictive model, and generate marketing recommendations using a Generative AI API (with an offline fallback).

## Features
- Upload CSV of sales/marketing data
- Basic analytics: total sales, conversion rate, top campaigns
- Predictive model: Logistic Regression for conversion probability (if target/feature columns exist)
- Generative insights: uses OpenAI if `OPENAI_API_KEY` is set; otherwise produces sensible rule-based insights
- Simple Bootstrap UI

## Project Structure
```
marketmind/
├─ app.py
├─ utils/
│  ├─ analytics.py
│  ├─ model.py
│  └─ generative.py
├─ templates/
│  ├─ index.html
│  └─ dashboard.html
├─ static/
│  ├─ styles.css
│  └─ sample_data/marketmind_sample.csv
└─ README.md
```

## Requirements
- Python 3.9+

Install dependencies:
```
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install minimal set directly:
```
pip install flask pandas scikit-learn openai
```

> The OpenAI package is optional. If `OPENAI_API_KEY` is not set or OpenAI call fails, the app will generate fallback insights.

## Running Locally
1. Navigate to the project directory:
```
cd marketmind
```
2. (Optional) Set your OpenAI API key:
- Windows (cmd):
```
set OPENAI_API_KEY=sk-...yourkey...
```
- PowerShell:
```
$Env:OPENAI_API_KEY="sk-...yourkey..."
```
3. Start the Flask app:
```
python app.py
```
4. Open your browser at http://127.0.0.1:5000

5. Use the provided sample dataset from the landing page or upload your own CSV. Recommended columns (best-effort mapping supported):
- sales (revenue/amount)
- converted (1/0 or true/false)
- campaign
- impressions
- clicks

## Notes
- The model requires a `converted` target and at least one feature among `sales`, `impressions`, `clicks`, `campaign`.
- If your dataset uses different column names, the app tries to auto-detect common aliases.
- For production, replace the Flask secret key and add proper validation and security hardening.

## License
MIT
