import os
from typing import Dict, List

# This module provides a simple abstraction for generating insights.
# It first tries to use OpenAI (if OPENAI_API_KEY is set), otherwise
# returns rule-based fallback insights so the app works out-of-the-box.


def _format_float(x, pct=False):
    if x is None:
        return 'N/A'
    if pct:
        return f"{x*100:.1f}%"
    return f"{x:,.2f}"


def _fallback_insights(analytics: Dict, model_info: Dict) -> List[str]:
    total_sales = analytics.get('total_sales')
    conversion_rate = analytics.get('conversion_rate')
    top_campaigns = analytics.get('top_campaigns_table') or []

    insights = []
    insights.append(
        f"Overall sales performance shows total revenue of {_format_float(total_sales)} with a conversion rate of {_format_float(conversion_rate, pct=True)}."
    )

    if top_campaigns:
        top = top_campaigns[0]
        camp_name = top.get('campaign') or next(iter(top.values()))
        insights.append(
            f"'{camp_name}' appears to be a top-performing campaign. Consider allocating more budget and replicating its creatives and targeting in similar audiences."
        )

    insights.append(
        "Focus on improving the upper funnel: increase CTR by refining ad copy and creative tests; then optimize landing pages for speed and clarity to lift conversions."
    )

    if model_info and 'metrics' in model_info:
        metrics = model_info['metrics']
        acc = metrics.get('accuracy')
        roc = metrics.get('roc_auc')
        if acc is not None:
            insights.append(f"Predictive model reached accuracy of {_format_float(acc)}; use probabilities to prioritize high-propensity leads.")
        if roc is not None:
            insights.append(f"ROC-AUC of {_format_float(roc)} indicates reasonable separability; consider feature engineering with engagement signals.")

    return insights


def generate_ai_insights(analytics: Dict, model_info: Dict) -> List[str]:
    # Try OpenAI if configured, else fallback.
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return _fallback_insights(analytics, model_info)

    try:
        # Lazy import to avoid hard dependency
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        prompt = (
            "You are a senior growth strategist. Given the following metrics, generate 4-6 concise, actionable insights "
            "covering budget allocation, targeting, creative tests, funnel optimization, and sales enablement.\n\n"
            f"Metrics: {analytics}\nModel: {model_info}\n\n"
            "Respond as a bullet list without numbering. Keep each bullet under 25 words."
        )

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You generate practical, data-driven marketing and sales recommendations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )

        text = completion.choices[0].message.content
        bullets = [b.strip("- • ") for b in text.splitlines() if b.strip()]
        return bullets[:8]
    except Exception:
        # Never fail the app if API call breaks
        return _fallback_insights(analytics, model_info)
