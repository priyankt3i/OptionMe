import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta

import os
import requests
import pandas as pd

def fetch_news_sentiment(ticker: str) -> float:
    """
    Fetch recent news sentiment score for the ticker using Polygon.io's Ticker News API with LLM insights.
    Returns a sentiment score between -1 (very negative) and 1 (very positive).
    """
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key or api_key == 'your_api_key_here':
        return 0.0

    url = f'https://api.polygon.io/v2/reference/news?ticker={ticker}&limit=50&sort=published_utc&order=desc&apiKey={api_key}'

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if 'results' not in data or len(data['results']) == 0:
            return 0.0

        # Aggregate sentiment scores from LLM insights in news articles
        sentiments = []
        for article in data['results']:
            if 'sentiment' in article and 'overall' in article['sentiment']:
                sentiments.append(article['sentiment']['overall'])
        if sentiments:
            avg_sentiment = sum(sentiments) / len(sentiments)
            # Normalize to -1 to 1 scale if needed (assuming API returns in this range)
            return avg_sentiment
        else:
            return 0.0
    except requests.RequestException:
        return 0.0

def fetch_social_media_trends(ticker: str) -> float:
    """
    Placeholder function to fetch social media trend score for the ticker.
    Returns a score between -1 and 1.
    """
    # For demonstration, return a neutral trend
    return 0.0

def calculate_t3i_growth_rates(historical_prices: pd.Series, ticker: str) -> dict:
    """
    Calculate growth rates based on historical prices, recent news sentiment,
    social media trends, and other factors.
    Returns dict with keys as periods and values as growth multipliers.
    """
    # Base growth rates from historical data (simple CAGR over periods)
    periods = {
        '3 Months': 63,
        '6 Months': 126,
        '1 Year': 252,
        '5 Years': 1260
    }
    growth_rates = {}
    for period_name, days in periods.items():
        if len(historical_prices) > days:
            past_price = historical_prices.iloc[-days]
            current_price = historical_prices.iloc[-1]
            base_growth = current_price / past_price
        else:
            base_growth = 1.0

        growth_rates[period_name] = base_growth

    # Fetch external factors
    news_sentiment = fetch_news_sentiment(ticker)
    social_trend = fetch_social_media_trends(ticker)

    # Combine factors into adjustment multiplier
    adjustment = 1 + 0.05 * news_sentiment + 0.03 * social_trend

    # Adjust growth rates
    for period in growth_rates:
        growth_rates[period] *= adjustment

    return growth_rates
