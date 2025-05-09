import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go # For better plotting

# --- Polygon.io API Configuration ---
API_KEY = os.getenv('POLYGON_API_KEY_STREAMLIT', 'POLYGON_API_KEY') # Use a different env var or enter here

# --- Helper Functions (Data Fetching - Simplified for Brevity) ---
def fetch_stock_data(ticker, date_from, date_to):
    """Fetches historical stock price data from Polygon.io."""
    if API_KEY == 'POLYGON_API_KEY':
        st.error("Polygon API key not configured.")
        return pd.DataFrame()
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{date_from}/{date_to}?adjusted=true&sort=asc&apiKey={API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if 'results' in data and data['results']:
            df = pd.DataFrame(data['results'])
            df['t'] = pd.to_datetime(df['t'], unit='ms')
            df = df.rename(columns={'t': 'Date', 'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})
            df = df.set_index('Date')
            return df[['Close']] # Keep it simple for now
        else:
            st.warning(f"No data found for {ticker} in the given range.")
            return pd.DataFrame()
    except requests.RequestException as e:
        st.error(f"Error fetching stock data: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return pd.DataFrame()


# (Your fetch_news_sentiment function can be included here)
# Placeholder for other data fetching (fundamentals, macro, etc.)
# For this demo, many will be simulated by sliders.

def calculate_baseline_projection(historical_prices, future_days):
    """Calculates a very simple baseline projection (e.g., last price flat or simple trend)."""
    if historical_prices.empty:
        return pd.Series()
    last_price = historical_prices['Close'].iloc[-1]
    # Simplistic: project last price forward or a slight trend
    # A real model would be more complex (e.g., ARIMA, Prophet)
    projected_dates = pd.date_range(start=historical_prices.index[-1] + timedelta(days=1), periods=future_days)
    # Simplistic: carry forward last price
    # projected_prices = pd.Series([last_price] * future_days, index=projected_dates)

    # Slightly less simplistic: extrapolate last N days trend (linear regression)
    if len(historical_prices) > 5: # Need a few points for trend
        x = np.arange(len(historical_prices))
        y = historical_prices['Close'].values
        coeffs = np.polyfit(x[-5:], y[-5:], 1) # Trend of last 5 days
        slope = coeffs[0]
        last_val_for_trend = y[-1]
        projected_values = [last_val_for_trend + slope * i for i in range(1, future_days + 1)]
        projected_prices = pd.Series(projected_values, index=projected_dates)

    else: # Fallback to flat
        projected_prices = pd.Series([last_price] * future_days, index=projected_dates)

    return projected_prices


def get_future_days(horizon_str):
    if horizon_str == '3 Months': return 63 # Approx trading days
    if horizon_str == '6 Months': return 126
    if horizon_str == '1 Year': return 252
    if horizon_str == '5 Years': return 252 * 5
    return 30 # Default

# --- Main Streamlit App ---
st.set_page_config(layout="wide")
st.title("Stock Price Scenario Simulator")

st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Stock Ticker (e.g., AAPL)", "AAPL").upper()
horizon = st.sidebar.selectbox("Prediction Horizon", ['3 Months', '6 Months', '1 Year', '5 Years'])
fetch_button = st.sidebar.button("Fetch Data & Analyze")

# --- Factor Sliders ---
st.sidebar.markdown("---")
st.sidebar.subheader("Factor Adjustments (Illustrative)")

# I. Fundamental Factors
with st.sidebar.expander("I. Fundamental Factors"):
    st.markdown("**Earnings & Profitability**")
    eps_growth_outlook = st.slider("EPS Growth Outlook Score", -1.0, 1.0, 0.0, 0.1, help="(-1 Negative, 0 Neutral, 1 Positive)")
    revenue_growth_outlook = st.slider("Revenue Growth Outlook Score", -1.0, 1.0, 0.0, 0.1)
    profit_margin_outlook = st.slider("Profit Margin Outlook Score", -1.0, 1.0, 0.0, 0.1)
    # ... many more fundamental sliders ...
    st.markdown("**Valuation**")
    pe_ratio_sentiment = st.slider("P/E Ratio (vs. Historical/Industry) Score", -1.0, 1.0, 0.0, 0.1, help="-1 Overvalued, 1 Undervalued")
    st.markdown("**Management & Governance**")
    management_quality = st.slider("Management Quality Score", -1.0, 1.0, 0.0, 0.1)

# II. Macroeconomic Factors
with st.sidebar.expander("II. Macroeconomic Factors"):
    interest_rate_impact = st.slider("Interest Rate Impact Score", -1.0, 1.0, 0.0, 0.1, help="-1 Higher rates hurt, 1 Lower rates help")
    inflation_impact = st.slider("Inflation Impact Score", -1.0, 1.0, 0.0, 0.1)
    gdp_growth_impact = st.slider("GDP Growth Impact Score", -1.0, 1.0, 0.0, 0.1)
    # ... more macro sliders ...

# III. Market Sentiment & Behavioral Factors
with st.sidebar.expander("III. Market Sentiment & Behavioral Factors"):
    # news_sentiment_score = fetch_news_sentiment(ticker) # You could try to use your function here
    # For demo, a slider is more direct for manual "what-if"
    news_sentiment_manual = st.slider("Overall News Sentiment Score", -1.0, 1.0, 0.0, 0.1)
    social_media_buzz = st.slider("Social Media Buzz Score", -1.0, 1.0, 0.0, 0.1)
    analyst_ratings_outlook = st.slider("Analyst Ratings Outlook Score", -1.0, 1.0, 0.0, 0.1)
    market_volatility_vix = st.slider("Market Volatility (VIX) Impact", -1.0, 1.0, 0.0, 0.1, help="-1 High VIX is negative, 1 Low VIX positive (oversimplified)")

# IV. Technical Factors
with st.sidebar.expander("IV. Technical Factors"):
    trend_strength = st.slider("Current Trend Strength (Technical Score)", -1.0, 1.0, 0.0, 0.1, help="-1 Strong Downtrend, 1 Strong Uptrend")
    rsi_level_score = st.slider("RSI Level Score", -1.0, 1.0, 0.0, 0.1, help="-1 Overbought, 1 Oversold")

# --- Display Area ---
if fetch_button and ticker:
    st.header(f"Analysis for {ticker} - {horizon}")

    # 1. Fetch Historical Data
    today = datetime.now()
    # Fetch more historical data to have context for the selected horizon
    if horizon == '5 Years':
        date_from_hist = (today - timedelta(days=5*365 + 2*365)).strftime('%Y-%m-%d') # ~7 years for 5 yr proj
    elif horizon == '1 Year':
        date_from_hist = (today - timedelta(days=365 + 365)).strftime('%Y-%m-%d') # ~2 years for 1 yr proj
    else:
        date_from_hist = (today - timedelta(days=365)).strftime('%Y-%m-%d') # ~1 year for shorter proj

    date_to_hist = today.strftime('%Y-%m-%d')
    historical_data = fetch_stock_data(ticker, date_from_hist, date_to_hist)

    if not historical_data.empty:
        # 2. Calculate Baseline Projection
        future_days_to_project = get_future_days(horizon)
        baseline_projection = calculate_baseline_projection(historical_data, future_days_to_project)

        # 3. Calculate Adjustment based on Sliders (VERY SIMPLIFIED - ILLUSTRATIVE WEIGHTS)
        # These weights are COMPLETELY ARBITRARY for demonstration.
        # In a real model, these would be learned or carefully calibrated,
        # and they would likely vary by horizon.
        adjustment_score = 0
        adjustment_score += eps_growth_outlook * 0.05        # Fundamental
        adjustment_score += revenue_growth_outlook * 0.04
        adjustment_score += profit_margin_outlook * 0.03
        adjustment_score += pe_ratio_sentiment * 0.03
        adjustment_score += management_quality * 0.05

        adjustment_score += interest_rate_impact * -0.04     # Macro (note negative for interest rate)
        adjustment_score += inflation_impact * -0.03
        adjustment_score += gdp_growth_impact * 0.05

        adjustment_score += news_sentiment_manual * 0.06     # Sentiment
        adjustment_score += social_media_buzz * 0.02
        adjustment_score += analyst_ratings_outlook * 0.03
        adjustment_score += market_volatility_vix * -0.02 # Higher VIX often negative

        adjustment_score += trend_strength * 0.07            # Technical
        adjustment_score += rsi_level_score * 0.02

        # Cap adjustment to avoid extreme scenarios in demo
        total_adjustment_multiplier = 1 + np.clip(adjustment_score, -0.5, 0.5)


        # 4. Apply Adjustment
        adjusted_projection = baseline_projection * total_adjustment_multiplier

        # 5. Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data['Close'], mode='lines', name='Historical Close'))
        fig.add_trace(go.Scatter(x=baseline_projection.index, y=baseline_projection, mode='lines', name='Baseline Projection', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=adjusted_projection.index, y=adjusted_projection, mode='lines', name=f'Adjusted Projection ({total_adjustment_multiplier-1:.1%})', line=dict(color='green')))

        fig.update_layout(title=f'{ticker} Price Scenarios ({horizon})',
                          xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Summary of Adjustments:")
        st.write(f"Cumulative Adjustment Score from Sliders: {adjustment_score:.3f}")
        st.write(f"Resulting Projection Multiplier: {total_adjustment_multiplier:.3f} (i.e., {(total_adjustment_multiplier-1)*100:.1f}% adjustment from baseline)")

    else:
        st.error(f"Could not retrieve or process data for {ticker}.")

st.sidebar.markdown("---")
st.sidebar.info("Disclaimer: This is an illustrative tool for scenario exploration, not financial advice. The model and weights are highly simplified.")