import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- Polygon.io API Configuration ---
API_KEY = os.getenv('POLYGON_API_KEY', 'YOUR_POLYGON_API_KEY') # Ensure your API key is set

# --- Constants ---
TRADING_DAYS_PER_MONTH = 21
TRADING_DAYS_PER_YEAR = 252

HORIZON_DAYS = {
    '3 Months': TRADING_DAYS_PER_MONTH * 3,
    '6 Months': TRADING_DAYS_PER_MONTH * 6,
    '1 Year': TRADING_DAYS_PER_YEAR,
    '5 Years': TRADING_DAYS_PER_YEAR * 5
}

# --- Data Fetching Functions (mostly unchanged, ensure API_KEY check is robust) ---
@st.cache_data(ttl=3600)
def fetch_stock_aggregates(ticker, date_from, date_to):
    if API_KEY == 'YOUR_POLYGON_API_KEY' or not API_KEY:
        # st.error("Polygon API key not configured. Please set the POLYGON_API_KEY environment variable.")
        # This error is better handled where the button is clicked.
        return pd.DataFrame()
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{date_from}/{date_to}?adjusted=true&sort=asc&apiKey={API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if 'results' in data and data['results']:
            df = pd.DataFrame(data['results'])
            df['Date'] = pd.to_datetime(df['t'], unit='ms')
            df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})
            df = df.set_index('Date')
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        return pd.DataFrame()
    except requests.RequestException as e:
        # st.error(f"Error fetching stock aggregates for {ticker}: {e}") # Show error in main UI
        return pd.DataFrame()
    except Exception as e:
        # st.error(f"An unexpected error occurred fetching aggregates: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def fetch_financials(ticker, limit=1):
    if API_KEY == 'YOUR_POLYGON_API_KEY' or not API_KEY: return {}
    url = f"https://api.polygon.io/vX/reference/financials?ticker={ticker}&limit={limit}&sort=filing_date&apiKey={API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get('results') and len(data['results']) > 0:
            return data['results'][0]
        return {}
    except requests.RequestException: return {} # Fail silently, UI will show N/A
    except Exception: return {}


@st.cache_data(ttl=3600)
def fetch_news_sentiment_polygon(ticker: str, limit: int = 20) -> float:
    if API_KEY == 'YOUR_POLYGON_API_KEY' or not API_KEY: return 0.0
    url = f'https://api.polygon.io/v2/reference/news?ticker={ticker}&limit={limit}&sort=published_utc&order=desc&apiKey={API_KEY}'
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if 'results' not in data or not data['results']: return 0.0
        sentiments = []
        for article in data['results']:
            if 'sentiment' in article and isinstance(article['sentiment'], dict) and 'overall' in article['sentiment']:
                sentiments.append(article['sentiment']['overall'])
            elif 'sentiment_score' in article and isinstance(article['sentiment_score'], (int, float)):
                sentiments.append(article['sentiment_score']) # Assuming it's already on a suitable scale or we accept it as is
        if sentiments:
            avg_sentiment = sum(sentiments) / len(sentiments)
            return max(-1.0, min(1.0, avg_sentiment))
        return 0.0
    except requests.RequestException: return 0.0
    except Exception: return 0.0

# --- Technical Indicator Calculations (unchanged) ---
def calculate_sma(series, window):
    return series.rolling(window=window).mean()

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    if loss.eq(0).all(): # Avoid division by zero if all losses are zero
        return pd.Series(100.0, index=series.index) # RSI is 100 if no losses
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- Baseline Projection & Adjustment Logic ---
def generate_baseline_projection(historical_closes: pd.Series, future_days: int):
    if historical_closes.empty or len(historical_closes) < 5: # Need at least a few points
        last_price = historical_closes.iloc[-1] if not historical_closes.empty else 0
        start_date_proj = historical_closes.index[-1] + timedelta(days=1) if not historical_closes.empty else datetime.today() + timedelta(days=1)
        projected_dates = pd.date_range(start=start_date_proj, periods=future_days)
        return pd.Series([last_price] * future_days, index=projected_dates)

    last_actual_close = historical_closes.iloc[-1] # Anchor projection from the very last closing price

    # Calculate the EMA to derive a short-term directional cue
    ema_short_span = 12
    if len(historical_closes) < ema_short_span: # Ensure we have enough data for the span
        ema_short_span = len(historical_closes)
    
    projected_daily_change = 0.0 # Default to a flat trend

    if ema_short_span > 1 : # Need at least 2 points to calculate EMA and trend
        ema_short = historical_closes.ewm(span=ema_short_span, adjust=False).mean()
        
        if len(ema_short) > 1: # Need at least 2 EMA points to calculate a trend
            trend_lookback = min(5, len(ema_short) - 1) # Look at up to last 5 EMA points for trend
            if trend_lookback > 0:
                # Calculate the recent daily change (slope) of the short-term EMA
                projected_daily_change = (ema_short.iloc[-1] - ema_short.iloc[-trend_lookback]) / trend_lookback
    
    # --- Crucial: Dampen the calculated trend for longer horizons ---
    # These dampening factors are illustrative; they make projections more conservative.
    if future_days >= HORIZON_DAYS['5 Years']:  # Approx 5 years
        projected_daily_change *= 0.05  # Heavily dampen (e.g., use only 5% of the calculated short-term slope)
    elif future_days >= HORIZON_DAYS['1 Year']:  # Approx 1 year
        projected_daily_change *= 0.25  # Moderately dampen (e.g., use only 25% of the slope)
    # For shorter horizons (3-6 months), we might use more of the calculated slope (e.g., 0.5 to 1.0)
    elif future_days >= HORIZON_DAYS['6 Months']:
        projected_daily_change *= 0.5 # Slightly less dampening

    # Generate projected values starting from the last actual close, using the (dampened) daily change
    projected_values = [last_actual_close + projected_daily_change * i for i in range(1, future_days + 1)]
    
    projected_dates = pd.date_range(start=historical_closes.index[-1] + timedelta(days=1), periods=future_days)
    return pd.Series(projected_values, index=projected_dates)


def apply_factor_adjustments(last_historical_price: float, 
                             baseline_projection: pd.Series, 
                             factors: dict, 
                             horizon: str):
    """
    Applies adjustments to the baseline projection based on factor scores and horizon.
    The adjustment is applied to the *change* from the last_historical_price.
    """
    if baseline_projection.empty: 
        return baseline_projection, 1.0 # No change if baseline is empty

    weights_config = {
        'fundamental': {'3 Months': 0.05, '6 Months': 0.10, '1 Year': 0.20, '5 Years': 0.30},
        'valuation':   {'3 Months': 0.03, '6 Months': 0.05, '1 Year': 0.15, '5 Years': 0.20},
        'macro':       {'3 Months': 0.10, '6 Months': 0.15, '1 Year': 0.15, '5 Years': 0.10},
        'sentiment':   {'3 Months': 0.20, '6 Months': 0.15, '1 Year': 0.05, '5 Years': 0.02},
        'technical':   {'3 Months': 0.15, '6 Months': 0.10, '1 Year': 0.02, '5 Years': 0.01},
    }

    total_adjustment_impact_score = 0.0
    for factor_group, score in factors.items():
        if factor_group in weights_config:
            total_adjustment_impact_score += score * weights_config[factor_group].get(horizon, 0.0)
    
    # This multiplier applies to the *change from the last historical price*
    multiplier_on_change = 1 + np.clip(total_adjustment_impact_score, -0.9, 0.9) # Clip to avoid excessive changes (e.g. negative prices if P_last is small)
                                                                                # Max impact capped at +/- 90% of baseline change

    # A_k = P_last + (B_k - P_last) * multiplier_on_change
    # Where B_k are values from baseline_projection
    adjusted_values = last_historical_price + (baseline_projection - last_historical_price) * multiplier_on_change
    
    adjusted_projection_series = pd.Series(adjusted_values, index=baseline_projection.index)
    
    # This reported multiplier reflects the effect on the *change*
    return adjusted_projection_series, multiplier_on_change


# --- Streamlit App UI (largely unchanged in structure, but calls to apply_factor_adjustments and plotting modified) ---
st.set_page_config(layout="wide")
st.title("📈 Interactive Stock Scenario Simulator (POC)")
st.markdown("This tool allows you to explore *potential* stock price scenarios based on your assessment of various factors. **It is not financial advice.**")

# --- Sidebar ---
st.sidebar.header("🛠️ Configuration")
ticker = st.sidebar.text_input("Stock Ticker (e.g., AAPL, MSFT)", "AMD").upper() # Default to AMD as per image
selected_horizon = st.sidebar.selectbox("Projection Horizon", list(HORIZON_DAYS.keys()), index=0) # Default to 3 Months

# Initialize session state
if 'historical_data' not in st.session_state: st.session_state.historical_data = pd.DataFrame()
if 'financial_data' not in st.session_state: st.session_state.financial_data = {}
if 'news_sentiment_score' not in st.session_state: st.session_state.news_sentiment_score = 0.0
if 'last_ticker' not in st.session_state: st.session_state.last_ticker = ""
if 'last_horizon' not in st.session_state: st.session_state.last_horizon = ""


fetch_data_button = st.sidebar.button("Fetch Market Data & Analyze")

if API_KEY == 'YOUR_POLYGON_API_KEY' or not API_KEY:
    st.sidebar.error("Polygon API key not configured. Set POLYGON_API_KEY env variable.")
    st.stop() # Stop execution if no API key

if fetch_data_button:
    if ticker:
        st.session_state.last_ticker = ticker
        st.session_state.last_horizon = selected_horizon # Store horizon too

        today = datetime.now()
        date_to_hist = today.strftime('%Y-%m-%d')
        if selected_horizon == '5 Years': date_from_hist = (today - timedelta(days=8*365)).strftime('%Y-%m-%d')
        elif selected_horizon == '1 Year': date_from_hist = (today - timedelta(days=3*365)).strftime('%Y-%m-%d')
        else: date_from_hist = (today - timedelta(days=2*365)).strftime('%Y-%m-%d')

        with st.spinner(f"Fetching data for {ticker}..."):
            hist_data_temp = fetch_stock_aggregates(ticker, date_from_hist, date_to_hist)
            if hist_data_temp.empty:
                st.error(f"Could not fetch historical price data for {ticker}. Check ticker or API key.")
                st.session_state.historical_data = pd.DataFrame() # Clear previous data
            else:
                st.session_state.historical_data = hist_data_temp
                st.session_state.financial_data = fetch_financials(ticker)
                st.session_state.news_sentiment_score = fetch_news_sentiment_polygon(ticker)
                st.success(f"Data fetched for {ticker}.")
    else:
        st.sidebar.warning("Please enter a stock ticker.")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Factor Scenario Inputs")
st.sidebar.caption("Adjust sliders based on your outlook (-1 Bad, 0 Neutral, +1 Good).")

# Factor Inputs (using example values from the screenshot as defaults where applicable)
factors = {}
with st.sidebar.expander("I. Fundamental Factors", expanded=True):
    eps_growth = st.slider("EPS Growth Outlook", -1.0, 1.0, -0.80, 0.05, help="Company's future earnings per share growth prospects.")
    revenue_growth = st.slider("Revenue Growth Outlook", -1.0, 1.0, -0.80, 0.05, help="Company's future revenue growth prospects.")
    margin_outlook = st.slider("Profit Margin Outlook", -1.0, 1.0, -0.90, 0.05, help="Outlook for gross, operating, or net profit margins.")
    management_quality = st.slider("Management & Governance Score", -1.0, 1.0, -0.85, 0.05, help="Perceived quality of management team and corporate governance.")
    economic_moat = st.slider("Competitive Advantage (Moat)", -1.0, 1.0, -0.95, 0.05, help="Strength of the company's sustainable competitive advantages.") # From image
    factors['fundamental'] = np.mean([eps_growth, revenue_growth, margin_outlook, management_quality, economic_moat])


with st.sidebar.expander("II. Valuation Perspective", expanded=True): # Expanded as in image
    pe_text = "Current P/E: N/A"
    ps_text = "Current P/S: N/A"
    if not st.session_state.historical_data.empty and 'Close' in st.session_state.historical_data.columns and st.session_state.financial_data:
        current_price = st.session_state.historical_data['Close'].iloc[-1]
        financials_data = st.session_state.financial_data.get('financials', {})
        income_statement = financials_data.get('income_statement', {})
        basic_eps_data = income_statement.get('basic_earnings_per_share', {})
        revenues_data = income_statement.get('revenues', {})
        
        basic_eps = basic_eps_data.get('value') if basic_eps_data else None
        revenues = revenues_data.get('value') if revenues_data else None
        
        if basic_eps and basic_eps != 0: pe_text = f"Reported Basic EPS: {basic_eps:.2f}. Current P/E (approx): {(current_price / basic_eps):.2f}"
        
        if revenues and revenues != 0:
            shares_data = st.session_state.financial_data.get('shares_outstanding', {})
            weighted_avg_shares = shares_data.get('weighted_average_shares_outstanding', {}).get('value') if shares_data else None
            if not weighted_avg_shares: # Fallback to basic average shares if diluted not available
                 basic_avg_shares_data = shares_data.get('basic_average_shares',{}) if shares_data else {}
                 weighted_avg_shares = basic_avg_shares_data.get('value') if basic_avg_shares_data else None


            if weighted_avg_shares and weighted_avg_shares != 0:
                ps_text = f"Reported Revenue: {revenues/1e9:.2f}B. Current P/S (approx): {(current_price * weighted_avg_shares / revenues):.2f}"
            else:
                ps_text = f"Reported Revenue: {revenues/1e9:.2f}B. (P/S needs shares outstanding)"
    
    valuation_score = st.slider("Valuation Score (P/E, P/S)", -1.0, 1.0, -0.85, 0.05, help=f"-1 Overvalued, +1 Undervalued. Info: {pe_text}; {ps_text}") # From image
    factors['valuation'] = valuation_score

with st.sidebar.expander("III. Macroeconomic Climate", expanded=True): # Expanded as in image
    interest_rate_env = st.slider("Interest Rate Environment Impact", -1.0, 1.0, -0.75, 0.05, help="Impact of current/expected interest rates (-1 Rates rising/high, +1 Rates falling/low).") # From image
    inflation_outlook = st.slider("Inflation Outlook Impact", -1.0, 1.0, -0.80, 0.05, help="Impact of inflation (-1 High/rising inflation, +1 Low/falling inflation).") # From image
    gdp_growth_env = st.slider("Economic Growth (GDP) Impact", -1.0, 1.0, -0.60, 0.05, help="Impact of broader economic growth outlook.") # From image
    factors['macro'] = np.mean([interest_rate_env, inflation_outlook, gdp_growth_env])

with st.sidebar.expander("IV. Market Sentiment & Behavior", expanded=False): # Collapsed as in image
    st.sidebar.markdown(f"**Fetched News Sentiment:** `{st.session_state.news_sentiment_score:.2f}` (Polygon.io)")
    news_override = st.slider("News Sentiment Adjustment", -1.0, 1.0, st.session_state.news_sentiment_score, 0.05, help="Adjust or use fetched news sentiment. Higher is more positive.")
    social_media_hype = st.slider("Social Media Buzz/Hype", -1.0, 1.0, 0.0, 0.05, help="Impact of social media trends (-1 Negative hype, +1 Positive hype).")
    analyst_consensus = st.slider("Analyst Ratings Consensus", -1.0, 1.0, 0.0, 0.05, help="Overall sentiment from analyst ratings (-1 Strong Sell, +1 Strong Buy).")
    factors['sentiment'] = np.mean([news_override, social_media_hype, analyst_consensus]) # Factor score for image seems to be 0.00

with st.sidebar.expander("V. Technical Analysis Signals", expanded=False): # Collapsed as in image
    sma_50_text = "50-day MA: N/A"
    sma_200_text = "200-day MA: N/A"
    rsi_text = "RSI(14): N/A"
    trend_score_val = -0.65 # Approx from image for technical factor score

    if not st.session_state.historical_data.empty and 'Close' in st.session_state.historical_data.columns and len(st.session_state.historical_data) > 14:
        closes = st.session_state.historical_data['Close']
        if len(closes) >= 50: sma_50 = calculate_sma(closes, 50); sma_50_text = f"50-day MA: {sma_50.iloc[-1]:.2f}"
        if len(closes) >= 200: sma_200 = calculate_sma(closes, 200); sma_200_text = f"200-day MA: {sma_200.iloc[-1]:.2f}"
        rsi_values = calculate_rsi(closes, 14); rsi_text = f"RSI(14): {rsi_values.iloc[-1]:.2f}"
        
        # Auto-set trend_score_val based on MAs (example)
        if len(closes) >= 200 and not sma_50.empty and not sma_200.empty: # ensure series not empty
             if closes.iloc[-1] > sma_50.iloc[-1] and sma_50.iloc[-1] > sma_200.iloc[-1]: trend_score_val = 0.75
             elif closes.iloc[-1] < sma_50.iloc[-1] and sma_50.iloc[-1] < sma_200.iloc[-1]: trend_score_val = -0.75
             elif closes.iloc[-1] > sma_50.iloc[-1]: trend_score_val = 0.25
             elif closes.iloc[-1] < sma_50.iloc[-1]: trend_score_val = -0.25
             else: trend_score_val = 0.0

    st.sidebar.markdown(f"Technicals: {sma_50_text}, {sma_200_text}, {rsi_text}")
    trend_strength = st.slider("Technical Trend Strength", -1.0, 1.0, trend_score_val, 0.05, help="Current technical trend (-1 Bearish, +1 Bullish). Informed by MAs.")
    # For image, technical score is -0.68. We can adjust momentum_rsi to get close.
    # If trend_strength is e.g. -0.75, then momentum_rsi should be around -0.61 for average to be -0.68
    momentum_rsi = st.slider("Momentum (RSI based)", -1.0, 1.0, -0.60, 0.05, help="Momentum from RSI (-1 Overbought, +1 Oversold - note interpretation).")
    factors['technical'] = np.mean([trend_strength, momentum_rsi])


# --- Main Display Area ---
# Ensure data is for the current ticker and horizon before displaying chart
if st.session_state.last_ticker == ticker and \
   st.session_state.last_horizon == selected_horizon and \
   not st.session_state.historical_data.empty and \
   'Close' in st.session_state.historical_data:

    st.header(f"Scenario for {ticker} - {selected_horizon}")

    historical_to_plot = st.session_state.historical_data.copy()
    display_history_days = HORIZON_DAYS[selected_horizon] * 2
    if selected_horizon == '5 Years': display_history_days = HORIZON_DAYS[selected_horizon]
    
    # Ensure we have enough data for display_history_days, otherwise use all available
    # Check if historical_to_plot has enough rows for .last()
    if len(historical_to_plot) > display_history_days:
        historical_to_plot = historical_to_plot.last(f'{display_history_days}D')
    
    # Ensure historical_to_plot is not empty after .last() operation (e.g. if display_history_days is very large)
    if historical_to_plot.empty:
        st.warning("Not enough historical data to display for the selected period/horizon combination.")
        st.stop()


    last_hist_close = historical_to_plot['Close'].iloc[-1]
    last_hist_date = historical_to_plot.index[-1]

    baseline_proj_future_values = generate_baseline_projection(historical_to_plot['Close'], HORIZON_DAYS[selected_horizon])
    adjusted_proj_future_values, reported_multiplier = apply_factor_adjustments(
        last_hist_close, baseline_proj_future_values, factors, selected_horizon
    )

    fig = go.Figure()
    # Historical Data
    fig.add_trace(go.Scatter(x=historical_to_plot.index, y=historical_to_plot['Close'], mode='lines', name='Historical Close', line=dict(color='blue')))

    # Baseline Projection - Prepend last historical point for visual continuity
    # Create a Series for the single starting point
    initial_point_baseline = pd.Series([last_hist_close], index=[last_hist_date])
    # Concatenate the initial point with the future projection values
    plot_baseline_series = pd.concat([initial_point_baseline, baseline_proj_future_values])
    fig.add_trace(go.Scatter(x=plot_baseline_series.index, y=plot_baseline_series.values, mode='lines', name='Baseline Projection (EMA based)', line=dict(dash='dash', color='grey')))

    # Adjusted Projection - Prepend last historical point for visual continuity
    # Create a Series for the single starting point
    initial_point_adjusted = pd.Series([last_hist_close], index=[last_hist_date])
    # Concatenate the initial point with the future projection values
    plot_adjusted_series = pd.concat([initial_point_adjusted, adjusted_proj_future_values])
    fig.add_trace(go.Scatter(x=plot_adjusted_series.index, y=plot_adjusted_series.values, mode='lines', name='Factor-Adjusted Scenario', line=dict(color='green', width=2)))
    
    # Determine y-axis range for better visualization
    min_val_hist = historical_to_plot['Close'].min() 
    max_val_hist = historical_to_plot['Close'].max()
    
    # Check if projection series are not empty before calling min/max
    min_val_base = plot_baseline_series.min() if not plot_baseline_series.empty else min_val_hist
    max_val_base = plot_baseline_series.max() if not plot_baseline_series.empty else max_val_hist
    min_val_adj = plot_adjusted_series.min() if not plot_adjusted_series.empty else min_val_hist
    max_val_adj = plot_adjusted_series.max() if not plot_adjusted_series.empty else max_val_hist

    overall_min = min(min_val_hist, min_val_base, min_val_adj) * 0.95
    overall_max = max(max_val_hist, max_val_base, max_val_adj) * 1.05
    fig.update_yaxes(range=[overall_min, overall_max])

    title_text = (f'{ticker} Price Scenario ({selected_horizon}) | '
                  f'Adj. Effect on Change: {(reported_multiplier-1)*100:.1f}% '
                  f'(Multiplier: {reported_multiplier:.3f})')
    fig.update_layout(
        title=title_text,
        xaxis_title='Date', yaxis_title='Price (USD)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Factor Group Scores & Overall Effect:")
    factor_details_md = ""
    for group, score_val in factors.items(): 
        factor_details_md += f"- **{group.capitalize()}:** Score {score_val:.2f}\n"
    
    factor_details_md += (f"\n**Overall Multiplier on Baseline Change:** {reported_multiplier:.3f} "
                          f"({(reported_multiplier-1)*100:.1f}% effect on the change from last price)")
    st.markdown(factor_details_md)


elif not fetch_data_button and st.session_state.last_ticker == "": 
    st.info("Enter a ticker and click 'Fetch Market Data & Analyze' to begin.")
elif fetch_data_button and st.session_state.historical_data.empty: 
    pass 
else: 
    st.info("Current display might be for a previous selection. Adjust inputs and click 'Fetch Market Data & Analyze' for the latest scenario.")


st.sidebar.markdown("---")
st.sidebar.info("This POC uses heuristic weights and a simplified model. Factor scores are averaged within groups. Use for educational exploration only.")