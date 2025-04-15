import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import os
from dotenv import load_dotenv

from models import disclaimer, monte_carlo_simulation, options_pricing, external_api_integration

load_dotenv() # Load environment variables from .env file

@st.cache_data(ttl=300)
def get_stock_data(ticker):
    """Fetches the latest price for a given stock ticker from Polygon.io (free endpoint)."""
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key or api_key == 'your_api_key_here':
        st.error("POLYGON_API_KEY not found or not set in .env file. Please add your Polygon.io API key.")
        return pd.DataFrame({'Date': [pd.Timestamp.now()], 'Price': [np.nan]})

    # Use free tier-compatible endpoint for previous day's closing price
    url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/prev?apiKey={api_key}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Extract closing price from free tier-compatible endpoint response
        if 'results' in data and len(data['results']) > 0:
            current_price = data['results'][0]['c']
        else:
            st.warning(f"No price data available for {ticker}.")
            return pd.DataFrame({'Date': [pd.Timestamp.now()], 'Price': [np.nan]})

        return pd.DataFrame({'Date': [pd.Timestamp.now()], 'Price': [current_price]})
            
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}")
        st.error(f"API Response: {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
    except KeyError as ke:
        st.error(f"Unexpected API response format: Missing key {ke}")
    except Exception as e:
        st.error(f"An error occurred processing data for {ticker}: {str(e)}")

def calculate_outcomes(investment_amount, stock_price, growth_rates=None, options_strategy=None):
    """
    Calculate investment outcomes with optional predicted growth rates.
    growth_rates: dict with keys as periods (e.g., '3m', '6m', '1y', '5y') and values as expected growth multipliers.
    options_strategy: string indicating selected options trading strategy.
    """
    shares = investment_amount / stock_price
    outcomes = {}

    # Direct Shares outcomes with predicted growth
    direct_outcomes = {}
    for period, growth in growth_rates.items():
        # Use growth directly without random sign simulation
        future_value = shares * stock_price * growth
        profit_loss = future_value - investment_amount
        direct_outcomes[period] = {
            'Future Value': future_value,
            'Profit/Loss': profit_loss,
            'Return %': 100 * profit_loss / investment_amount
        }
    outcomes['Direct Shares'] = direct_outcomes

    # Options Strategy outcomes based on selected strategy
    option_outcomes = {}
    premium = 2  # Simplified fixed premium for example
    strike_price = stock_price

    if options_strategy == "Long Call (bullish)":
        for period, growth in growth_rates.items():
            future_price = stock_price * growth
            intrinsic_value = max(future_price - strike_price, 0)
            option_value = intrinsic_value * shares - premium
            profit_loss = option_value - investment_amount
            option_outcomes[period] = {
                'Option Value': option_value,
                'Profit/Loss': profit_loss,
                'Return %': 100 * profit_loss / investment_amount
            }
    elif options_strategy == "Long Put (bearish)":
        for period, growth in growth_rates.items():
            future_price = stock_price * growth
            intrinsic_value = max(strike_price - future_price, 0)
            option_value = intrinsic_value * shares - premium
            profit_loss = option_value - investment_amount
            option_outcomes[period] = {
                'Option Value': option_value,
                'Profit/Loss': profit_loss,
                'Return %': 100 * profit_loss / investment_amount
            }
    else:
        # Default simplified option strategy outcome (no growth prediction)
        option_value = max(stock_price - strike_price, 0) * shares - premium
        profit_loss = option_value - investment_amount
        for period in growth_rates.keys():
            option_outcomes[period] = {
                'Option Value': option_value,
                'Profit/Loss': profit_loss,
                'Return %': 100 * profit_loss / investment_amount
            }

    outcomes['Options Strategy'] = option_outcomes

    return outcomes

st.set_page_config(page_title='Investment Strategy Analyzer', layout='wide')

# Sidebar inputs
with st.sidebar:
    st.header('Investment Parameters')

    # Add text box for Polygon.io API key input
    api_key_input = st.text_input(
        "Enter your Polygon.io API Key",
        value=os.getenv('POLYGON_API_KEY') or '',
        type="password",
        help="Get your API key from https://polygon.io/dashboard/keys"
    )

    investment_amount = st.slider(
        'Select Investment Amount',
        min_value=100,
        max_value=100000,
        value=1000,
        step=100
    )
    @st.cache_data(ttl=3600)
    def fetch_stock_list():
        import os
        import requests

        api_key = os.getenv('POLYGON_API_KEY')
        if not api_key or api_key == 'your_api_key_here':
            st.error("POLYGON_API_KEY not found or not set in .env file. Please add your Polygon.io API key.")
            return []

        url = f'https://api.polygon.io/v3/reference/tickers?market=stocks&active=true&limit=1000&apiKey={api_key}'

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if 'results' not in data:
                st.error("Failed to fetch stock list from Polygon.io.")
                return []

            # Extract ticker symbols
            tickers = [item['ticker'] for item in data['results']]
            return tickers

        except requests.RequestException as e:
            st.error(f"Failed to fetch stock list: {e}")
            return []

    stock_list = fetch_stock_list()

    stock = st.selectbox(
        'Select Stock',
        stock_list if stock_list else ['AMD', 'AAPL', 'TSLA', 'MSFT']
    )
    investment_type = st.radio(
        "Select Investment Type",
        ["Direct Shares", "Options Strategy"]
    )
    prediction_model = st.selectbox(
        "Select Prediction Model",
        ["T3i Prediction Model", "Historical Volatility", "ARIMA Forecasting", "Monte Carlo Simulation"]
    )
    options_strategy = None
    covered_call_params = {}
    if investment_type == "Options Strategy":
        options_strategy = st.selectbox(
            "Select Options Trading Strategy",
            [
                "Long Call (bullish)",
                "Long Put (bearish)",
                "Covered Call",
                "Cash Secured Put",
                "Naked Call (bearish)",
                "Naked Put (bullish)",
                "Credit Spread",
                "Call Spread",
                "Put Spread",
                "Poor Man's Covered Call",
                "Calendar Spread",
                "Ratio Back Spread",
                "Iron Condor",
                "Butterfly",
                "Collar",
                "Diagonal Spread",
                "Double Diagonal",
                "Straddle",
                "Strangle",
                "Covered Strangle",
                "Synthetic Put",
                "Reverse Conversion",
                "Custom (8 Legs)",
                "Custom (6 Legs)",
                "Custom (5 Legs)",
                "Custom (4 Legs)",
                "Custom (3 Legs)",
                "Custom (2 Legs)"
            ]
        )
        # Add explanation text below dropdown
        explanations = {
            "Long Call (bullish)": "A Long Call is a bullish strategy where you buy a call option expecting the stock price to rise.",
            "Long Put (bearish)": "A Long Put is a bearish strategy where you buy a put option expecting the stock price to fall.",
            "Covered Call": "Selling call options on stocks you own to generate income, with limited upside.",
            "Cash Secured Put": "Selling put options while holding enough cash to buy the stock if assigned.",
            "Naked Call (bearish)": "Selling call options without owning the underlying stock, expecting the price to fall.",
            "Naked Put (bullish)": "Selling put options without holding the stock, expecting the price to rise.",
            "Credit Spread": "An options strategy involving buying and selling options to limit risk and profit.",
            "Call Spread": "Buying and selling call options at different strike prices to limit risk.",
            "Put Spread": "Buying and selling put options at different strike prices to limit risk.",
            "Poor Man's Covered Call": "A long-term call option combined with a short-term call option to generate income.",
            "Calendar Spread": "Buying and selling options with different expiration dates to profit from time decay.",
            "Ratio Back Spread": "An options strategy involving buying more options than sold to profit from volatility.",
            "Iron Condor": "A strategy combining two credit spreads to profit from low volatility.",
            "Butterfly": "An options strategy combining multiple options to profit from minimal price movement.",
            "Collar": "Protecting gains by holding the stock and buying protective options.",
            "Diagonal Spread": "Combining options with different strike prices and expiration dates.",
            "Double Diagonal": "A complex strategy combining two diagonal spreads.",
            "Straddle": "Buying a call and put option at the same strike price to profit from volatility.",
            "Strangle": "Buying a call and put option at different strike prices to profit from volatility.",
            "Covered Strangle": "Selling a call and put option while owning the underlying stock.",
            "Synthetic Put": "Combining a long stock position with a short call option.",
            "Reverse Conversion": "An arbitrage strategy involving options and the underlying stock.",
            "Custom (8 Legs)": "A custom strategy involving 8 option legs.",
            "Custom (6 Legs)": "A custom strategy involving 6 option legs.",
            "Custom (5 Legs)": "A custom strategy involving 5 option legs.",
            "Custom (4 Legs)": "A custom strategy involving 4 option legs.",
            "Custom (3 Legs)": "A custom strategy involving 3 option legs.",
            "Custom (2 Legs)": "A custom strategy involving 2 option legs."
        }
        explanation_text = explanations.get(options_strategy, "Select an options trading strategy to see its explanation.")
        st.markdown(f"**Strategy Explanation:** {explanation_text}")

    # Main window
    st.title('Investment Strategy Analyzer')
    st.subheader('Selected Investment Parameters')
    col1, col2 = st.columns(2)
    with col1:
        st.metric('Investment Amount', f'$ {investment_amount}')
        st.metric('Selected Stock', stock)
        st.metric('Investment Type', investment_type)
        
    with col2:
        # Get and display current stock price
        stock_data = get_stock_data(stock)
        if not stock_data['Price'].isnull().all():
            current_price = stock_data['Price'].iloc[-1]
            st.metric('Current Price', f'$ {current_price:.2f}')
        else:
            st.error("Failed to fetch current price. Check API status and key.")
            current_price = np.nan

    # Covered Call parameters input UI inside main window after current_price is defined
    if options_strategy == "Covered Call" and not np.isnan(current_price):
        st.sidebar.markdown("### Covered Call Parameters")
        covered_call_params['stock_quantity'] = st.sidebar.number_input("Stock Quantity", min_value=1, value=100, step=1)
        covered_call_params['call_strike'] = st.sidebar.number_input("Call Option Strike Price", min_value=1.0, value=round(current_price * 1.05, 2), step=0.01)
        covered_call_params['call_expiration'] = st.sidebar.number_input("Call Option Expiration (days)", min_value=1, max_value=365, value=30, step=1)

# Main window
st.title('Investment Strategy Analyzer')
st.subheader('Selected Investment Parameters')
col1, col2 = st.columns(2)
with col1:
    st.metric('Investment Amount', f'$ {investment_amount}')
    st.metric('Selected Stock', stock)
    st.metric('Investment Type', investment_type)
    
with col2:
    # Get and display current stock price
    stock_data = get_stock_data(stock)
    if not stock_data['Price'].isnull().all():
        current_price = stock_data['Price'].iloc[-1]
        st.metric('Current Price', f'$ {current_price:.2f}')
    else:
        st.error("Failed to fetch current price. Check API status and key.")
        current_price = np.nan

# Calculate and display outcomes only if current_price is valid
if not np.isnan(current_price):
    import pandas as pd
    from models import historical_data_analysis, arima_forecasting

    # Fetch historical price data for the selected stock (simplified example)
    # In real app, fetch from Polygon.io or other data source
    @st.cache_data(ttl=300)
    def fetch_historical_prices(ticker):
        # Fetch historical prices from Polygon.io API with rate limit handling
        import os
        import requests
        import pandas as pd
        import time

        api_key = os.getenv('POLYGON_API_KEY')
        if not api_key or api_key == 'your_api_key_here':
            st.error("POLYGON_API_KEY not found or not set in .env file. Please add your Polygon.io API key.")
            return pd.Series(dtype=float)

        url = f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/2010-01-01/{pd.Timestamp.today().date()}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}'

        max_retries = 5
        retry_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                response = requests.get(url)
                if response.status_code == 429:
                    # Rate limit exceeded, wait and retry
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                response.raise_for_status()
                data = response.json()

                if 'results' not in data:
                    st.error(f"No historical data found for {ticker}.")
                    return pd.Series(dtype=float)

                prices = pd.Series(
                    [item['c'] for item in data['results']],
                    index=pd.to_datetime([item['t'] for item in data['results']], unit='ms')
                )
                return prices

            except requests.RequestException as e:
                st.error(f"Failed to fetch historical prices: {e}")
                return pd.Series(dtype=float)

        st.error("Exceeded maximum retries due to rate limiting.")
        return pd.Series(dtype=float)

    historical_prices = fetch_historical_prices(stock)

    periods_days = {
        '3 Months': 63,
        '6 Months': 126,
        '1 Year': 252,
        '5 Years': 1260
    }

    # Determine growth rates based on selected prediction model
    if prediction_model == "T3i Prediction Model":
        from models.t3i_prediction import calculate_t3i_growth_rates
        growth_rates = calculate_t3i_growth_rates(historical_prices, stock)
    elif prediction_model == "Historical Volatility":
        growth_rates = historical_data_analysis.calculate_historical_growth_rates(historical_prices)
    elif prediction_model == "ARIMA Forecasting":
        growth_rates = arima_forecasting.forecast_arima(historical_prices, periods_days)
    elif prediction_model == "Monte Carlo Simulation":
        # Use Monte Carlo simulation to estimate growth rates
        sim_df = monte_carlo_simulation.monte_carlo_simulation(
            start_price=current_price,
            drift=0.05,  # example drift, could be improved
            volatility=0.2,  # example volatility, could be improved
            days=1260,
            num_simulations=1000
        )
        # Calculate average growth over periods
        growth_rates = {}
        for period, days in periods_days.items():
            avg_price = sim_df.iloc[days - 1].mean()
            growth_rates[period] = avg_price / current_price
    else:
        growth_rates = {
            '3 Months': 1.05,
            '6 Months': 1.10,
            '1 Year': 1.20,
            '5 Years': 1.50
        }

    # Calculate outcomes using current selections
    outcomes = calculate_outcomes(
        investment_amount,
        current_price,
        growth_rates,
        options_strategy
    )

    # Covered Call strategy integration
    if investment_type == "Options Strategy" and options_strategy == "Covered Call" and not np.isnan(current_price):
        from models.options_strategies import OptionLeg, CoveredCallStrategy

        stock_qty = covered_call_params.get('stock_quantity', 100)
        call_strike = covered_call_params.get('call_strike', round(current_price * 1.05, 2))
        call_exp_days = covered_call_params.get('call_expiration', 30)
        call_exp = call_exp_days / 365

        call_leg = OptionLeg(option_type='call', strike=call_strike, quantity=1, expiration=call_exp, is_call=True, is_long=False)
        covered_call = CoveredCallStrategy(stock_price=current_price, stock_quantity=stock_qty, call_leg=call_leg)

        # Simulate payoff at expiration prices around current price
        prices = np.linspace(current_price * 0.5, current_price * 1.5, 20)
        payoffs = [covered_call.profit_loss(p) for p in prices]

        st.subheader("Covered Call Strategy Payoff at Expiration")
        st.line_chart(data={'Stock Price': prices, 'Profit/Loss': payoffs})

    # Display outcomes based on investment type
    if investment_type == "Direct Shares":
        st.subheader('Investment Outcomes (Predicted)')
        # Transpose display: columns are durations, rows are Future Value, Gain / Loss, Return %
        periods = list(outcomes['Direct Shares'].keys())
        future_values = [outcomes['Direct Shares'][p]['Future Value'] for p in periods]
        profit_losses = [outcomes['Direct Shares'][p]['Profit/Loss'] for p in periods]
        returns = [outcomes['Direct Shares'][p]['Return %'] for p in periods]

        cols = st.columns(len(periods))
        # Display header row with period names
        for col, period in zip(cols, periods):
            col.markdown(f"**{period}**")

        # Display Future Value row
        for col, val in zip(cols, future_values):
            col.metric("Future Value", f"$ {val:.2f}")

        # Display Gain / Loss row
        for col, val in zip(cols, profit_losses):
            arrow = "↑" if val >= 0 else "↓"
            color = "green" if val >= 0 else "red"
            # Use markdown with superscript arrow colored accordingly, keep text size same as metric
            col.markdown(f"Gain / Loss<br><span style='font-size: 2.25em;'>$ {val:.2f}</span><span style='color:{color}; font-weight: bold; font-size: 1.25em; vertical-align: super; margin-left: 2px;'>{arrow}</span>", unsafe_allow_html=True)

        # Display Return % row
        for col, val in zip(cols, returns):
            col.metric("Return %", f"{val:.2f}%")
        
    elif investment_type == "Options Strategy":
        option_outcomes = outcomes['Options Strategy']
        st.subheader('Options Strategy Outcomes')
        # Transpose display: columns are durations, rows are Option Value, Gain / Loss, Return %
        periods = list(option_outcomes.keys())
        option_values = [option_outcomes[p]['Option Value'] for p in periods]
        profit_losses = [option_outcomes[p]['Profit/Loss'] for p in periods]
        returns = [option_outcomes[p]['Return %'] for p in periods]

        cols = st.columns(len(periods))
        # Display header row with period names
        for col, period in zip(cols, periods):
            col.markdown(f"**{period}**")

        # Display Option Value row
        for col, val in zip(cols, option_values):
            col.metric("Option Value", f"$ {val:.2f}")

        # Display Gain / Loss row
        for col, val in zip(cols, profit_losses):
            arrow = "↑" if val >= 0 else "↓"
            color = "green" if val >= 0 else "red"
            # Use markdown to display value with colored superscript arrow since metric() does not support unsafe_allow_html
            col.markdown(f"Gain / Loss<br><span style='font-size: 2.25em;'>$ {val:.2f}</span><span style='color:{color}; font-weight: bold; font-size: 1.25em; vertical-align: super; margin-left: 2px;'>{arrow}</span>", unsafe_allow_html=True)

        # Display Return % row
        for col, val in zip(cols, returns):
            col.metric("Return %", f"{val:.2f}%")

import matplotlib.pyplot as plt

# Plot historical data and predicted growth for next 5 years
if 'historical_prices' in locals() and historical_prices is not None and not historical_prices.empty:
    st.subheader("Historical Price Data")
    st.line_chart(historical_prices)

    st.subheader("Predicted Growth Over Next 5 Years")
    periods = ['3 Months', '6 Months', '1 Year', '5 Years']
    last_price = historical_prices.iloc[-1]
    predicted_prices = {}

    if prediction_model == "T3i Prediction Model":
        growth_factors = [growth_rates.get(p, 1.0) for p in periods]
    elif prediction_model == "Historical Volatility":
        growth_factors = [growth_rates.get(p, 1.0) for p in periods]
    elif prediction_model == "ARIMA Forecasting":
        growth_factors = [growth_rates.get(p, 1.0) for p in periods]
    elif prediction_model == "Monte Carlo Simulation":
        growth_factors = [growth_rates.get(p, 1.0) for p in periods]
    else:
        growth_factors = [1.05, 1.10, 1.20, 1.50]

    for period, factor in zip(periods, growth_factors):
        predicted_prices[period] = last_price * factor

    # Create plot data
    plot_dates = [historical_prices.index[-1]]
    plot_prices = [last_price]
    # Approximate days for each period
    days_map = {'3 Months': 63, '6 Months': 126, '1 Year': 252, '5 Years': 1260}
    for period in periods:
        plot_dates.append(plot_dates[-1] + pd.Timedelta(days=days_map[period]))
        plot_prices.append(predicted_prices[period])

    # Plot
    fig, ax = plt.subplots()
    ax.plot(historical_prices.index, historical_prices.values, label='Historical Prices')
    ax.plot(plot_dates[1:], plot_prices[1:], marker='o', linestyle='--', color='orange', label='Predicted Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

# Add disclaimer at the bottom of the app
st.markdown("---")
st.markdown(f"**Disclaimer:** {disclaimer.DISCLAIMER_TEXT}")
