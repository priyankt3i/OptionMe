import pandas as pd
import numpy as np

def calculate_historical_growth_rates(historical_prices: pd.Series) -> dict:
    """
    Calculate average growth rates for different time horizons based on historical price data.
    Returns a dict with keys as periods and values as growth multipliers.
    """
    growth_rates = {}

    # Calculate simple returns over different periods
    periods = {
        '3 Months': 63,   # Approx trading days in 3 months
        '6 Months': 126,
        '1 Year': 252,
        '5 Years': 1260
    }

    for period_name, days in periods.items():
        if len(historical_prices) > days:
            past_price = historical_prices.iloc[-days]
            current_price = historical_prices.iloc[-1]
            growth = current_price / past_price
            growth_rates[period_name] = growth
        else:
            growth_rates[period_name] = 1.0  # No growth if insufficient data

    return growth_rates
