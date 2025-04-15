import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def forecast_arima(historical_prices: pd.Series, periods: dict) -> dict:
    """
    Use ARIMA model to forecast future prices for given periods.
    periods: dict with keys as period names and values as forecast horizon in days.
    Returns dict with keys as period names and values as forecasted growth multipliers.
    """
    growth_rates = {}
    model = ARIMA(historical_prices, order=(5,1,0))
    model_fit = model.fit()

    last_price = historical_prices.iloc[-1]

    for period_name, days in periods.items():
        forecast = model_fit.forecast(steps=days)
        # Use iloc to get last forecast value to avoid KeyError
        forecast_price = forecast.iloc[-1]
        growth = forecast_price / last_price
        growth_rates[period_name] = growth

    return growth_rates
