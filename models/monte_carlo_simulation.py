import numpy as np
import pandas as pd

def monte_carlo_simulation(start_price: float, drift: float, volatility: float, days: int, num_simulations: int = 1000) -> pd.DataFrame:
    """
    Simulate price paths using Monte Carlo method.
    :param start_price: Initial price of the asset
    :param drift: Expected return (annualized)
    :param volatility: Volatility of the asset (annualized)
    :param days: Number of days to simulate
    :param num_simulations: Number of simulation paths
    :return: DataFrame with simulated price paths (each column is a simulation)
    """
    dt = 1/252  # Time step in years (assuming 252 trading days)
    price_paths = np.zeros((days, num_simulations))
    price_paths[0] = start_price

    for t in range(1, days):
        random_shocks = np.random.normal(loc=0, scale=1, size=num_simulations)
        price_paths[t] = price_paths[t-1] * np.exp((drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * random_shocks)

    return pd.DataFrame(price_paths)
