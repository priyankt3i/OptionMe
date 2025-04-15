import numpy as np
from scipy.stats import norm

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """
    Calculate Black-Scholes option price.
    :param S: Current stock price
    :param K: Strike price
    :param T: Time to maturity in years
    :param r: Risk-free interest rate
    :param sigma: Volatility of the underlying asset
    :param option_type: 'call' or 'put'
    :return: Option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price

def binomial_tree_price(S, K, T, r, sigma, steps=100, option_type='call'):
    """
    Calculate option price using binomial tree model.
    :param S: Current stock price
    :param K: Strike price
    :param T: Time to maturity in years
    :param r: Risk-free interest rate
    :param sigma: Volatility of the underlying asset
    :param steps: Number of steps in the binomial tree
    :param option_type: 'call' or 'put'
    :return: Option price
    """
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Initialize asset prices at maturity
    asset_prices = np.zeros(steps + 1)
    option_values = np.zeros(steps + 1)

    for i in range(steps + 1):
        asset_prices[i] = S * (u ** (steps - i)) * (d ** i)

    # Calculate option values at maturity
    if option_type == 'call':
        option_values = np.maximum(asset_prices - K, 0)
    else:
        option_values = np.maximum(K - asset_prices, 0)

    # Backward induction
    for step in range(steps - 1, -1, -1):
        for i in range(step + 1):
            option_values[i] = (p * option_values[i] + (1 - p) * option_values[i + 1]) * np.exp(-r * dt)

    return option_values[0]
