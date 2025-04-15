from typing import List, Dict, Union
import numpy as np
from models.options_pricing import black_scholes_price, binomial_tree_price

class OptionLeg:
    def __init__(self, option_type: str, strike: float, quantity: int, expiration: float, is_call: bool, is_long: bool):
        """
        Represents a single option leg.
        :param option_type: 'call' or 'put'
        :param strike: Strike price
        :param quantity: Number of contracts (positive integer)
        :param expiration: Time to expiration in years
        :param is_call: True if call option, False if put
        :param is_long: True if long position, False if short
        """
        self.option_type = option_type
        self.strike = strike
        self.quantity = quantity
        self.expiration = expiration
        self.is_call = is_call
        self.is_long = is_long

    def price(self, S: float, r: float, sigma: float) -> float:
        """
        Calculate option price using Black-Scholes model.
        :param S: Current stock price
        :param r: Risk-free interest rate
        :param sigma: Volatility
        :return: Price of this leg (positive for long, negative for short)
        """
        price = black_scholes_price(S, self.strike, self.expiration, r, sigma, self.option_type)
        return price * self.quantity * (1 if self.is_long else -1)

    def payoff(self, S_T: float) -> float:
        """
        Calculate payoff at expiration for this leg.
        :param S_T: Stock price at expiration
        :return: Payoff value (positive for long, negative for short)
        """
        if self.is_call:
            intrinsic = max(S_T - self.strike, 0)
        else:
            intrinsic = max(self.strike - S_T, 0)
        return intrinsic * self.quantity * (1 if self.is_long else -1)

class CoveredCallStrategy:
    def __init__(self, stock_price: float, stock_quantity: int, call_leg: OptionLeg):
        """
        Covered Call strategy: long stock + short call option.
        :param stock_price: Current stock price
        :param stock_quantity: Number of shares held long
        :param call_leg: OptionLeg representing the short call
        """
        self.stock_price = stock_price
        self.stock_quantity = stock_quantity
        self.call_leg = call_leg

    def initial_cost(self) -> float:
        """
        Calculate initial cost of the strategy.
        """
        call_price = self.call_leg.price(self.stock_price, r=0.01, sigma=0.2)  # example r and sigma
        stock_cost = self.stock_price * self.stock_quantity
        return stock_cost - call_price  # short call premium reduces cost

    def payoff(self, S_T: float) -> float:
        """
        Calculate payoff at expiration.
        """
        stock_payoff = self.stock_quantity * S_T
        call_payoff = self.call_leg.payoff(S_T)
        return stock_payoff + call_payoff

    def profit_loss(self, S_T: float) -> float:
        """
        Calculate profit or loss at expiration.
        """
        return self.payoff(S_T) - self.initial_cost()

def example_covered_call():
    # Example usage
    stock_price = 100
    stock_quantity = 100
    call_leg = OptionLeg(option_type='call', strike=105, quantity=1, expiration=30/365, is_call=True, is_long=False)
    strategy = CoveredCallStrategy(stock_price, stock_quantity, call_leg)

    # Simulate payoff at different stock prices at expiration
    prices = np.linspace(80, 130, 11)
    for price in prices:
        print(f"Stock price: {price}, P/L: {strategy.profit_loss(price):.2f}")
