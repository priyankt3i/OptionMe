# 📈   OptionMe 📉 - An Investment Strategy Analyzer

## Overview

This project is an investment strategy analysis tool that provides predictions and outcomes for various investment types, including direct stock shares and options trading strategies. It leverages real-time and historical market data, advanced forecasting models, and sentiment analysis to deliver data-driven insights.

---

## Models and Prediction Algorithms

### 1. Historical Data Analysis

- **Location:** `models/historical_data_analysis.py`
- **Description:** Calculates average growth rates over multiple time horizons (3 months, 6 months, 1 year, 5 years) based on historical price data.
- **Data Source:** Uses real historical price data fetched from Polygon.io.
- **Usage:** Provides baseline growth multipliers for other models and outcome calculations.

### 2. ARIMA Forecasting

- **Location:** `models/arima_forecasting.py`
- **Description:** Uses the ARIMA (AutoRegressive Integrated Moving Average) model to forecast future prices for specified periods.
- **Data Source:** Operates on real historical price data.
- **Enhancements:** Fixed indexing to avoid errors when accessing forecasted values.

### 3. Monte Carlo Simulation

- **Location:** `models/monte_carlo_simulation.py`
- **Description:** Simulates multiple possible future price paths using stochastic processes based on volatility and drift.
- **Data Source:** Uses current price and volatility parameters; randomness is intrinsic to the method.
- **Usage:** Provides probabilistic growth estimates over future periods.

### 4. Options Pricing Models

- **Location:** `models/options_pricing.py`
- **Description:** Implements Black-Scholes and binomial tree models for pricing options.
- **Usage:** Used as building blocks for options trading strategy evaluations.

### 5. Options Trading Strategies

- **Location:** `models/options_strategies.py`
- **Description:** Modular framework representing individual option legs and multi-leg strategies.
- **Implemented Strategy:** Covered Call (long stock + short call option) with payoff and profit/loss calculations.
- **Future Work:** Additional strategies to be implemented following the modular design.

### 6. T3i Prediction Model

- **Location:** `models/t3i_prediction.py`
- **Description:** A data-driven prediction model that combines historical price trends with real-time news sentiment from Polygon.io's Ticker News API (using LLM-powered sentiment analysis) and social media trend placeholders.
- **Data Source:** Real historical prices and Polygon.io news sentiment.
- **Purpose:** Provides enhanced growth rate predictions incorporating market sentiment and trends.

---

## Data Sources

- **Polygon.io API:** Used extensively for fetching real-time and historical stock prices, ticker lists, and news with sentiment analysis.
- **Local Simulation:** Monte Carlo simulations use stochastic modeling based on real parameters.
- **Placeholders:** Social media trend analysis is currently a placeholder for future integration.

---

## Application Features

- Dynamic stock list fetched from Polygon.io for user selection.
- Multiple prediction models selectable by the user.
- Detailed options trading strategy analysis with modular design.
- Interactive UI with investment parameters and strategy-specific inputs.
- Visualizations of historical data and predicted price growth.
- Regulatory disclaimer included for compliance.

---

## Future Enhancements

- **Expand Options Strategies:** Implement detailed logic for all listed options trading strategies beyond Covered Call.
- **Social Media Integration:** Incorporate real-time social media sentiment and trend analysis.
- **Advanced ML Models:** Integrate machine learning models for improved prediction accuracy.
- **User Authentication:** Add user profiles and saved strategies.
- **Performance Optimization:** Improve caching and API call efficiency.
- **Error Handling:** Enhance robustness against API rate limits and data unavailability.
- **UI Improvements:** Add more interactive charts and detailed strategy explanations.

---

## Setup and Usage

1. Clone the repository.
2. Create a `.env` file with your Polygon.io API key:
   ```
   POLYGON_API_KEY=your_api_key_here
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the app:
   ```
   streamlit run app.py
   ```
5. Use the UI to select stocks, investment types, prediction models, and options strategies.

---

## License

This project is provided as-is for educational and demonstration purposes.

---

## Contact

For questions or support, please contact [Kumar Priyank] (https://github.com/priyankt3i).
