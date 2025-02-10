import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px

# Step 1: Load the dataset
df = pd.read_csv('Data.csv')  # Replace with your actual CSV file path

# Step 2: Convert 'Date' column to datetime format and handle any errors in conversion
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Step 3: Remove rows with invalid or missing 'Date' values
df = df.dropna(subset=['Date'])

# Step 4: Set 'Date' as the index and ensure there are no duplicate dates
df.set_index('Date', inplace=True)
df = df[~df.index.duplicated(keep='last')]

# Step 5: Sort the DataFrame by Date (ascending order)
df.sort_index(inplace=True)

# Step 6: Forward fill missing values in the dataset (e.g., for missing stock prices)
df.fillna(method='ffill', inplace=True)

# Data Overview
print("Data Overview:")
print(df.head())

# Step 7: Descriptive Statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Step 8: Calculate daily returns (percentage change from previous day)
df['Daily Return'] = df['Close'].pct_change()

# Step 9: Calculate MACD (Moving Average Convergence Divergence) and Signal Line
short_ema = df['Close'].ewm(span=12, adjust=False).mean()
long_ema = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = short_ema - long_ema
df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Step 10: Calculate Bollinger Bands (upper and lower bands for volatility)
rolling_mean = df['Close'].rolling(window=20).mean()
rolling_std = df['Close'].rolling(window=20).std()
df['Upper Band'] = rolling_mean + (rolling_std * 2)
df['Lower Band'] = rolling_mean - (rolling_std * 2)

# Step 11: Calculate RSI (Relative Strength Index)
delta = df['Close'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# Step 12: Calculate Exponential Moving Average (EMA) for 50 and 200 periods
df['50-day EMA'] = df['Close'].ewm(span=50, adjust=False).mean()
df['200-day EMA'] = df['Close'].ewm(span=200, adjust=False).mean()

# Step 13: Portfolio Analysis (Sharpe Ratio)
returns = df['Daily Return'].mean() * 252  # Annualized return
portfolio_risk = df['Daily Return'].std() * np.sqrt(252)  # Annualized risk (volatility)
sharpe_ratio = returns / portfolio_risk  # Sharpe ratio: return / risk
print(f"\nPortfolio Sharpe Ratio: {sharpe_ratio}")

# Step 14: ARIMA Forecasting (predicting future stock prices)
model = ARIMA(df['Close'], order=(5, 1, 0))  # ARIMA model (example order, adjust based on ACF/PACF)
model_fit = model.fit()
forecast = model_fit.forecast(steps=10)  # Forecast the next 10 days
print("\nARIMA Forecasting (Next 10 Days):")
print(forecast)

# Step 15: Monte Carlo Simulation for Stock Prices
num_simulations = 1000  # Number of simulations
num_days = 252  # 1 year of trading days
simulated_prices = np.zeros((num_simulations, num_days))

# Simulate price movements based on daily returns
for i in range(num_simulations):
    daily_returns = np.random.normal(df['Daily Return'].mean(), df['Daily Return'].std(), num_days)
    price_series = df['Close'].iloc[-1] * (1 + daily_returns).cumprod()  # Apply daily returns to initial price
    simulated_prices[i] = price_series
    
# Step 16: Visualize Monte Carlo Simulation results
plt.figure(figsize=(14, 7))
plt.plot(simulated_prices.T, color='blue', alpha=0.1)
plt.title("Monte Carlo Simulation of Stock Prices")
plt.xlabel("Days")
plt.ylabel("Stock Price ($)")
plt.show()

# Step 17: Backtesting a Moving Average Crossover Strategy
df['7-day MA'] = df['Close'].rolling(window=7).mean()
df['30-day MA'] = df['Close'].rolling(window=30).mean()

# Generate buy/sell signals based on the crossover strategy
df['Signal'] = 0
df['Signal'][df['7-day MA'] > df['30-day MA']] = 1  # Buy signal: 7-day MA crosses above 30-day MA
df['Signal'][df['7-day MA'] < df['30-day MA']] = -1  # Sell signal: 7-day MA crosses below 30-day MA

# Calculate strategy returns
df['Strategy Return'] = df['Daily Return'] * df['Signal'].shift(1)  # Lagging signal by 1 day for correct backtest
df['Cumulative Strategy Return'] = (1 + df['Strategy Return']).cumprod() - 1  # Cumulative return of the strategy

# Step 18: Visualize Cumulative Strategy Return vs. Market Return
plt.figure(figsize=(14, 7))
sns.lineplot(x=df.index, y=df['Cumulative Strategy Return'], label='Strategy Return')
sns.lineplot(x=df.index, y=(1 + df['Daily Return']).cumprod() - 1, label='Market Return')
plt.title("Cumulative Returns: Strategy vs. Market")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Step 19: Plot Technical Indicators (RSI and MACD)
plt.figure(figsize=(14, 7))

# Step 20: Interactive Visualization with Plotly for Closing Price and Volume
fig1 = px.line(df, x=df.index, y='Close', title='Stock Prices Over Time', color='Symbol')
fig1.show()

fig2 = px.line(df, x=df.index, y='Volume', title='Trading Volume Over Time', color='Symbol')
fig2.show()

# Step 21: Save the analyzed data to a new CSV file
df.to_csv("analyzed_stock_data.csv")

print("\nAnalysis Complete. The data has been saved as 'analyzed_stock_data.csv'.")