![Screenshot from 2024-11-29 17-39-27](https://github.com/user-attachments/assets/f019cf1f-199c-477b-b5f8-e604b0738b96)
![Screenshot from 2024-11-29 17-39-20](https://github.com/user-attachments/assets/4123aba8-44b0-4899-9e26-3514e5760ca7)
![Screenshot from 2024-11-29 17-38-50](https://github.com/user-attachments/assets/b7dbbf63-41b5-4de5-85c5-7e0dedd54528)
![Screenshot from 2024-11-29 17-38-44](https://github.com/user-attachments/assets/9d850190-9479-49dc-8667-5b5c629c4d2d)
![Screenshot from 2024-11-29 17-38-37](https://github.com/user-attachments/assets/174c6778-3c7a-4e2b-be00-4495edc33708)

### 1. **Importing Libraries**
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px
```
- **pandas**: Used for data manipulation, specifically for reading and handling the dataset.
- **numpy**: Useful for numerical operations, like generating random values and calculating statistics.
- **matplotlib.pyplot**: For basic plotting of graphs and charts.
- **seaborn**: Extends `matplotlib` for statistical plots (such as lineplots).
- **statsmodels.tsa.arima.model**: Used for time series analysis and forecasting, specifically for ARIMA models.
- **plotly.express**: For interactive plots, useful for detailed visualizations.

### 2. **Load the dataset**
```python
df = pd.read_csv('File Path !')  # Replace with your actual CSV file path
```
- This line reads the CSV file containing stock data into a DataFrame `df`. The placeholder `'File Path !'` should be replaced with the actual path to your data file.

### 3. **Convert 'Date' column to datetime**
```python
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
```
- Converts the 'Date' column to a `datetime` format. Any errors (like invalid date formats) are coerced to `NaT` (Not a Time), meaning these rows are treated as missing values.

### 4. **Remove rows with invalid or missing 'Date' values**
```python
df = df.dropna(subset=['Date'])
```
- Drops rows where the 'Date' column has missing or invalid values (`NaT`), ensuring the DataFrame contains valid dates only.

### 5. **Set 'Date' as the index and handle duplicates**
```python
df.set_index('Date', inplace=True)
df = df[~df.index.duplicated(keep='last')]
```
- Sets the 'Date' column as the index of the DataFrame for time-based operations.
- Removes any duplicate dates, keeping only the most recent entry for each date.

### 6. **Sort the DataFrame by Date**
```python
df.sort_index(inplace=True)
```
- Sorts the DataFrame by the index (which is 'Date'), ensuring that the data is ordered chronologically.

### 7. **Forward fill missing values**
```python
df.fillna(method='ffill', inplace=True)
```
- Forward fills missing values (i.e., fills missing values with the previous valid value). This is often used for stock price data to carry forward the last known price.

### 8. **Data Overview**
```python
print("Data Overview:")
print(df.head())
```
- Displays the first few rows of the dataset for an overview.

### 9. **Descriptive Statistics**
```python
print("\nDescriptive Statistics:")
print(df.describe())
```
- Displays descriptive statistics (mean, median, standard deviation, etc.) of the numerical columns in the DataFrame.

### 10. **Calculate Daily Returns**
```python
df['Daily Return'] = df['Close'].pct_change()
```
- Calculates the daily return (percentage change in the 'Close' price from the previous day).

### 11. **Calculate MACD (Moving Average Convergence Divergence)**
```python
short_ema = df['Close'].ewm(span=12, adjust=False).mean()
long_ema = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = short_ema - long_ema
df['Signal Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
```
- Calculates the MACD (short-term minus long-term Exponential Moving Average) to identify momentum trends.
- The signal line is the 9-day EMA of the MACD, which helps in signaling potential buy/sell decisions.

### 12. **Calculate Bollinger Bands**
```python
rolling_mean = df['Close'].rolling(window=20).mean()
rolling_std = df['Close'].rolling(window=20).std()
df['Upper Band'] = rolling_mean + (rolling_std * 2)
df['Lower Band'] = rolling_mean - (rolling_std * 2)
```
- Calculates the Bollinger Bands, which are two lines that represent the upper and lower volatility bounds around the moving average of the closing price. Typically, a 20-day window is used for these calculations.

### 13. **Calculate RSI (Relative Strength Index)**
```python
delta = df['Close'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))
```
- RSI measures the magnitude of recent price changes to evaluate overbought or oversold conditions in a market.
- It is calculated using a 14-day rolling window, and the result is between 0 and 100, where values above 70 indicate overbought conditions, and below 30 indicate oversold conditions.

### 14. **Calculate EMA for 50 and 200 periods**
```python
df['50-day EMA'] = df['Close'].ewm(span=50, adjust=False).mean()
df['200-day EMA'] = df['Close'].ewm(span=200, adjust=False).mean()
```
- The 50-day and 200-day EMAs are widely used in technical analysis to identify trends in the stock price.

### 15. **Portfolio Analysis (Sharpe Ratio)**
```python
returns = df['Daily Return'].mean() * 252  # Annualized return
portfolio_risk = df['Daily Return'].std() * np.sqrt(252)  # Annualized risk (volatility)
sharpe_ratio = returns / portfolio_risk  # Sharpe ratio: return / risk
print(f"\nPortfolio Sharpe Ratio: {sharpe_ratio}")
```
- The Sharpe ratio is used to measure the risk-adjusted return of the portfolio. A higher Sharpe ratio indicates better risk-adjusted performance.

### 16. **ARIMA Forecasting**
```python
model = ARIMA(df['Close'], order=(5, 1, 0))  # ARIMA model (example order, adjust based on ACF/PACF)
model_fit = model.fit()
forecast = model_fit.forecast(steps=10)  # Forecast the next 10 days
print("\nARIMA Forecasting (Next 10 Days):")
print(forecast)
```
- ARIMA is used for time series forecasting. The parameters `(5, 1, 0)` indicate the model’s order (autoregressive terms, differencing, and moving average terms).
- Forecasts the next 10 days of stock prices.

### 17. **Monte Carlo Simulation**
```python
# Simulate price movements based on daily returns
```
- The Monte Carlo simulation simulates a large number of potential price paths based on the historical daily returns, which can be used to predict potential future prices under uncertainty.

### 18. **Visualizing Monte Carlo Simulation Results**
```python
plt.plot(simulated_prices.T, color='blue', alpha=0.1)
```
- Visualizes the simulated stock prices over time, showing a variety of possible price paths.

### 19. **Backtesting a Moving Average Crossover Strategy**
```python
# Generate buy/sell signals based on the crossover strategy
```
- Implements a simple trading strategy using two moving averages (7-day and 30-day). The strategy buys when the short-term MA crosses above the long-term MA and sells when it crosses below.

### 20. **Plotting Cumulative Strategy Return vs. Market Return**
```python
sns.lineplot(x=df.index, y=df['Cumulative Strategy Return'], label='Strategy Return')
sns.lineplot(x=df.index, y=(1 + df['Daily Return']).cumprod() - 1, label='Market Return')
```
- Compares the performance of the moving average strategy against simply holding the stock (market return).

### 21. **Interactive Visualization with Plotly**
```python
fig1 = px.line(df, x=df.index, y='Close', title='Stock Prices Over Time', color='Symbol')
```
- Uses Plotly to create an interactive line chart of the closing stock price over time.

### 22. **Save the analyzed data to CSV**
```python
df.to_csv("analyzed_stock_data.csv")
```
- Saves the DataFrame with all the technical indicators and analysis results to a new CSV file.

---
