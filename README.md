# 📊 Stock Data Analysis & Forecasting

Welcome to the **Stock Data Analysis & Forecasting** project! 📈 This Python-based tool leverages various analytical techniques to provide insights into stock market trends. It integrates time series forecasting, technical indicators, portfolio analysis, and even Monte Carlo simulations to project future stock prices. Whether you're a trader, analyst, or just someone interested in data science, this project will help you understand stock price behavior and make informed decisions.

### 🛠️ Features:
- 📅 **Data Preprocessing**: Clean and format stock data, including handling missing values and ensuring the correct date format.
- 📈 **Technical Indicators**: Calculate key indicators like MACD, RSI, Bollinger Bands, and Exponential Moving Averages (EMA).
- 📉 **Portfolio Analysis**: Calculate important portfolio metrics such as the Sharpe ratio.
- 🔮 **ARIMA Forecasting**: Use the ARIMA model for time series forecasting of future stock prices.
- 🏝️ **Monte Carlo Simulation**: Simulate thousands of potential future stock price paths to understand price volatility.
- 📊 **Backtesting Strategies**: Test the effectiveness of trading strategies, such as moving average crossovers, with backtesting.
- 📊 **Interactive Visualizations**: Visualize stock prices, trading volumes, and technical indicators using **Plotly** and **Matplotlib**.
- 💾 **Save Results**: Export the processed data and analysis to a new CSV file.

---

### 🧑‍💻 Getting Started:
To run this analysis, you'll need to have Python installed along with the necessary libraries.

#### 📥 Prerequisites:
1. **Python 3.x** installed
2. Install the required libraries using `pip`:
    ```bash
    pip install pandas numpy matplotlib seaborn statsmodels plotly
    ```

#### 📂 How to Run:
1. **Clone** or **Download** this repository to your local machine.
2. **Update the File Path**:
    - Replace `'File Path !'` in the code with the path to your stock data CSV file. The CSV should contain a **Date** column and a **Close** price column (other columns like 'Volume' can be included for additional analysis).
3. **Run the Script**:
    ```bash
    python stock_analysis.py
    ```
4. **Results**: The script will output a variety of visualizations, statistics, and forecasts. The processed data will be saved as a new CSV file (`analyzed_stock_data.csv`).

---

### 🧐 Code Breakdown:

Here’s a step-by-step explanation of what the script does:

#### 1. **Data Loading & Preprocessing**:
   - Load the dataset from a CSV file.
   - Convert the **Date** column to a datetime format and remove any invalid or missing dates.
   - Forward fill missing values in the dataset to ensure completeness.

```python
df = pd.read_csv('File Path !')
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])
df.set_index('Date', inplace=True)
df = df[~df.index.duplicated(keep='last')]
df.sort_index(inplace=True)
df.fillna(method='ffill', inplace=True)
```

#### 2. **Descriptive Statistics**:
   - The code prints the first few rows and descriptive statistics of the data, giving you a summary of the stock's historical performance.

```python
print(df.head())
print(df.describe())
```

#### 3. **Technical Indicators**:
   - Calculate popular technical indicators:
     - **MACD (Moving Average Convergence Divergence)**.
     - **RSI (Relative Strength Index)**.
     - **Bollinger Bands** for volatility.
     - **50-day and 200-day Exponential Moving Averages**.
     
```python
df['MACD'] = short_ema - long_ema
df['RSI'] = 100 - (100 / (1 + rs))
df['Upper Band'] = rolling_mean + (rolling_std * 2)
df['Lower Band'] = rolling_mean - (rolling_std * 2)
```

#### 4. **Portfolio Analysis**:
   - Calculate the **Sharpe ratio** based on daily returns to assess the risk-adjusted return of the stock.

```python
sharpe_ratio = returns / portfolio_risk
print(f"Portfolio Sharpe Ratio: {sharpe_ratio}")
```

#### 5. **ARIMA Forecasting**:
   - Use the ARIMA model to forecast the stock's price for the next 10 days.

```python
forecast = model_fit.forecast(steps=10)
print(forecast)
```

#### 6. **Monte Carlo Simulation**:
   - Simulate 1000 possible future price paths based on historical returns. The simulation helps visualize the potential volatility and price movements.

```python
simulated_prices[i] = price_series
```

#### 7. **Backtesting Moving Average Strategy**:
   - Implement and backtest a **Moving Average Crossover** strategy.
   - Generate buy and sell signals based on the 7-day and 30-day moving averages.

```python
df['Signal'] = 0
df['Signal'][df['7-day MA'] > df['30-day MA']] = 1  # Buy signal
df['Signal'][df['7-day MA'] < df['30-day MA']] = -1  # Sell signal
```

#### 8. **Visualization**:
   - Visualize the **Monte Carlo simulations**, **cumulative returns** of the strategy versus the market, and technical indicators using both **Matplotlib** and **Plotly**.

```python
plt.plot(simulated_prices.T, color='blue', alpha=0.1)
sns.lineplot(x=df.index, y=df['Cumulative Strategy Return'])
fig1 = px.line(df, x=df.index, y='Close', title='Stock Prices Over Time')
fig2 = px.line(df, x=df.index, y='Volume', title='Trading Volume Over Time')
```

#### 9. **Export the Data**:
   - Save the processed and analyzed data to a new CSV file (`analyzed_stock_data.csv`).

```python
df.to_csv("analyzed_stock_data.csv")
```

---

### 📈 Visualizations:
- **Stock Price Over Time**: Interactive plot of stock closing prices.
- **Volume Over Time**: Interactive plot of trading volumes.
- **Monte Carlo Simulation**: Visualize the possible future paths of the stock price.
- **Cumulative Strategy Return vs. Market Return**: A comparison of your strategy's cumulative return versus the market.

---

### 🔧 Future Improvements:
- **Sentiment Analysis**: Incorporate news and social media sentiment data to improve forecasting.
- **Advanced Backtesting**: Implement other strategies, such as MACD crossovers or RSI-based strategies.
- **Machine Learning Models**: Explore machine learning models for predicting stock prices beyond ARIMA.

---

### 💬 Feedback & Contributions:
- Contributions are welcome! Feel free to fork this repository, submit issues, or make pull requests to improve the analysis or add new features.

---

### 🙏 Thank You:
Thank you for exploring the **Stock Data Analysis & Forecasting** project! We hope this tool helps you gain valuable insights and make better decisions in the stock market. Happy analyzing! 📊
