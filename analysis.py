import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px

# Step 1: Load the dataset
df = pd.read_csv('File Path !')  # Replace with your actual CSV file path

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