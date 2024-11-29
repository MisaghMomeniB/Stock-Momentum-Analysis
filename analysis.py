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