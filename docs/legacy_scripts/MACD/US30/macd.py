import matplotlib
matplotlib.use('TkAgg')  # Set backend

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.table import Table
from statsmodels.tsa.arima.model import ARIMA

# Load the CSV file
file_path = 'US 30 Cash Historical Data.csv'
data = pd.read_csv(file_path)

# Convert the 'Date' column to datetime format and sort by date in ascending order
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
data.sort_values('Date', ascending=True, inplace=True)

# Remove commas from the 'Price' column and convert it to a numeric type
data['Price'] = data['Price'].str.replace(',', '').astype(float)

# Calculate the 12-period EMA and 26-period EMA for the 'Price' column
data['12_EMA'] = data['Price'].ewm(span=12, adjust=False).mean()
data['26_EMA'] = data['Price'].ewm(span=26, adjust=False).mean()

# Calculate the MACD line (12_EMA - 26_EMA)
data['MACD'] = data['12_EMA'] - data['26_EMA']

# Calculate the 9-period EMA of the MACD line to get the Signal line
data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

# Filter the data for the specific dates: Aug 21, 2024, Aug 20, 2024, Aug 19, 2024
specific_dates = data[(data['Date'] == '2024-08-21') |
                      (data['Date'] == '2024-08-20') |
                      (data['Date'] == '2024-08-19')]

# Display the relevant data
print(specific_dates[['Date', 'Price', '12_EMA', '26_EMA', 'MACD', 'Signal_Line']])
