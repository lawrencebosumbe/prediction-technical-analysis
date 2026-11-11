import matplotlib
matplotlib.use('TkAgg')  # Set backend

import pandas as pd
import matplotlib.pyplot as plt

#-----------------------
#Step 1: Clean & Load the data
#-----------------------

file_path = 'Dow Jones Industrial Average Historical Data - 09-08-2024.csv'

data = pd.read_csv(file_path)

# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Clean the 'Price', 'Open', 'High', 'Low' columns
columns_to_clean = ['Price', 'Open', 'High', 'Low']

for column in columns_to_clean:
    data[column] = data[column].str.replace(',', '').astype(float)

# Set the 'Date' column as the index
data.set_index('Date', inplace=True)

# Sort the data by date
data.sort_index(inplace=True)

# Calculate a 10-day simple moving average (SMA)
data['SMA_10'] = data['Price'].rolling(window=10).mean()

# Plot the closing prices and SMA
plt.figure(figsize=(10, 6))
plt.plot(data['Price'], label='Price')
plt.plot(data['SMA_10'], label='10-Day SMA', color='red')
plt.title('Dow Jones Industrial Average Price with 10-Day SMA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
