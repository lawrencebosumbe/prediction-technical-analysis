import matplotlib
matplotlib.use('TkAgg')  # Set backend

import pandas as pd
import matplotlib.pyplot as plt

#-----------------------
#Step 1: Clean & Load the data
#-----------------------

file_path = 'FTSE 100 Historical Data.csv'

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

# Calculate the MACD
data['EMA_12'] = data['Price'].ewm(span=12, adjust=False).mean()
data['EMA_26'] = data['Price'].ewm(span=26, adjust=False).mean()
data['MACD'] = data['EMA_12'] - data['EMA_26']
data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

# Plot the MACD with the Signal Line
plt.figure(figsize=(10, 6))
plt.plot(data['MACD'], label='MACD', color='blue')
plt.plot(data['Signal_Line'], label='Signal Line', color='orange')
plt.title('MACD for FTSE 100 (CFD)')
plt.xlabel('Date')
plt.ylabel('MACD')
plt.legend()
plt.show()
plt.show()