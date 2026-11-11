import matplotlib
matplotlib.use('TkAgg')  # Set backend

import pandas as pd
import matplotlib.pyplot as plt

#------------------------------
#Step 1: Clean & Load the data
#------------------------------

file_path = 'BTC_USD Bitfinex Historical Data.csv'

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

# Get the last value of the SMA to predict the next 10 days
last_sma = data['SMA_10'].iloc[-1]

# Predicting the next 10 days based on the last SMA value
predicted_prices = [last_sma] * 10

# Generate the next 10 dates (business days)
future_dates = pd.bdate_range(start=data.index[-1], periods=10)

# Create a DataFrame to hold the predictions
future_predictions = pd.DataFrame({'Predicted_Price': predicted_prices}, index=future_dates)

# Display the predictions
print(future_predictions)

# Plot historical data with future predictions
plt.figure(figsize=(10, 6))
plt.plot(data['Price'], label='Historical Price')
plt.plot(data['SMA_10'], label='10-Day SMA', color='red')
plt.plot(future_predictions.index, future_predictions['Predicted_Price'], label='Predicted Price', color='green', linestyle='dashed')
plt.title('BitCoin Bitfinex Average Price Prediction for Next 10 Days')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()