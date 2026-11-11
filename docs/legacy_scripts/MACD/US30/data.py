import matplotlib
matplotlib.use('TkAgg')  # Set backend

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.table import Table
from statsmodels.tsa.arima.model import ARIMA

# Load the CSV file
file_path = 'US 30 Cash Historical Data.csv'
data = pd.read_csv(file_path)

# Clean and preprocess the data
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')

for column in ['Price', 'Open', 'High', 'Low']:
    data[column] = data[column].str.replace(',', '').astype(float)

data['Change %'] = data['Change %'].str.replace('%', '').astype(float)

data.sort_values('Date', inplace=True)
data.reset_index(drop=True, inplace=True)

price_data = data['Price']

model = ARIMA(price_data, order=(5, 1, 0))
model_fit = model.fit()

forecast = model_fit.forecast(steps=10)
forecast_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=10, freq='B')

forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predicted Price': forecast.round(2)})

# Plot the historical and forecasted data
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the data
ax.plot(data['Date'], data['Price'], label='Historical Prices', color='blue')
ax.plot(forecast_df['Date'], forecast_df['Predicted Price'], label='Forecasted Prices', linestyle='--', color='red')

# Chart settings
ax.set_title('Market Price Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.legend()
ax.grid(True)

# Add the table below the chart with a margin
table = plt.table(cellText=forecast_df.values,
                  colLabels=forecast_df.columns,
                  cellLoc='center',
                  loc='bottom',
                  colColours=['#f2f2f2', '#f2f2f2'],
                  colWidths=[0.2, 0.2],
                  bbox=[0, -0.5, 1, 0.3])  # bbox positions the table with margin

# Style adjustments for the table

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)
table.auto_set_column_width([0, 1])

# Adjust the plot layout to provide space between the chart and table
plt.subplots_adjust(left=0.1, bottom=0.25)  # Adjust this to add space below the chart

plt.show()
