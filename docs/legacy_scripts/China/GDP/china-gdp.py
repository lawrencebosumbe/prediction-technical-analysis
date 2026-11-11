import requests
import matplotlib.pyplot as plt

# Example URL for GDP data
url = 'https://brains.tradingeconomics.com/v2/search/wb,fred,comtrade?q=china gdp&pp=50&p=0&_=1557934352427&stance=2'

# Fetch the data
response = requests.get(url)
data = response.json()

# Example code to process and plot data
dates = [entry['date'] for entry in data]
values = [entry['value'] for entry in data]

plt.figure(figsize=(10, 5))
plt.plot(dates, values, marker='o')
plt.title('China GDP Over Time')
plt.xlabel('Date')
plt.ylabel('GDP')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
