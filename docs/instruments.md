# Instruments

Bundled daily sample histories are under `data/raw`. They are small snapshots from August 2024 and are provided to demonstrate loading, indicators, charts, and the CLI. They are not a substitute for current or complete market data.

| Symbol | Instrument | Source file |
| --- | --- | --- |
| US30 | Dow Jones Industrial Average | `us30.csv` |
| US500 | S&P 500 | `us500.csv` |
| US100 | Nasdaq 100 | `us100.csv` |
| CAC40 | CAC 40 | `cac40.csv` |
| CHINA50 | FTSE China A50 | `china50.csv` |
| DAX30 | DAX | `dax30.csv` |
| FTSE100 | FTSE 100 | `ftse100.csv` |
| NIKKEI225 | Nikkei 225 | `nikkei225.csv` |
| STOXX50E | Euro Stoxx 50 | `stoxx50e.csv` |
| BTCUSD | Bitcoin / US Dollar | `btcusd.csv` |

China GDP is available separately through `tpta.data.gdp.load_china_gdp`, which reads annual current-dollar GDP from the World Bank API and therefore needs network access.

