"""World Bank GDP data loader for the original China macroeconomic example."""

from __future__ import annotations

import json
from urllib.request import Request, urlopen

import pandas as pd


def load_china_gdp(start_year: int = 1960, end_year: int = 2025, timeout: float = 15.0) -> pd.DataFrame:
    """Fetch China's annual GDP (current US dollars) from the World Bank API."""
    if start_year > end_year or timeout <= 0:
        raise ValueError("year range or timeout is invalid")
    url = (
        "https://api.worldbank.org/v2/country/CHN/indicator/NY.GDP.MKTP.CD"
        f"?format=json&per_page=1000&date={start_year}:{end_year}"
    )
    request = Request(url, headers={"User-Agent": "prediction-technical-analysis/0.1"})
    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        raise ValueError("World Bank returned no GDP observations")
    rows = [{"year": int(item["date"]), "gdp_usd": item["value"]} for item in payload[1] if item["value"] is not None]
    if not rows:
        raise ValueError("World Bank returned no non-null GDP values")
    return pd.DataFrame(rows).sort_values("year").set_index("year")

