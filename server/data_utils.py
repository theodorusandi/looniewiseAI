from file_utils import open_data, save_data
import requests
import pandas as pd


def fetch_data(symbol, output_size="full", api_key="demo"):
    if not symbol:
        raise ValueError("symbol parameter is required and cannot be empty")

    filename = f"{symbol}_{output_size}"

    data = open_data(filename)

    if not data:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize={output_size}&apikey={api_key}"
        print(f"Fetching data for {symbol} from API...")
        r = requests.get(url)
        if r.status_code != 200:
            raise Exception(f"API request failed with status code {r.status_code}")
        elif "Error Message" in r.json():
            raise Exception(f"API request error: {r.json()['Error Message']}")
        elif "Information" in r.json():
            raise Exception(f"API request info: {r.json()['Information']}")
        data = r.json()
        save_data(data, filename)

    time_series = [
        {
            "date": pd.to_datetime(key),
            "open": float(value["1. open"]),
            "high": float(value["2. high"]),
            "low": float(value["3. low"]),
            "close": float(value["4. close"]),
            "volume": int(value["5. volume"]),
        }
        for key, value in data["Time Series (Daily)"].items()
    ]

    time_series_df = pd.DataFrame(time_series).sort_values(by="date").set_index("date")

    return time_series_df
