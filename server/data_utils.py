from file_utils import open_data, save_data, get_data_filepath
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def fetch_data(symbol, output_size="full", api_key="demo"):
    if not symbol:
        raise ValueError("symbol parameter is required and cannot be empty")

    filename = f"{symbol}_{output_size}"

    data = open_data(symbol, filename)

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
        save_data(data, symbol, filename)

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


def create_window(data, lookback):
    X, y = [], []

    for i in range(len(data) - lookback - 1):
        X.append(data["return"].iloc[i : i + lookback].values)
        y.append(data["return"].iloc[i + lookback : i + lookback + 1].values)

    return np.array(X), np.array(y)


def temporal_split(data, split_size):
    split = int(len(data) * split_size)

    return data[:split], data[split:]


def prepare_data(data):
    processed_data = data.copy()
    processed_data["return"] = processed_data["close"].pct_change()
    processed_data = processed_data.dropna(subset=["return"])

    X, y = create_window(processed_data, 5)

    X_train_val, X_test = temporal_split(X, 0.9)
    X_train, X_val = temporal_split(X_train_val, 0.9)

    y_train_val, y_test = temporal_split(y, 0.9)
    y_train, y_val = temporal_split(y_train_val, 0.9)

    return (X_train, X_val, X_test, y_train, y_val, y_test, processed_data)
