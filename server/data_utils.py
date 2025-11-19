from file_utils import open_data, save_data
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


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


# # # Data Preparation # # #
def temporal_split(data, split_size):
    split = int(len(data) * split_size)

    return data[:split], data[split:]


def create_window(data, lookback):
    X, y = [], []

    for i in range(len(data) - lookback):
        X.append(data.iloc[i : i + lookback].values)
        y.append(data.iloc[i + lookback])

    return np.array(X), np.array(y)


def prepare_data(data, lookback):
    processed_data = data.copy()["close"].dropna()

    train_val_data, test_data = temporal_split(processed_data, 0.9)
    train_data, val_data = temporal_split(train_val_data, 0.9)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data.values.reshape(-1, 1))

    X_train, y_train = create_window(train_data, lookback)
    X_val, y_val = create_window(val_data, lookback)
    X_test, y_test = create_window(test_data, lookback)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    X_train_2d = X_train.reshape(-1, 1)
    X_val_2d = X_val.reshape(-1, 1)
    X_test_2d = X_test.reshape(-1, 1)

    X_train = scaler.transform(X_train_2d).reshape(X_train.shape)
    X_val = scaler.transform(X_val_2d).reshape(X_val.shape)
    X_test = scaler.transform(X_test_2d).reshape(X_test.shape)

    y_train = scaler.transform(y_train.reshape(-1, 1)).flatten()
    y_val = scaler.transform(y_val.reshape(-1, 1)).flatten()
    y_test = scaler.transform(y_test.reshape(-1, 1)).flatten()

    return ((X_train, y_train), (X_val, y_val), (X_test, y_test), scaler)
