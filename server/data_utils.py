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


# # #


def create_window(data, lookback):
    X, y = [], []

    for i in range(len(data) - lookback):
        X.append(data.iloc[i : i + lookback].values)
        y.append(data.iloc[i + lookback : i + lookback + 1])

    return np.array(X), np.array(y)


def temporal_split(data, split_size):
    split = int(len(data) * split_size)

    return data[:split], data[split:]


def prepare_data(data, lookback):
    processed_data = data.copy()

    X, y = create_window(processed_data["close"], lookback)

    X = X.reshape(X.shape[0], X.shape[1], 1)

    X_train_val, X_test = temporal_split(X, 0.9)
    X_train, X_val = temporal_split(X_train_val, 0.9)

    y_train_val, y_test = temporal_split(y, 0.9)
    y_train, y_val = temporal_split(y_train_val, 0.9)

    X_scaler = MinMaxScaler(feature_range=(0, 1))

    X_train = X_scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    X_val = X_scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
    X_test = X_scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

    y_scaler = MinMaxScaler(feature_range=(0, 1))

    y_train = y_scaler.fit_transform(y_train)
    y_val = y_scaler.transform(y_val)
    y_test = y_scaler.transform(y_test)

    training_data = (X_train, y_train)
    validation_data = (X_val, y_val)
    testing_data = (X_test, y_test)
    scalers = (X_scaler, y_scaler)

    return (training_data, validation_data, testing_data, scalers)
