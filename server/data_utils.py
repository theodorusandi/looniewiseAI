import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

from file_utils import open_data, save_data


def fetch_data(
    symbol: str, output_size: str = "full", api_key: str = "demo"
) -> pd.DataFrame:
    print(f"Fetching data for symbol: {symbol}")

    if not symbol:
        raise ValueError("symbol parameter is required and cannot be empty")

    filename = f"{symbol}_{output_size}"

    data = open_data(symbol, filename)

    if not data:
        print(f"Fetching data for {symbol} from API...")
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize={output_size}&apikey={api_key}"
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
def split_data(data: pd.DataFrame, split_size: float = 0.8):
    split = int(len(data) * split_size)
    return data.iloc[:split], data.iloc[split:]


def create_window(data, lookback: int):
    X, y = [], []

    for i in range(len(data) - lookback):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback])

    return np.array(X), np.array(y)


def prepare_data(data: pd.DataFrame, lookback: int):
    processed_data = data.copy()

    processed_data = processed_data[["close"]].dropna()

    training_data, test_set = split_data(processed_data, split_size=0.9)
    training_set, validation_set = split_data(training_data, split_size=0.9)

    scaler = RobustScaler().fit(training_set)

    full_set = scaler.transform(processed_data)
    training_set = scaler.transform(training_set)
    validation_set = scaler.transform(validation_set)
    test_set = scaler.transform(test_set)

    X_train, y_train = create_window(training_set, lookback)
    X_val, y_val = create_window(validation_set, lookback)
    X_test, y_test = create_window(test_set, lookback)

    return full_set, (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler
