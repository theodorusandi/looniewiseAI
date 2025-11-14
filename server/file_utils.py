from datetime import datetime
import json
import os


def open_data(symbol, filename):
    today = datetime.now().strftime("%Y-%m-%d")
    path = f"data/{today}/{symbol}"
    file = f"{path}/{filename}.json"

    if not os.path.exists(path):
        os.makedirs(path)

    try:
        with open(file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def save_data(data, symbol, filename):
    today = datetime.now().strftime("%Y-%m-%d")
    path = f"data/{today}/{symbol}"
    file = f"{path}/{filename}.json"

    if not os.path.exists(path):
        os.makedirs(path)

    with open(file, "w") as f:
        json.dump(data, f)


def get_data_filepath(symbol):
    today = datetime.now().strftime("%Y-%m-%d")
    path = f"data/{today}/{symbol}"

    return path
