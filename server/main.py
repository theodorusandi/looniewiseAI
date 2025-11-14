from data_utils import fetch_data
from config import CONFIG


def main():
    symbol = CONFIG["symbol"]

    print("fetch data...")
    data = fetch_data(symbol)
    print("data fetched successfully")


if __name__ == "__main__":
    main()
