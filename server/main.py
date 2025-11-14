from config import CONFIG
from data_utils import fetch_data, prepare_data


def main():
    symbol = CONFIG["symbol"]

    print("fetch data...")
    data = fetch_data(symbol)
    print("data fetched successfully")

    print("=" * 50)

    print("prepare data...")
    (X_train, X_val, X_test, y_train, y_val, y_test, processed_data) = prepare_data(
        data
    )
    print("data prepared successfully")


if __name__ == "__main__":
    main()
