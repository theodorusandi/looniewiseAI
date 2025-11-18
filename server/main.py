import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from config import CONFIG
from data_utils import fetch_data, prepare_data
from model_utils import build_model, train_model, evaluate_model, predict_future


def main():
    symbol = CONFIG["symbol"]
    output_size = CONFIG["output_size"]
    api_key = CONFIG["api_key"]
    lookback = CONFIG["lookback"]

    print("fetching data...")
    data = fetch_data(symbol, output_size=output_size, api_key=api_key)
    print("data fetched successfully")

    print("=" * 50)

    print("preparing data...")
    # (training_data, validation_data, testing_data, input_shape, y_scaler) = (
    #     prepare_data(data, lookback=lookback)
    # )
    prepare_data(data, lookback=lookback)
    print("data prepared successfully")

    # print("=" * 50)

    # print("building model...")
    # model = build_model(input_shape)
    # print("model built successfully")

    # print("=" * 50)

    # print("training model...")
    # trained_model = train_model(
    #     model, training_data=training_data, validation_data=validation_data
    # )
    # print("training complete")

    # print("=" * 50)

    # print("evaluating model...")
    # evaluate_model(
    #     trained_model, testing_data=testing_data, symbol=symbol, scaler=y_scaler
    # )
    # print("")

    # print("=" * 50)
    # predict_future(
    #     trained_model,
    #     data=data,
    #     lookback=lookback,
    #     symbol=symbol,
    #     scaler=y_scaler,
    # )
    # print("predicting future values...")


if __name__ == "__main__":
    main()
