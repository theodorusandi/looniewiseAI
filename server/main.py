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

    data = fetch_data(symbol, output_size=output_size, api_key=api_key)

    (training_data, validation_data, testing_data, scaler) = prepare_data(
        data, lookback=lookback
    )

    input_shape = (lookback, 1)

    model = build_model(input_shape)

    trained_model = train_model(
        model, training_data=training_data, validation_data=validation_data
    )

    r2 = evaluate_model(
        trained_model, testing_data=testing_data, symbol=symbol, scaler=scaler
    )
    print("r2 score on test data: {:.4f}".format(r2))

    print("=" * 50)
    predict_future(
        trained_model,
        data=data,
        lookback=lookback,
        symbol=symbol,
        scaler=scaler,
    )
    print("predicting future values...")


if __name__ == "__main__":
    main()
