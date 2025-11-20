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

    full_set, (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = (
        prepare_data(data, lookback=lookback)
    )

    # (lookback, features)
    input_shape = (lookback, 1)

    model = build_model(input_shape)

    trained_model = train_model(
        model, training_data=(X_train, y_train), validation_data=(X_val, y_val)
    )

    r2 = evaluate_model(
        trained_model,
        testing_data=(X_test, y_test),
        symbol=symbol,
        scaler=scaler,
        lookback=lookback,
    )

    if r2 > 0.55:
        last_sequence = full_set[-lookback:].reshape(1, lookback, 1)
        predict_future(
            trained_model,
            last_sequence=last_sequence,
            symbol=symbol,
            scaler=scaler,
        )
    else:
        print(f"Model R² score {r2:.3f} is below threshold. Future prediction skipped.")


if __name__ == "__main__":
    main()
