from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
)
from model import TransformerEncoder, LearnedPositionalEncoding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from file_utils import get_data_filepath


def build_model(input_shape):
    embed_dim = 128
    num_heads = 8
    ff_dim = 512
    num_layers = 4
    dropout_rate = 0.1
    learning_rate = 0.001

    transformer_encoder = TransformerEncoder(
        num_layers=num_layers, embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim
    )

    inputs = Input(shape=input_shape)
    x = Dense(embed_dim)(inputs)
    x = LearnedPositionalEncoding()(x)

    encoder_outputs = transformer_encoder(x)
    flatten = GlobalAveragePooling1D()(encoder_outputs)
    dropout = Dropout(dropout_rate)(flatten)
    outputs = Dense(1)(dropout)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"]
    )

    return model


def train_model(model, training_data, validation_data):
    epochs = 100
    batch_size = 8

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            cooldown=2,
        ),
    ]

    X_train, y_train = training_data

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1,
    )

    return model


def evaluate_model(model, testing_data, symbol, scaler):
    X_test, y_test = testing_data

    y_pred = model.predict(X_test)

    y_pred_inverse = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_true_inverse = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    r2 = r2_score(y_true_inverse, y_pred_inverse)

    plt.figure(figsize=(10, 5))
    plt.plot(y_true_inverse, label="True Values", alpha=0.7)
    plt.plot(y_pred_inverse, label="Predicted Values", linestyle="--")
    plt.title(f"True vs Predicted Values")
    plt.xlabel("Time Steps")
    plt.legend()
    path = get_data_filepath(symbol)
    plt.savefig(f"{path}/evaluation.png", dpi=300, bbox_inches="tight")

    print(f"R² Score: {r2:.4f}")

    return r2


def predict_future(
    model,
    data,
    lookback,
    symbol,
    scaler,
):
    close_data = data["close"].values[-lookback:]

    input_sequence = scaler.transform(close_data.reshape(-1, 1)).reshape(1, lookback, 1)

    n_simulations = 100
    num_future_steps = 5

    all_predictions = []

    for _ in range(n_simulations):
        predicted_values = []
        current_sequence = input_sequence.copy()

        for _ in range(num_future_steps):
            next_value_tf = model(current_sequence, training=True)
            next_value = next_value_tf.numpy()
            predicted_values.append(next_value.flatten()[0])
            current_sequence = np.append(
                current_sequence[:, 1:, :], next_value.reshape(1, 1, 1), axis=1
            )

        all_predictions.append(predicted_values)

    all_predictions = np.array(all_predictions)

    mean_predictions = np.mean(all_predictions, axis=0)
    std_predictions = np.std(all_predictions, axis=0)
    ci_lower = np.percentile(all_predictions, 2.5, axis=0)
    ci_upper = np.percentile(all_predictions, 97.5, axis=0)

    mean_predictions_inverse = scaler.inverse_transform(
        mean_predictions.reshape(-1, 1)
    ).flatten()
    ci_lower_inverse = scaler.inverse_transform(ci_lower.reshape(-1, 1)).flatten()
    ci_upper_inverse = scaler.inverse_transform(ci_upper.reshape(-1, 1)).flatten()

    plt.figure(figsize=(10, 5))
    time_steps = np.arange(1, num_future_steps + 1)

    error_lower = mean_predictions_inverse - ci_lower_inverse
    error_upper = ci_upper_inverse - mean_predictions_inverse

    plt.errorbar(
        time_steps,
        mean_predictions_inverse,
        yerr=[error_lower, error_upper],
        fmt="o-",
        capsize=5,
        capthick=2,
        label="Mean Prediction",
        color="blue",
        ecolor="lightblue",
        linewidth=2,
        markersize=8,
    )

    plt.fill_between(
        time_steps,
        ci_lower_inverse,
        ci_upper_inverse,
        alpha=0.2,
        color="blue",
        label="95% Confidence Interval",
    )

    plt.xlabel("Future Time Steps", fontsize=12)
    plt.ylabel("Predicted Price", fontsize=12)
    plt.title("Future Price Predictions with 95% Confidence Intervals", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = get_data_filepath(symbol)
    plt.savefig(f"{path}/evaluation.png", dpi=300, bbox_inches="tight")

    return {
        "mean": mean_predictions_inverse,
        "ci_lower": ci_lower_inverse,
        "ci_upper": ci_upper_inverse,
        "std": std_predictions,
    }
