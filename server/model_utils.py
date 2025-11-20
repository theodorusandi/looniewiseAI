import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
)

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from file_utils import get_data_filepath
from model import TransformerEncoder, LearnedPositionalEncoding


def build_model(input_shape):
    print("Building model...")

    embed_dim = 64
    num_heads = 4
    ff_dim = 256
    num_layers = 2

    final_dropout_rate = 0.1
    learning_rate = 0.0001

    transformer_encoder = TransformerEncoder(
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
    )

    inputs = Input(shape=input_shape)
    x = Dense(embed_dim)(inputs)
    x = LearnedPositionalEncoding()(x)

    encoder_outputs = transformer_encoder(x)
    flatten = GlobalAveragePooling1D()(encoder_outputs)
    dropout = Dropout(final_dropout_rate)(flatten)
    outputs = Dense(1)(dropout)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"]
    )

    return model


def tune_batch_size(training_data, validation_data, callbacks):
    print("Tuning batch size...")

    best_batch_size = None
    X_train, y_train = training_data

    results: dict = {}

    for batch_size in [8, 16]:
        print(f"Testing batch size: {batch_size}")

        model = build_model(input_shape=X_train.shape[1:])

        history = model.fit(
            X_train,
            y_train,
            epochs=20,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0,
        )

        best_val_loss = min(history.history["val_loss"])
        results[batch_size] = best_val_loss

        print(f"Batch size: {batch_size}, Best Val Loss: {best_val_loss:.3f}")

    best_batch_size = min(results, key=results.get)  # type: ignore

    print(f"Best batch size: {best_batch_size}")

    return best_batch_size


def train_model(model, training_data, validation_data):
    print("Training model...")

    epochs = 100

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

    batch_size = tune_batch_size(
        training_data=training_data,
        validation_data=validation_data,
        callbacks=callbacks,
    )

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


def evaluate_model(model, testing_data, symbol, scaler, lookback):
    print("Evaluating model...")

    X_test, y_test = testing_data

    y_pred = model.predict(X_test)

    y_pred_inverse = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    r2 = r2_score(y_test_inverse, y_pred_inverse)

    plt.plot(y_test_inverse, label="True Values", alpha=0.7)
    plt.plot(y_pred_inverse, label="Predicted Values", linestyle="--")
    plt.title(f"True vs Predicted Values")
    plt.xlabel("Time Steps")
    plt.legend()
    # Add R² score as text on the plot
    plt.text(
        0.02,
        0.98,
        f"R² Score: {r2:.3f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    path = get_data_filepath(symbol)
    plt.savefig(f"{path}/evaluation_{lookback}day(s).png", dpi=300, bbox_inches="tight")

    print(f"R² Score: {r2:.3f}")

    return r2


def predict_future(
    model,
    last_sequence,
    symbol,
    scaler,
):
    print("Predicting future values with uncertainty estimation...")

    n_simulations = 100
    num_future_steps = 10

    all_predictions = []

    for _ in range(n_simulations):
        predicted_values = []
        current_sequence = last_sequence.copy()

        for _ in range(num_future_steps):
            next_value = model(current_sequence, training=True).numpy().flatten()[0]
            predicted_values.append(next_value)
            current_sequence = np.append(
                current_sequence[:, 1:, :], next_value.reshape(1, 1, 1), axis=1
            )

        all_predictions.append(predicted_values)

    all_predictions = np.array(all_predictions)

    mean_predictions = np.mean(all_predictions, axis=0)
    ci_lower = np.percentile(all_predictions, 2.5, axis=0)
    ci_upper = np.percentile(all_predictions, 97.5, axis=0)

    mean_predictions_inverse = scaler.inverse_transform(
        mean_predictions.reshape(-1, 1)
    ).flatten()
    ci_lower_inverse = scaler.inverse_transform(ci_lower.reshape(-1, 1)).flatten()
    ci_upper_inverse = scaler.inverse_transform(ci_upper.reshape(-1, 1)).flatten()

    # Calculate average return
    returns = np.diff(mean_predictions_inverse) / mean_predictions_inverse[:-1]
    average_return = np.mean(returns)
    print(
        f"Average return for next {num_future_steps} days: {average_return:.4f} ({average_return*100:.2f}%)"
    )

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

    plt.text(
        0.02,
        0.98,
        f"Avg Return: {average_return:.4f} ({average_return*100:.2f}%)",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.xlabel("Future Time Steps", fontsize=12)
    plt.ylabel("Predicted Price", fontsize=12)
    plt.title("Future Price Predictions with 95% Confidence Intervals", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = get_data_filepath(symbol)
    plt.savefig(f"{path}/prediction.png", dpi=300, bbox_inches="tight")
