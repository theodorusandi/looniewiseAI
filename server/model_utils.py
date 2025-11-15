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


def evaluate_model(model, testing_data, symbol, scalers):
    X_test, y_test = testing_data
    X_scaler, y_scaler = scalers

    y_pred = model.predict(X_test)

    y_pred_inverse = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_true_inverse = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

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
