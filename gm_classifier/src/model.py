from typing import List, Dict, Tuple, Union, Callable


import wandb
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

import ml_tools


class MetricRecorder(keras.callbacks.Callback):
    def __init__(
        self, training_data: Tuple, val_data: Tuple, loss_fn: tf.function, log_wandb: bool = False
    ):
        super().__init__()

        self.X_train, self.y_train, _ = training_data
        self.X_val, self.y_val, _ = val_data

        # self.X_train_rep = {cur_key: np.tile(cur_item, (10, 1)) for cur_key, cur_item in self.X_train.items()}
        # self.X_val_rep = {cur_key: np.tile(cur_item, (10, -1)) for cur_key, cur_item in self.X_val.items()}

        self.loss_fn = loss_fn

        self.log_wandb = log_wandb

    def on_epoch_end(self, epoch, logs=None):
        y_train_est = self.model.predict(self.X_train)
        y_val_est = self.model.predict(self.X_val)

        score_train_mse = tf.losses.mse(self.y_train[:, 0], y_train_est[:, 0]).numpy()
        f_min_train_mse = tf.losses.mse(self.y_train[:, 1], y_train_est[:, 1]).numpy()

        score_val_mse = tf.losses.mse(self.y_val[:, 0], y_val_est[:, 0]).numpy()
        f_min_val_mse = tf.losses.mse(self.y_val[:, 1], y_val_est[:, 1]).numpy()

        logs["score_train_mse"], logs["f_min_train_mse"] = (
            score_train_mse,
            f_min_train_mse,
        )
        logs["score_val_mse"], logs["f_min_val_mse"] = score_val_mse, f_min_val_mse

        if epoch % 25 == 0:
            mc_loss = [self.loss_fn(self.y_train, self.model.predict(self.X_train)).numpy() for ix in range(10)]
            mc_val_loss = [self.loss_fn(self.y_val, self.model.predict(self.X_val)).numpy() for ix in range(10)]

            mc_loss, mc_loss_std = np.mean(mc_loss), np.std(mc_loss)
            mc_val_loss, mc_val_loss_std = np.mean(mc_val_loss), np.std(mc_val_loss)

            print(f"MC Loss: {mc_loss:.4f} +/- {mc_loss_std:.4f}, "
                  f"MC Val Loss: {mc_val_loss:.4f} +/- {mc_val_loss_std:.4f}")


        if self.log_wandb:
            wandb.log(
                {
                    f"score_train_mse": score_train_mse,
                    f"f_min_train_mse": f_min_train_mse,
                    f"score_val_mse": score_val_mse,
                    f"f_min_val_mse": f_min_val_mse,
                    f"loss": logs["loss"],
                    f"val_loss": logs["val_loss"],
                    "epoch": epoch,
                }
            )


        print(
            f"Score - Train: {score_train_mse:.4f}, Val: {score_val_mse:.4f} -- "
            f"F_min - Train: {f_min_train_mse:.4f}, Val: {f_min_val_mse:.4f}"
        )

        return


def build_dense_cnn_model(
    dense_config: Dict,
    snr_config: Dict,
    dense_final_config: Dict,
    n_features: int,
    n_snr_steps: int,
    n_outputs: int,
    out_act_func: tf.function,
):
    hidden_layer_func = dense_config["hidden_layer_func"]
    hidden_layer_config = dense_config["hidden_layer_config"]

    # Build the scalar features dense layers
    scalar_inputs = keras.Input(n_features, name="features")
    x_nn = hidden_layer_func(
        scalar_inputs, dense_config["units"][0], **hidden_layer_config
    )
    for n_units in dense_config["units"][1:]:
        x_nn = hidden_layer_func(x_nn, n_units, **hidden_layer_config)

    # Build the CNN + LSTM layers
    cnn_filters, cnn_kernels = snr_config["filters"], snr_config["kernel_sizes"]
    dropout, cnn_layer_config = snr_config["dropout"], snr_config["layer_config"]
    pool_size, lstm_units = snr_config["pool_size"], snr_config["lstm_units"]

    snr_input = keras.Input((n_snr_steps, 1), name="snr_series")
    x_snr = keras.layers.Conv1D(
        filters=cnn_filters[0], kernel_size=cnn_kernels[0], **cnn_layer_config
    )(snr_input)
    for n_filters, n_kernels in zip(cnn_filters[1:], cnn_kernels[1:]):
        keras.layers.Conv1D(
            filters=n_filters, kernel_size=n_kernels, **cnn_layer_config
        )(x_snr)
        if dropout is not None:
            x_snr = ml_tools.hidden_layers.MCSpatialDropout1D(rate=dropout)(x_snr)
        if pool_size is not None:
            x_snr = keras.layers.MaxPooling1D(pool_size)(x_snr)

    if len(lstm_units) > 0:
        for ix, n_units in enumerate(lstm_units):
            last_lstm = True if ix + 1 == len(lstm_units) else False
            x_snr = keras.layers.Bidirectional(
                layer=keras.layers.LSTM(
                    units=n_units,
                    return_sequences=True if not last_lstm else False,
                    return_state=True if last_lstm else False,
                )
            )(x_snr)
            if last_lstm:
                x_snr = keras.layers.Concatenate()(x_snr[1:])
    else:
        x_snr = keras.layers.Flatten()(x_snr)

    # Build the output layers
    x = keras.layers.Concatenate()([x_nn, x_snr])

    for n_units in dense_final_config["units"]:
        x = dense_final_config["hidden_layer_func"](
            x, n_units, **dense_final_config["hidden_layer_config"]
        )

    outputs = keras.layers.Dense(n_outputs, activation=out_act_func, name="output")(x)
    return keras.Model(inputs=[scalar_inputs, snr_input], outputs=outputs)
