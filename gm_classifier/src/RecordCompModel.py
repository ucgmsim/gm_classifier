import json
from pathlib import Path
from typing import Dict, Sequence, Tuple, Union, List

import wandb
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from . import model
from . import training
from . import pre
from . import utils

import ml_tools


class RecordCompModel:
    comp_label_names = [
        "score_X",
        "f_min_X",
        "score_Y",
        "f_min_Y",
        "score_Z",
        "f_min_Z",
    ]
    label_names = ["score", "f_min"]

    feature_config = {
        "signal_pe_ratio_max": ["standard", "whiten"],
        "signal_ratio_max": ["standard", "whiten"],
        "snr_min": ["standard", "whiten"],
        "snr_max": ["standard", "whiten"],
        "snr_average": ["standard", "whiten"],
        "max_tail_ratio": ["standard", "whiten"],
        "average_tail_ratio": ["standard", "whiten"],
        "max_head_ratio": ["standard", "whiten"],
        "snr_average_0.1_0.2": ["standard", "whiten"],
        "snr_average_0.2_0.5": ["standard", "whiten"],
        "snr_average_0.5_1.0": ["standard", "whiten"],
        "snr_average_1.0_2.0": ["standard", "whiten"],
        "snr_average_2.0_5.0": ["standard", "whiten"],
        "snr_average_5.0_10.0": ["standard", "whiten"],
        "fas_ratio_low": ["standard", "whiten"],
        "fas_ratio_high": ["standard", "whiten"],
        "pn_pga_ratio": ["standard", "whiten"],
        "is_vertical": None,

        "spike_detector": ["standard", "whiten"],
        "jerk_detector": ["standard", "whiten"],
        "lowres_detector": ["standard", "whiten"],
        "gainjump_detector": ["standard", "whiten"],
        # "flatline_detector": ["standard", "whiten"],

        "p_numpeaks_detector": ["standard", "whiten"],
        "p_multimax_detector": ["standard", "whiten"],
        "p_multidist_detector": ["standard", "whiten"],

        "s_numpeaks_detector": ["standard", "whiten"],
        "s_multimax_detector": ["standard", "whiten"],
        "s_multidist_detector": ["standard", "whiten"],
    }

    model_config = {
        "dense_config": {
            "hidden_layer_func": ml_tools.hidden_layers.elu_mc_dropout,
            "hidden_layer_config": {"dropout": 0.3},
            "units": [32, 16],
        },
        "snr_config": {
            "filters": [16, 32],
            "kernel_sizes": [11, 5],
            "layer_config": {
                "activation": "elu",
                "kernel_initializer": "glorot_uniform",
                "padding": "same",
            },
            "pool_size": 2,
            "dropout": 0.1,
            "lstm_units": [64, 32],
            # "lstm_units": [],
        },
        "dense_final_config": {
            "hidden_layer_func": ml_tools.hidden_layers.elu_mc_dropout,
            "hidden_layer_config": {"dropout": 0.3},
            "units": [32, 16],
        },
    }

    snr_freq_values = np.logspace(np.log(0.01), np.log(25), 100, base=np.e)

    score_values = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    f_min_weights = np.asarray(training.f_min_loss_weights(score_values)) * 0.25

    # score_loss_fn = training.create_huber(0.05)
    # f_min_loss_fn = training.create_huber(0.5)

    score_loss_fn, f_min_loss_fn = training.squared_error, training.squared_error

    loss = training.CustomScaledLoss(
        score_loss_fn,
        f_min_loss_fn,
        score_values,
        f_min_weights,
        score_loss_fn(
            tf.constant(0.0, dtype=tf.float32), tf.constant(1.0, dtype=tf.float32)
        ),
        f_min_loss_fn(
            tf.constant(0.1, dtype=tf.float32), tf.constant(2, dtype=tf.float32)
        ),
    )

    compile_kwargs = {"optimizer": "Adam", "loss": loss}
    fit_kwargs = {"batch_size": 32, "epochs": 250, "verbose": 2}

    components = ["X", "Y", "Z"]

    def __init__(
        self,
        base_dir: Union[str, Path],
        model: tf.keras.Model,
        feature_config: Dict,
        snr_freq_values: np.ndarray,
        model_config: Dict = None,
        log_wandb: bool = False,
    ):
        self.base_dir = utils.to_path(base_dir)
        self.model = model

        self.log_wandb = log_wandb

        self.feature_config = (
            RecordCompModel.feature_config if feature_config is None else feature_config
        )

        self.feature_names = list(self.feature_config.keys())
        self._snr_freq_values = (
            RecordCompModel.snr_freq_values
            if snr_freq_values is None
            else snr_freq_values
        )
        self.snr_feature_keys = [
            f"snr_value_{freq:.3f}" for freq in self._snr_freq_values
        ]

        self.model_config = model_config

        self.compile_kwargs = None
        self.fit_kwargs = None

        # Status flags
        self.is_trained = False
        self.train_df = None
        self.train_history = {}

    @classmethod
    def from_config(
        cls,
        base_dir: Path,
        model_config: Dict = None,
        feature_config: Dict = None,
        snr_freq_values: np.ndarray = None,
        log_wandb: bool = False,
    ):
        model_config = model_config if model_config is not None else cls.model_config
        feature_config = (
            feature_config if feature_config is not None else cls.feature_config
        )
        snr_freq_values = (
            snr_freq_values if snr_freq_values is not None else cls.snr_freq_values
        )

        # Build the model
        gm_model = model.build_dense_cnn_model(
            model_config["dense_config"],
            model_config["snr_config"],
            model_config["dense_final_config"],
            len(list(feature_config.keys())),
            len(snr_freq_values),
            2,
            training.create_custom_act_fn(
                tf.keras.activations.linear,
                # training.create_soft_clipping(30, z_min=0.0, z_max=1.0, x_min=0.0, x_max=1.0),
                training.create_soft_clipping(30, z_min=0.1, z_max=10.0),
            ),
        )

        return cls(
            base_dir,
            gm_model,
            feature_config,
            snr_freq_values,
            model_config=model_config,
            log_wandb=log_wandb,
        )

    @classmethod
    def load(cls, base_dir: Path):
        model = tf.keras.models.load_model(base_dir / "best_model")

        cls(
            base_dir,
            model,
            ml_tools.utils.load_json(base_dir / "feature_config.json"),
            np.load(str(base_dir / "snr_freq_values.npy")),
        )

    def predict(
        self,
        feature_dfs: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        n_preds: int = 100,
    ):
        result_dict, model_uncertainty = {}, {}
        for cur_comp, cur_feature_df in zip(self.components, feature_dfs):
            cur_snr = np.log(cur_feature_df.loc[:, self.snr_feature_keys].values.copy())
            cur_feature_df = cur_feature_df.loc[:, self.feature_names].copy()

            # Pre-processing
            cur_feature_df = pre.apply(
                cur_feature_df,
                self.feature_config,
                mu=pd.read_csv(
                    self.base_dir / f"features_mu.csv", index_col=0, squeeze=True,
                ),
                sigma=pd.read_csv(
                    self.base_dir / f"features_sigma.csv", index_col=0, squeeze=True,
                ),
                W=np.load(self.base_dir / f"features_W.npy"),
            )

            if n_preds == 1:
                y_hat = self.model.predict(
                    {"features": cur_feature_df.values, "snr_series": cur_snr}
                )
                result_dict[f"score_{cur_comp}"] = y_hat[:, 0]
                result_dict[f"f_min_{cur_comp}"] = y_hat[:, 1]
            else:
                y_hats = []
                print("Running predictions")
                for ix in range(n_preds):
                    print(f"Prediction sample {ix + 1}/{n_preds}")
                    y_hats.append(
                        self.model.predict(
                            {"features": cur_feature_df.values, "snr_series": cur_snr}
                        )
                    )
                y_hats = np.stack(y_hats, axis=2)
                y_hat_mean, y_hat_std = np.mean(y_hats, axis=2), np.std(y_hats, axis=2)

                result_dict[f"score_{cur_comp}"] = y_hat_mean[:, 0]
                result_dict[f"f_min_{cur_comp}"] = y_hat_mean[:, 1]

                model_uncertainty[f"score_{cur_comp}"] = y_hat_std[:, 0]
                model_uncertainty[f"f_min_{cur_comp}"] = y_hat_std[:, 1]

        if n_preds == 1:
            return pd.DataFrame.from_dict(result_dict).set_index(cur_feature_df.index)
        return (
            pd.DataFrame.from_dict(result_dict).set_index(cur_feature_df.index),
            pd.DataFrame.from_dict(model_uncertainty).set_index(cur_feature_df.index),
        )

    def train(
        self,
        feature_dfs: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        label_dfs: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        val_size: float = 0.0,
        compile_kwargs: Dict = {},
        fit_kwargs: Dict = None,
    ):
        """Pre-processes the labelled data and trains the model

        Parameters
        ----------
        train_df: dataframe
            Dataframe that contains the labelled data for training
        val_size: float, optional
            Proportion of the labelled data to use for validation
            during training
        compile_kwargs: dictionary, optional
            Keyword arguments for the model compile,
            see https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile
        fit_kwargs: dictionary
            Keyword arguments for the model fit functions,
            see https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        """
        # Get the labelled record ids & split into training and validation
        ids = (
            label_dfs[0]
            .loc[np.isin(label_dfs[0].index, feature_dfs[0].index)]
            .index.values.astype(str)
        )
        self.train_ids, self.val_ids = ids, None
        if val_size > 0.0:
            self.train_ids, self.val_ids = train_test_split(ids, test_size=val_size)

        X_train = pd.concat(
            [cur_df.loc[self.train_ids, self.feature_names] for cur_df in feature_dfs],
            axis=0,
        )
        X_snr_train = pd.concat(
            [
                cur_df.loc[self.train_ids, self.snr_feature_keys]
                for cur_df in feature_dfs
            ],
            axis=0,
        )
        y_train = pd.concat(
            [cur_df.loc[self.train_ids, self.label_names] for cur_df in label_dfs],
            axis=0,
        )

        X_val = pd.concat(
            [cur_df.loc[self.val_ids, self.feature_names] for cur_df in feature_dfs],
            axis=0,
        )
        X_snr_val = pd.concat(
            [cur_df.loc[self.val_ids, self.snr_feature_keys] for cur_df in feature_dfs],
            axis=0,
        )
        y_val = pd.concat(
            [cur_df.loc[self.val_ids, self.label_names] for cur_df in label_dfs], axis=0
        )

        # Sanity checks
        assert np.all(
            np.isin(self.feature_names, X_train.columns.values)
        ), "Not all features are in the feature dataframe"

        # Get the relevant features and labels
        # Save
        for cur_comp, cur_feature_df, cur_label_df in zip(
            ["X", "Y", "Z"], feature_dfs, label_dfs
        ):
            cur_feature_df.loc[ids, self.feature_names].to_csv(
                self.base_dir / f"features_{cur_comp}.csv"
            )
            cur_feature_df.loc[ids, self.snr_feature_keys].to_csv(
                self.base_dir / f"snr_series_{cur_comp}.csv"
            )
            cur_label_df.loc[ids, self.label_names].to_csv(
                self.base_dir / f"labels_{cur_comp}.csv"
            )

        # Pre-processing
        X_train, X_val = training.apply_pre(
            X_train.loc[self.train_ids].copy(),
            self.feature_config,
            self.base_dir,
            val_data=X_val.loc[self.val_ids].copy()
            if self.val_ids is not None
            else None,
            output_prefix=f"features",
        )
        X_snr_train = np.log(X_snr_train.loc[self.train_ids])

        val_data = None
        if self.val_ids is not None:
            cur_snr_val = np.log(X_snr_val.loc[self.val_ids])
            val_data = (
                X_val.values,
                cur_snr_val.values[:, :, None],
                y_val.loc[self.val_ids].values,
                self.val_ids,
            )

        print(
            f"Using {X_train.shape[0]} sample for training and "
            f"{X_val.shape[0]} for validation"
        )
        self.train_history = self.fit(
            self.model,
            X_train.values,
            X_snr_train.values[:, :, None],
            y_train.loc[self.train_ids].values,
            self.train_ids,
            self.base_dir,
            val_data=val_data,
            compile_kwargs=compile_kwargs,
            fit_kwargs=fit_kwargs,
        )

        # Reload the best model (only needed when not using the EarlyStopping callback)
        # self.model = tf.keras.models.load_model(
        #     self.base_dir / "best_model", compile=False
        # )

        ml_tools.utils.write_to_json(
            self.model_config, self.base_dir / "model_config.json"
        )
        ml_tools.utils.write_to_json(
            self.feature_config, self.base_dir / "feature_config.json"
        )
        np.save(self.base_dir / "snr_freq_values.npy", self._snr_freq_values)
        self.is_trained = True

    def fit(
        self,
        gmc_model: tf.keras.Model,
        X_features_train: np.ndarray,
        X_snr_train: np.ndarray,
        y_train: np.ndarray,
        ids_train: np.ndarray,
        output_dir: Path,
        val_data: Union[
            None, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = None,
        compile_kwargs: Dict = {},
        fit_kwargs: Dict = None,
    ):
        """
        Fits the model, does not do any pre-processing,
        merely compiles the model and fits it.

        Parameters
        ----------
        X_features_train: numpy array
            The already pre-processed scalar features for the FC-NN
            Shape: [n_samples, n_features * 3]
        X_snr_train: numpy array
            The already pre-processed SNR series for each component
            Shape: [n_samples, n_frequencies, 3]
        y_train: numpy array
            The true labels, requires column order [score_X, f_min_X, score_Y, f_min_Y, score_Z, f_min_Z]
            Shape: [n_samples, 6]
        ids_train: numpy array
            The ids used for training
            Shape: [n_samples]
        val_data: tuple, optional
            Validation data, expected tuple data:
            (X_features_val, X_snr_val, y_train, ids_train)
        compile_kwargs: dictionary, optional
            Keyword arguments for the model compile,
            see https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile
        fit_kwargs: dictionary
            Keyword arguments for the model fit functions,
            see https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        """
        self.compile_kwargs = {**RecordCompModel.compile_kwargs, **compile_kwargs}
        self.fit_kwargs = (
            RecordCompModel.fit_kwargs if fit_kwargs is None else fit_kwargs
        )

        training_data = (
            {"features": X_features_train, "snr_series": X_snr_train},
            y_train,
            ids_train,
        )
        val_data = (
            (
                {"features": val_data[0], "snr_series": val_data[1]},
                val_data[2],
                val_data[3],
            )
            if val_data is not None
            else None
        )

        history, _ = training.fit(
            output_dir,
            gmc_model,
            training_data,
            val_data=val_data,
            compile_kwargs=self.compile_kwargs,
            fit_kwargs=self.fit_kwargs,
            callbacks=[
                model.MetricRecorder(
                    training_data,
                    val_data,
                    self.compile_kwargs["loss"],
                    log_wandb=self.log_wandb,
                ),
                keras.callbacks.EarlyStopping(
                    min_delta=0.005, patience=100, verbose=1, restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    factor=0.5, patience=50, verbose=1, min_lr=1e-6, min_delta=5e-3
                ),
            ],
        )

        best_train_loss = [
            self.compile_kwargs["loss"](
                y_train.astype(np.float32), gmc_model.predict(training_data[0])
            ).numpy()
            for ix in range(100)
        ]
        best_train_loss, best_train_loss_std = (
            np.mean(best_train_loss),
            np.std(best_train_loss),
        )
        best_val_loss = [
            self.compile_kwargs["loss"](
                val_data[1].astype(np.float32), gmc_model.predict(val_data[0])
            ).numpy()
            for ix in range(100)
        ]
        best_val_loss, best_val_loss_std = np.mean(best_val_loss), np.std(best_val_loss)
        print(
            f"Final model - Training loss: {best_train_loss:.4f} +/- {best_train_loss_std:.4f}, "
            f"Validation loss: {best_val_loss:.4f} +/- {best_val_loss_std:.4f}"
        )

        if self.log_wandb is not None:
            wandb.run.summary[f"best_train_loss"] = best_train_loss
            wandb.run.summary[f"best_val_loss"] = best_val_loss

        # Save the config
        config = {
            "feature_config": str(self.feature_config),
            "model_config": str(self.model_config),
            "compiler_kwargs": str(self.compile_kwargs),
            "fit_kwargs": str(fit_kwargs),
        }
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f)

        return history
