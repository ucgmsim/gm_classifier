import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

from . import model
from . import training
from . import plots
from . import pre
from . import utils

import ml_tools


class RecordCompModel:
    label_names = ["score_X", "f_min_X", "score_Y", "f_min_Y", "score_Z", "f_min_Z"]

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
            },
            "dropout": 0.1,
            "lstm_units": [64, 32]
        },
        "dense_final_config": {
            "hidden_layer_func": ml_tools.hidden_layers.elu_mc_dropout,
            "hidden_layer_config": {"dropout": 0.3},
            "units": [32, 16],
        },
    }

    snr_freq_values = np.logspace(np.log(0.01), np.log(25), 100, base=np.e)

    score_values = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    f_min_weights = np.asarray(training.f_min_loss_weights(score_values))

    score_loss_fn = training.create_huber(0.05)
    f_min_loss_fn = training.create_huber(0.5)
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

    def __init__(
        self,
        base_dir: Union[str, Path],
        label_names: List[str] = None,
        feature_config: Dict = None,
        snr_freq_values: np.ndarray = None,
        model_config: Dict = None,
    ):
        self.base_dir = utils.to_path(base_dir)
        self.model_dir = self.base_dir / "best_model" / "model"

        self.label_names = (
            RecordCompModel.label_names if label_names is None else label_names
        )
        self.feature_config = (
            RecordCompModel.feature_config if feature_config is None else feature_config
        )

        feature_config_X = {f"{key}_X": val for key, val in self.feature_config.items()}
        feature_config_Y = {f"{key}_Y": val for key, val in self.feature_config.items()}
        feature_config_Z = {f"{key}_Z": val for key, val in self.feature_config.items()}
        self.feature_config = {
            **feature_config_X,
            **{**feature_config_Y, **feature_config_Z},
        }
        self.feature_names = [
            key for key in self.feature_config.keys() if not "*" in key
        ]

        snr_freq_values = (
            RecordCompModel.snr_freq_values
            if snr_freq_values is None
            else snr_freq_values
        )
        self.snr_feature_keys = [f"snr_value_{freq:.3f}" for freq in snr_freq_values]

        self.model_config = (
            RecordCompModel.model_config if model_config is None else model_config
        )

        # Build the model
        self.gm_model = model.build_dense_cnn_model(
            self.model_config["dense_config"],
            self.model_config["snr_config"],
            self.model_config["dense_final_config"],
            len(self.feature_names),
            len(snr_freq_values),
            6,
            training.create_custom_act_fn(
                tf.keras.activations.linear,
                training.create_soft_clipping(30, z_min=0.1, z_max=10.0),
            ),
        )

        self.compile_kwargs = None
        self.fit_kwargs = None

        # Status flags
        self.is_trained = False
        self.train_df = None
        self.train_history = None

        # These are one set when the model is trained via
        # the train function not fit
        self.X_features_train, self.X_snr_train = None, None
        self.y_train, self.ids_train = None, None
        self.X_features_val, self.X_snr_val = None, None
        self.y_val, self.ids_val = None, None

    def load(self):
        self.gm_model.load_weights(self.model_dir)

    def predict(self, feature_df: pd.DataFrame, n_preds: int = 1):
        X_features = feature_df.loc[:, self.feature_names].copy()

        # Pre-processing
        X_features = pre.apply(
            X_features,
            self.feature_config,
            mu=pd.read_csv(
                self.base_dir / "features_mu.csv", index_col=0, squeeze=True
            ),
            sigma=pd.read_csv(
                self.base_dir / "features_sigma.csv", index_col=0, squeeze=True
            ),
            W=np.load(self.base_dir / "features_W.npy"),
        )

        X_snr = np.log(
            np.stack(
                (
                    feature_df.loc[:, np.char.add(self.snr_feature_keys, "_X")].values,
                    feature_df.loc[:, np.char.add(self.snr_feature_keys, "_Y")].values,
                    feature_df.loc[:, np.char.add(self.snr_feature_keys, "_Z")].values,
                ),
                axis=2,
            )
        )

        if n_preds == 1:
            y_hat = self.gm_model.predict(
                {"features": X_features.values, "snr_series": X_snr}
            )
            return y_hat
        else:
            y_hats = []
            for ix in range(n_preds):
                y_hats.append(
                    self.gm_model.predict(
                        {"features": X_features.values, "snr_series": X_snr}
                    )
                )
            y_hats = np.stack(y_hats, axis=2)
            y_hat_mean, y_hat_std = np.mean(y_hats, axis=2), np.std(y_hats, axis=2)
            return y_hat_mean, y_hat_std

    def train(
        self,
        train_df: pd.DataFrame,
        val_size: float = 0.0,
        compile_kwargs: Dict = {},
        fit_kwargs: Dict = None,
        tensorboard_kwargs: Dict = None,
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
        tensorboard_kwargs: dictionary, optional
            Tensorboard callback keyword arguments
        """
        self.train_df = train_df
        train_df.to_csv(self.base_dir / "training_data.csv")

        assert np.all(
            np.isin(self.feature_names, train_df.columns.values)
        ), "Not all features are in the feature dataframe"

        # Data preparation
        X_features = train_df.loc[:, self.feature_names]

        # Bit of hack atm, need to update
        X_snr = np.stack(
            (
                train_df.loc[:, np.char.add(self.snr_feature_keys, "_X")].values,
                train_df.loc[:, np.char.add(self.snr_feature_keys, "_Y")].values,
                train_df.loc[:, np.char.add(self.snr_feature_keys, "_Z")].values,
            ),
            axis=2,
        )

        ids = train_df.index.values
        y = train_df.loc[:, self.label_names]

        # Split into training and validation
        if val_size > 0.0:
            data = train_test_split(X_features, X_snr, y, ids, test_size=val_size)
            (
                self.X_features_train,
                self.X_snr_train,
                self.y_train,
                self.ids_train,
            ) = data[::2]
            self.X_features_val, self.X_snr_val, self.y_val, self.ids_val = data[1::2]
        else:
            self.X_features_train, self.X_snr_train = X_features, X_snr
            self.y_train, self.ids_train = y, ids

        # Pre-processing
        self.X_features_train, self.X_features_val = training.apply_pre(
            self.X_features_train.copy(),
            self.feature_config,
            self.base_dir,
            val_data=self.X_features_val.copy()
            if self.X_features_val is not None
            else None,
            output_prefix="features",
        )
        self.X_snr_train = np.log(self.X_snr_train)
        self.X_snr_val = np.log(self.X_snr_val) if self.X_snr_val is not None else None

        val_data = (
            (
                self.X_features_val.values,
                self.X_snr_val,
                self.y_val.values,
                self.ids_val,
            )
            if val_size > 0.0
            else None
        )

        print(
            f"Using {self.X_features_train.shape[0]} sample for training and "
            f"{self.X_features_val.shape[0]} for validation"
        )
        self.fit(
            self.X_features_train.values,
            self.X_snr_train,
            self.y_train.values,
            self.ids_train,
            val_data=val_data,
            compile_kwargs=compile_kwargs,
            fit_kwargs=fit_kwargs,
            tensorboard_kwargs=tensorboard_kwargs,
        )

        self.is_trained = True

    def fit(
        self,
        X_features_train: np.ndarray,
        X_snr_train: np.ndarray,
        y_train: np.ndarray,
        ids_train: np.ndarray,
        val_data: Union[
            None, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = None,
        compile_kwargs: Dict = {},
        fit_kwargs: Dict = None,
        tensorboard_kwargs: Dict = None,
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
        tensorboard_kwargs: dictionary, optional
            Tensorboard callback keyword arguments
        """
        self.compile_kwargs = {**RecordCompModel.compile_kwargs, **compile_kwargs}
        self.fit_kwargs = (
            RecordCompModel.fit_kwargs if fit_kwargs is None else fit_kwargs
        )
        self.tensorboard_kwargs = (
            tensorboard_kwargs if tensorboard_kwargs is not None else {}
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

        self.train_history, _ = training.fit(
            self.base_dir,
            self.gm_model,
            training_data,
            val_data=val_data,
            model_config=self.model_config,
            compile_kwargs=self.compile_kwargs,
            fit_kwargs=self.fit_kwargs,
            tensorboard_cb_kwargs=self.tensorboard_kwargs,
        )

        # Save the config
        config = {
            "feature_config": str(self.feature_config),
            "model_config": str(self.model_config),
            "compiler_kwargs": str(self.compile_kwargs),
            "fit_kwargs": str(fit_kwargs),
        }
        with open(self.base_dir / "config.json", "w") as f:
            json.dump(config, f)

        # Load the best model
        self.gm_model.load_weights(str(self.base_dir / "best_model" / "model"))

    def create_eval_plots(self):
        if not self.is_trained:
            print(
                f"Model is either loaded or not trained, "
                f"unable to create eval plots in this situation."
            )
            return

        with sns.axes_style("whitegrid"):
            fig_size = (16, 10)

            fig, ax = plots.plot_loss(
                self.train_history,
                # output_ffp=str(output_dir / "loss_plot.png"),
                fig_kwargs={"figsize": fig_size},
            )
            ax.set_ylim((0, 1))
            plt.savefig(str(self.base_dir / "loss_plot.png"))
            plt.close()

            # Predict train and validation
            y_train_est = self.gm_model.predict(
                {
                    "features": self.X_features_train.values,
                    "snr_series": self.X_snr_train,
                }
            )
            y_val_est = self.gm_model.predict(
                {"features": self.X_features_val.values, "snr_series": self.X_snr_val}
            )

            est_df = pd.DataFrame(
                np.concatenate((y_train_est, y_val_est), axis=0),
                index=np.concatenate((self.ids_train, self.ids_val)),
                columns=self.label_names,
            )
            est_df.to_csv(self.base_dir / "est_df.csv", index_label="record_id")

            cmap = "coolwarm"
            # cmap = sns.color_palette("coolwarm")
            m_size = 4.0
            for cur_comp, (score_ix, f_min_ix) in zip(
                ["X", "Y", "Z"], [(0, 1), (2, 3), (4, 5)]
            ):
                # for cur_comp, score_ix in zip(["X", "Y", "Z"], [0, 1, 2]):
                # Plot true vs estimated
                score_min, score_max = (
                    np.min(self.train_df[f"score_{cur_comp}"]),
                    np.max(self.train_df[f"score_{cur_comp}"]),
                )
                fig, ax, train_scatter = plots.plot_true_vs_est(
                    y_train_est[:, score_ix],
                    self.y_train.iloc[:, score_ix]
                    + np.random.normal(0, 0.01, self.y_train.iloc[:, score_ix].size),
                    c_train="b",
                    c_val="r",
                    y_val_est=y_val_est[:, score_ix],
                    y_val_true=self.y_val.iloc[:, score_ix]
                    + np.random.normal(0, 0.01, self.y_val.iloc[:, score_ix].size),
                    title="Score",
                    min_max=(score_min, score_max),
                    scatter_kwargs={"s": m_size},
                    fig_kwargs={"figsize": fig_size},
                    output_ffp=self.base_dir / f"score_true_vs_est_{cur_comp}.png",
                )

                f_min_min, f_min_max = (
                    np.min(self.train_df[f"f_min_{cur_comp}"]),
                    np.max(self.train_df[f"f_min_{cur_comp}"]),
                )
                fig, ax, train_scatter = plots.plot_true_vs_est(
                    y_train_est[:, f_min_ix],
                    self.y_train.iloc[:, f_min_ix],
                    # + np.random.normal(0, 0.025, y_train.iloc[:, f_min_ix].size),
                    c_train=self.y_train.iloc[:, score_ix].values,
                    c_val=self.y_val.iloc[:, score_ix].values,
                    y_val_est=y_val_est[:, f_min_ix],
                    y_val_true=self.y_val.iloc[:, f_min_ix],
                    # + np.random.normal(0, 0.025, self.y_val.iloc[:, f_min_ix].size),
                    title="f_min",
                    min_max=(f_min_min, f_min_max),
                    scatter_kwargs={"s": m_size, "cmap": cmap},
                    fig_kwargs={"figsize": fig_size},
                    # output_ffp=output_dir / f"f_min_true_vs_est_{cur_comp}.png",
                )
                cbar = fig.colorbar(train_scatter)
                cbar.set_label("Quality score (True)")
                plt.savefig(self.base_dir / f"f_min_true_vs_est_{cur_comp}.png")
                plt.close()

                fig, ax, train_scatter = plots.plot_true_vs_est(
                    y_train_est[:, f_min_ix],
                    self.y_train.iloc[:, f_min_ix],
                    # + np.random.normal(0, 0.025, y_train.iloc[:, f_min_ix].size),
                    c_train=self.y_train.iloc[:, score_ix].values,
                    c_val=self.y_val.iloc[:, score_ix].values,
                    y_val_est=y_val_est[:, f_min_ix],
                    y_val_true=self.y_val.iloc[:, f_min_ix],
                    # + np.random.normal(0, 0.025, y_val.iloc[:, f_min_ix].size),
                    title="f_min",
                    min_max=(f_min_min, f_min_max),
                    scatter_kwargs={"s": m_size, "cmap": cmap},
                    fig_kwargs={"figsize": fig_size},
                    # output_ffp=output_dir / "f_min_true_vs_est.png"
                )
                ax.set_xlim((0.0, 2.0))
                ax.set_ylim((0.0, 2.0))
                cbar = fig.colorbar(train_scatter)
                cbar.set_label("Quality score (True)")
                plt.savefig(self.base_dir / f"f_min_true_vs_est_zoomed_{cur_comp}.png")
                plt.close()
