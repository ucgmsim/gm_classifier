from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
from tensorflow import keras

from . import pre_processing as pre
from . import model


#
# def predict(gm_model: keras.Model, model_dir: Union[str, Path], X: pd.DataFrame, feature_config: Dict = None):
#     """Does the prediction using the specified model,
#     performs any pre & post processing if required"""
#     model_dir = Path(model_dir) if isinstance(model_dir, str) else model_dir
#
#     # Load the model
#     gm_model.load_weights(str(model_dir / "model.h5"))
#
#     # Pre-processing of the features
#     if feature_config is not None:
#         X_mu = pd.read_csv(model_dir / "feature_mu.csv")
#         X_sigma = pd.read_csv(model_dir / "feature_sigma.csv")
#
#         W = np.load(model_dir / "feature_W.npy") if feature_config.get("whiten") is True else None
#         X = pre.apply(X, feature_config, X_mu, X_sigma, W)
#
#     y_est = gm_model.predict(X)

    # # Post-processing of the labels (i.e. reversing the pre-processing)
    # if label_config is not None:
    #     standardise = label_config.get("standardise")
    #     if standardise is True:
    #         y_mu = pd.read_csv(model_dir / "label_mu.csv")
    #         y_sigma = pd.read_csv(model_dir / "label_sigma.csv")
    #
    #         y_est = pre.reverse_standardise(y_est, y_mu, y_sigma)
    #
    #     shift = label_config.get("shift")
    #     if shift is not None:
    #         shift = np.asarray(shift)
    #         y_est = y_est - shift

    # return y_est

