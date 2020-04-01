from pathlib import Path
from typing import Dict, Union

import numpy as np
from tensorflow import keras

from . import pre_processing as pre

def predict(model_dir: Union[str, Path], X: np.ndarray, feature_pre_config: Dict = None, label_pre_config: Dict = None):
    """Does the prediction using the specified model,
    performs any pre & post processing if required"""
    model_dir = Path(model_dir) if isinstance(model_dir, str) else model_dir

    # Load the model
    model = keras.models.load_model(model_dir / "model.h5")

    # Pre-processing of the features
    if feature_pre_config is not None:
        X_mu = np.load(model_dir / "feature_mu.npy") if feature_pre_config.get("standardise") is True else None
        X_sigma = np.load(model_dir / "feature_sigma.npy") if feature_pre_config.get("standardise") is True else None
        W = np.load(model_dir / "feature_W.npy") if feature_pre_config.get("whiten") is True else None
        X = pre.apply(X, X_mu, X_sigma, W)

    y_est = model.predict(X)

    # Post-processing of the labels (i.e. reversing the pre-processing)
    if label_pre_config is not None:
        standardise = label_pre_config.get("standardise")
        if standardise is True:
            y_mu = np.load(model_dir / "label_mu.npy")
            y_sigma = np.load(model_dir / "label_sigma.npy")

            y_est = pre.reverse_standardise(y_est, y_mu, y_sigma)

        shift = label_pre_config.get("shift")
        if shift is not None:
            shift = np.asarray(shift)
            y_est = y_est - shift

    return y_est

