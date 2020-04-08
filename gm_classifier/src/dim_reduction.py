from typing import Dict

import numpy as np
import pandas as pd

from sklearn import decomposition
from tensorflow import keras


def pca(X: np.ndarray, n_dims: int = 2, **kwargs):
    """Uses PCA for dimensionality reduction"""
    pca = decomposition.PCA(n_components=n_dims, **kwargs)
    return pca, pca.fit_transform(X)


def ae(X: np.ndarray, n_dims: int = 2, activation: str = "relu", fit_kwargs: Dict = {}):
    """Trains an autoencoder for dimensionality reduction"""
    input = keras.layers.Input(shape=X.shape[1])
    enc = keras.layers.Dense(15, activation=activation)(input)
    enc = keras.layers.Dense(6, activation=activation)(enc)
    z = keras.layers.Dense(n_dims, activation=activation)(enc)
    dec = keras.layers.Dense(6, activation=activation)(z)
    dec = keras.layers.Dense(15, activation=activation)(dec)
    output = keras.layers.Dense(X.shape[1], activation=activation)(dec)

    # Define the models
    ae_model = keras.Model(input, output)

    encoder = keras.Model(input, z)


    ae_model.compile(optimizer="Adam", loss="mse")

    # Merge fit arguments
    fit_kwargs = {
        **{"batch_size": 256, "shuffle": True, "epochs": 100, "verbose": 2},
        **fit_kwargs,
    }
    history = ae_model.fit(X, X, **fit_kwargs)

    X_emb = encoder.predict(X)

    return encoder, X_emb, history.history["loss"][-1]


def tSNE(
    X: np.ndarray,
    n_dims: int = 2,
    p: float = 30,
    learning_rate: float = 200,
    n_iter: int = 1000,
    cuda: bool = False,
    n_jobs: int = 4,
):
    """Uses t-SNE for dimensionality reduction, uses the significantly faster
    tsnecuda library if cuda is set to True"""
    if cuda:
        from tsnecuda import TSNE

        tsne = TSNE(
            perplexity=p,
            learning_rate=learning_rate,
            n_components=n_dims,
            verbose=1,
            n_iter=n_iter,
        )
    else:
        from sklearn.manifold import TSNE

        tsne = TSNE(
            perplexity=p,
            learning_rate=learning_rate,
            n_components=2,
            n_jobs=7,
            verbose=1,
            n_iter=n_iter,
        )

    return tsne, tsne.fit_transform(X)
