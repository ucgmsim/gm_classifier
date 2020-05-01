import json
from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from sklearn.model_selection import KFold
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import minmax_scale, robust_scale
from sklearn.cluster import KMeans
from scipy import stats
from tensorflow import keras

import gm_classifier as gm

# ----- Helper functions ------
def __sample_record_ids(n: int, df: pd.DataFrame, mask: pd.Series):
    if np.count_nonzero(mask) < n:
        n = np.count_nonzero(mask)
    return list(df.loc[mask].sample(n).index.values)

def loss_selection_fn(df_full: pd.DataFrame, train_df: pd.DataFrame, cluster_label: int, loss_name: str):
    train_cl_mask = train_df.cluster == cluster_label
    full_cl_mask = df_full.cluster == cluster_label

    # Only interested in records/samples that haven't already been labelled
    full_cl_unlabelled_mask = full_cl_mask & ~np.isin(df_full.index.values, train_df.index.values)

    loss_median = train_df[loss_name].median()
    loss_iqr = stats.iqr(train_df[loss_name], rng=(10, 90))

    cur_n_new_ids = None
    if np.count_nonzero(train_cl_mask) == 0:
        cur_n_new_ids = 2
    else:
        # Use mean here since we don't want "ignore"
        # loss outliers in the current cluster
        cur_mean_loss = train_df.loc[train_cl_mask, loss_name].mean()
        if cur_mean_loss < (loss_median + 0.5 * loss_iqr):
            cur_n_new_ids = 1
        elif cur_mean_loss < (loss_median + 1.0 * loss_iqr):
            cur_n_new_ids = 2
        elif cur_mean_loss < (loss_median + 2.0 * loss_iqr):
            cur_n_new_ids = 3
        else:
            cur_n_new_ids = 4

    cur_new_ids = __sample_record_ids(cur_n_new_ids, df_full, full_cl_unlabelled_mask)
    return cur_new_ids, len(cur_new_ids)

# ---- Config ----
# label_dir = "/Users/Clus/code/work/gm_classifier/data/records/training_data/iter"
label_dir = (
    "/Users/Clus/code/work/gm_classifier/data/records/local/training_data/labels"
)
features_dir = "/Users/Clus/code/work/gm_classifier/data/records/local/training_data/all_records_features/200401"

output_dir = "/Users/Clus/code/work/tmp/gm_classifier/active_learning"

label_config = {
    "score_X": None,
    "f_min_X": None,
    "score_Y": None,
    "f_min_Y": None,
    "score_Z": None,
    "f_min_Z": None,
}
label_names = list(label_config.keys())

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
    # "is_vertical": None,
}
feature_config_X = {f"{key}_X": val for key, val in feature_config.items()}
feature_config_Y = {f"{key}_Y": val for key, val in feature_config.items()}
feature_config_Z = {f"{key}_Z": val for key, val in feature_config.items()}
feature_config = {**feature_config_X, **{**feature_config_Y, **feature_config_Z}}
feature_names = [key for key in feature_config.keys() if not "*" in key]

snr_features = [
    f"snr_value_{freq:.3f}"
    for freq in np.logspace(np.log(0.05), np.log(20), 50, base=np.e)
]

# Model config
dropout_rate = 0.5
model_config = {
    "dense_layer_config": [
        (keras.layers.Dense, {"units": 32, "activation": "relu"}),
        (keras.layers.Dense, {"units": 16, "activation": "relu"}),
    ],
    "dense_input_name": "features",
    "cnn_layer_config": [
        (
            keras.layers.Conv1D,
            {
                "filters": 32,
                "kernel_size": 15,
                "strides": 1,
                "activation": "relu",
                "padding": "same",
            },
        ),
        (keras.layers.MaxPooling1D, {"pool_size": 2}),
        (keras.layers.Dropout, {"rate": dropout_rate}),
        (
            keras.layers.Conv1D,
            {
                "filters": 16,
                "kernel_size": 5,
                "strides": 1,
                "activation": "relu",
                "padding": "same",
            },
        ),
        # (keras.layers.MaxPooling1D, {"pool_size": 2}),
        (keras.layers.Dropout, {"rate": dropout_rate}),
    ],
    "cnn_input_name": "snr_series",
    "comb_layer_config": [
        (keras.layers.Dense, {"units": 128, "activation": "relu"}),
        (keras.layers.Dropout, {"rate": dropout_rate}),
        (keras.layers.Dense, {"units": 42, "activation": "relu"}),
        (keras.layers.Dropout, {"rate": dropout_rate}),
        (keras.layers.Dense, {"units": 16, "activation": "relu"}),
    ],
    "n_outputs": 6,
}

scores = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
weights = np.asarray(gm.training.f_min_loss_weights(scores))

# Training details
optimizer = "Adam"
loss = gm.training.WeightedFMinMSELoss(scores, weights)
n_epochs = 250
batch_size = 32
n_folds = 20

# Data-point selection config
eval_loss = gm.training.WeightedFMinMSELoss(scores, weights, reduction="none")
eval_loss_name = "f_min_loss"

output_dir = Path(output_dir)

# ---- Run ----

label_df = gm.utils.load_labels_from_dir(label_dir, f_min_100_value=10)
feature_df = gm.utils.load_features_from_dir(features_dir)

train_df = pd.merge(feature_df, label_df, left_index=True, right_index=True)
train_df.to_csv(output_dir / "training_data.csv")

# Some sanity checks
assert np.all(
    np.isin(feature_names, feature_df.columns.values)
), "Not all features are in the feature dataframe"


# Data preperation
ids = train_df.index.values.astype(str)
X_features = train_df.loc[:, feature_names]

# Bit of hack atm, need to update
X_snr = np.stack(
    (
        train_df.loc[:, np.char.add(snr_features, "_X")].values,
        train_df.loc[:, np.char.add(snr_features, "_Y")].values,
        train_df.loc[:, np.char.add(snr_features, "_Z")].values,
    ),
    axis=2,
)

y = train_df.loc[:, label_names]

# Use K-fold to get estimated values for every labelled sample
kf = KFold(n_splits=n_folds, shuffle=True)
r_ids, r_y_val, r_y_val_est, r_val_loss = [], [], [], []
for ix, (train_ind, val_ind) in enumerate(kf.split(X_features)):
    cur_id = f"iter_{ix}"

    # Create the output directory
    cur_output_dir = output_dir / cur_id
    cur_output_dir.mkdir()

    # Get the current training and validation data
    X_features_train, X_snr_train, y_train, ids_train = (
        X_features.iloc[train_ind].copy(),
        X_snr[train_ind],
        y.iloc[train_ind].copy(),
        ids[train_ind],
    )
    X_features_val, X_snr_val, y_val, ids_val = (
        X_features.iloc[val_ind].copy(),
        X_snr[val_ind],
        y.iloc[val_ind].copy(),
        ids[val_ind],
    )

    # Pre-processing
    X_features_train, X_features_val = gm.training.apply_pre(
        X_features_train.copy(),
        feature_config,
        output_dir,
        val_data=X_features_val.copy(),
        output_prefix="features",
    )
    X_snr_train, X_snr_val = np.log(X_snr_train), np.log(X_snr_val)

    # Run training of the model
    compile_kwargs = {"optimizer": optimizer, "loss": loss}
    fit_kwargs = {"batch_size": batch_size, "epochs": n_epochs, "verbose": 2}
    history, gm_model = gm.training.fit(
        cur_output_dir,
        gm.model.CnnSnrModel,
        model_config,
        (
            {"features": X_features_train.values, "snr_series": X_snr_train},
            y_train.values,
            ids_train,
        ),
        val_data=(
            {"features": X_features_val.values, "snr_series": X_snr_val},
            y_val.values,
            ids_val,
        ),
        compile_kwargs=compile_kwargs,
        fit_kwargs=fit_kwargs,
    )

    # Loss plot
    fig, ax = gm.plots.plot_loss(
        history,
        # output_ffp=str(output_dir / "loss_plot.png"),
        fig_kwargs={"figsize": (16, 10)},
    )
    ax.set_ylim((0, 1))
    plt.savefig(str(cur_output_dir / "loss_plot.png"))
    plt.close()

    # Load the best model
    gm_model.load_weights(str(cur_output_dir / "model.h5"))
    cur_y_est = gm_model.predict(
        {"features": X_features_val.values, "snr_series": X_snr_val}
    )

    cur_val_loss = eval_loss(y_val.values, cur_y_est).numpy()

    r_ids.append(ids_val)
    r_y_val.append(y_val)
    r_y_val_est.append(cur_y_est)
    r_val_loss.append(cur_val_loss)

r_ids = np.concatenate(r_ids)
r_y_val = np.concatenate(r_y_val)
r_y_val_est = np.concatenate(r_y_val_est)
r_val_loss = np.concatenate(r_val_loss)

# Save the config
config = {
    "feature_config": str(feature_config),
    "model_config": str(model_config),
    "compiler_kwargs": str(compile_kwargs),
    "fit_kwargs": str(fit_kwargs),
}
with open(output_dir / "config.json", "w") as f:
    json.dump(config, f)


# --- Combine the results ----
df = pd.DataFrame(
    data=np.concatenate((r_y_val_est, r_val_loss), axis=1),
    index=r_ids,
    columns=np.concatenate(
        (np.char.add(label_names, "_est"), np.char.add(label_names, "_loss")), axis=0
    ),
)
train_df = train_df.merge(df, right_index=True, left_index=True)
del df

# Compute the residuals
for cur_comp in ["X", "Y", "Z"]:
    # Residual
    train_df[f"res_score_{cur_comp}"] = (
        train_df[f"score_{cur_comp}"] - train_df[f"score_{cur_comp}_est"]
    )
    train_df[f"res_f_min_{cur_comp}"] = (
        train_df[f"f_min_{cur_comp}"] - train_df[f"f_min_{cur_comp}_est"]
    )

    # Log residual
    train_df[f"rel_res_f_min_{cur_comp}"] = np.abs(
        train_df[f"f_min_{cur_comp}_est"] / train_df[f"f_min_{cur_comp}"]
    )

# Compute mean losses
train_df["score_loss"] = train_df.loc[
    :, ["score_X_loss", "score_Y_loss", "score_Z_loss"]
].sum(axis=1)
train_df["f_min_loss"] = train_df.loc[
    :, ["f_min_X_loss", "f_min_Y_loss", "f_min_Z_loss"]
].sum(axis=1)
train_df["loss"] = train_df.loc[
    :,
    [
        "score_X_loss",
        "f_min_X_loss",
        "score_Y_loss",
        "f_min_Y_loss",
        "score_Z_loss",
        "f_min_Z_loss",
    ],
].sum(axis=1)

df_full = feature_df.loc[:, feature_names].copy()
X_full = df_full.values

# --- Dim reduction + Clustering + nearest neighbour selection ----

# Scale the features
X_full = gm.pre.standardise(X_full, X_full.mean(axis=0), X_full.std(axis=0))

# Whiten the features
# W = gm.pre.compute_W_ZCA(X)
# X = gm.pre.whiten(X, W)

enc, X_trans, rec_loss = gm.dim_reduction.ae(X_full)
enc.save(output_dir / "encoder.h5")

# Do I need to scale here again???
# X_trans = robust_scale(X_trans)
df_full["X_1"] = X_trans[:, 0]
df_full["X_2"] = X_trans[:, 1]

train_df["X_1"] = df_full.loc[train_df.index.values, "X_1"]
train_df["X_2"] = df_full.loc[train_df.index.values, "X_2"]

# Run k-means
print(f"Running k-means clustering")
k_means = KMeans(n_clusters=100, max_iter=300, n_jobs=1)
k_means_labels = k_means.fit_predict(X_trans)
df_full["cluster"] = k_means_labels

# Identify records for each cluster (select more from clusters with bad predicted scores)
new_record_ids, n_new_ids = [], {}
train_df["cluster"] = k_means.predict(
    np.concatenate((train_df.X_1.values[:, None], train_df.X_2.values[:, None]), axis=1)
)
for cur_label in np.unique(k_means_labels):
    cur_new_record_ids, cur_n_new_record_ids = loss_selection_fn(
        df_full, train_df, cur_label, eval_loss_name
    )
    new_record_ids.extend(cur_new_record_ids)
    n_new_ids[cur_label] = cur_n_new_record_ids

# For the worst ten also grab the nearest 2 neighbours
worst_ids = train_df[eval_loss_name].sort_values(ascending=False).index.values[:10]
tree = KDTree(df_full.loc[:, ["X_1", "X_2"]].values)

# First column are the query points itself (with distance 0)
ind = tree.query(
    train_df.loc[worst_ids, ["X_1", "X_2"]].values, k=3, return_distance=False
)[:, 1:]
new_record_ids.extend(list(df_full.iloc[ind.ravel()].index.values))

# Save relevant dataframes
df_full.to_csv(output_dir / "all_data.csv", index_label="record_id")
train_df.to_csv(output_dir / "labelled_data.csv", index_label="record_id")

new_record_ids = np.unique(new_record_ids)
new_record_ids = new_record_ids[~np.isin(new_record_ids, train_df.index.values)]
print(f"Identified {new_record_ids.size} new record ids to label")

# Save new record_ids
with open(output_dir / "new_record_ids.txt", "w") as f:
    f.writelines([f"{id}\n" for id in new_record_ids])

plt.figure(figsize=(16, 10))
c_m_combs = gm.plots.get_color_marker_comb()
for cur_label, (cur_c, cur_m) in zip(np.unique(k_means_labels), c_m_combs):
    cur_mask = k_means_labels == cur_label
    plt.scatter(
        X_trans[cur_mask, 0], X_trans[cur_mask, 1], c=cur_c, marker=cur_m, s=1.0
    )

plt.scatter(train_df.X_1, train_df.X_2, s=5.0, marker="X", c="k")
plt.scatter(
    df_full.loc[new_record_ids].X_1,
    df_full.loc[new_record_ids].X_2,
    s=5.0,
    marker="X",
    c="b",
)

# plt.show()
plt.savefig(output_dir / "clusters.png")

plt.xlim(
    (
        df_full.loc[new_record_ids].X_1.min(),
        stats.iqr(df_full.loc[new_record_ids].X_1, rng=(5, 95)),
    )
)
plt.ylim(
    (
        df_full.loc[new_record_ids].X_2.min(),
        stats.iqr(df_full.loc[new_record_ids].X_2, rng=(5, 95)),
    )
)
plt.savefig(output_dir / "clusters_zoomed.png")
plt.close()

exit()
