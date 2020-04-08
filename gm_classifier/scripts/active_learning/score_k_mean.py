from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.neighbors import KDTree
from sklearn.model_selection import KFold
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import minmax_scale, robust_scale
from sklearn.cluster import KMeans
from scipy import stats
import matplotlib.pyplot as plt

import gm_classifier as gm


# ---- Config ----
# label_dir = "/Users/Clus/code/work/gm_classifier/data/records/training_data/iter"
label_dir = (
    "/Users/Clus/code/work/gm_classifier/data/records_local/training_data/labels"
)

# features_dir = "/Users/Clus/code/work/gm_classifier/data/records/training_data/all_records_features/200401"
features_dir = "/Users/Clus/code/work/gm_classifier/data/records_local/training_data/all_records_features/200401"

model_dir = "/Users/Clus/code/work/docs/gm_classifier/results_200404"
output_dir = (
    "/Users/Clus/code/work/tmp/gm_classifier/active_learning"
)

label_config = {
    "score_X": None,
    "score_Y": None,
    "score_Z": None,
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
}
feature_config_X = {f"{key}_X": val for key, val in feature_config.items()}
feature_config_Y = {f"{key}_Y": val for key, val in feature_config.items()}
feature_config_Z = {f"{key}_Z": val for key, val in feature_config.items()}
feature_config = {**feature_config_X, **{**feature_config_Y, **feature_config_Z}}
feature_names = [key for key in feature_config.keys() if not "*" in key]


# Model details
model_config = {
    "units": [128, 64],
    "act_funcs": "relu",
    "n_outputs": 3,
    # "n_outputs": 6,
    "output_act_func": "linear",
    "output_names": label_names,
    "dropout": 0.5,
}

# Training details
optimizer = "Adam"
# loss = gm.training.mape
loss = "mse"
n_epochs = 250
batch_size = 32

# ---- Run ----
output_dir = Path(output_dir)

label_df = gm.utils.load_labels_from_dir(label_dir)
feature_df = gm.utils.load_features_from_dir(features_dir)

train_df = pd.merge(feature_df, label_df, left_index=True, right_index=True)
train_df.to_csv(output_dir / "training_data.csv")

ids = train_df.index.values.astype(str)
X = train_df.loc[:, feature_names]
y = train_df.loc[:, label_names]

# Use K-fold to get estimated values for every labelled sample
kf = KFold(n_splits=10, shuffle=True)
r_ids, r_y_val, r_y_val_est = [], [], []
for ix, (train_ind, val_ind) in enumerate(kf.split(X)):
    cur_id = f"iter_{ix}"

    # Create the output directory
    cur_output_dir = output_dir / cur_id
    cur_output_dir.mkdir()

    # Get the current training and validation data
    X_train, y_train, ids_train = (
        X.iloc[train_ind].copy(),
        y.iloc[train_ind].copy(),
        ids[train_ind],
    )
    X_val, y_val, ids_val = X.iloc[val_ind].copy(), y.iloc[val_ind].copy(), ids[val_ind]

    _, X_train, X_val, y_train, y_val = gm.training.train(
        cur_output_dir,
        feature_config,
        model_config,
        (X_train, y_train, ids_train),
        val_data=(X_val, y_val, ids_val),
        label_config=None,
        compile_kwargs={"optimizer": optimizer, "loss": loss},
        fit_kwargs={"batch_size": batch_size, "epochs": n_epochs, "verbose": 2},
    )

    cur_y_est = gm.predict.predict(cur_output_dir, X_val)

    r_ids.append(ids_val)
    r_y_val.append(y_val)
    r_y_val_est.append(cur_y_est)

r_ids = np.concatenate(r_ids)
r_y_val = np.concatenate(r_y_val)
r_y_val_est = np.concatenate(r_y_val_est)


# --- Clustering + nearest neighbour selection ----
df = pd.DataFrame(
    data=np.concatenate((r_y_val, r_y_val_est), axis=1),
    index=r_ids,
    columns=np.concatenate((label_names, np.char.add(label_names, "_est"))),
)

df["res_X"] = r_y_val[:, 0] - r_y_val_est[:, 0]
df["res_Y"] = r_y_val[:, 1] - r_y_val_est[:, 1]
df["res_Z"] = r_y_val[:, 2] - r_y_val_est[:, 2]
df["res_mean"] = (np.abs(df.res_X) + np.abs(df.res_Y) + np.abs(df.res_Z)) / 3.0
train_df = train_df.merge(df, right_index=True, left_index=True)
del df

res_mean, res_sigma = train_df.res_mean.mean(), train_df.res_mean.std()

df_full = feature_df.loc[:, feature_names].copy()
X_full = df_full.values

# Scale the features
X_full = gm.pre.standardise(X_full, X_full.mean(axis=0), X_full.std(axis=0))

# Whiten the features
# W = gm.pre.compute_W_ZCA(X)
# X = gm.pre.whiten(X, W)

enc, X_trans, rec_loss = gm.dim_reduction.ae(X_full)

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


def __sample_record_ids(n: int, df: pd.DataFrame, mask: pd.Series):
    if np.count_nonzero(mask) < n:
        n = np.count_nonzero(mask)
    return list(df.loc[mask].sample(n).index.values)

# Identify records for each cluster (select more from clusters with bad predicted scores)
new_record_ids, n_new_ids = [], {}
train_df["cluster"] = k_means.predict(np.concatenate((train_df.X_1.values[:, None], train_df.X_2.values[:, None]), axis=1))
for cur_label in np.unique(k_means_labels):
    label_mask = train_df.cluster == cur_label
    full_mask = df_full.cluster == cur_label

    # No labelled data in cluster -- select a random one
    cur_n_new_ids = None
    if np.count_nonzero(label_mask) == 0:
        cur_n_new_ids = 2
    else:
        cur_mean_res = train_df.loc[label_mask].res_mean.mean()
        if cur_mean_res < 0.25:
            cur_n_new_ids = 1
        elif 0.25 < cur_mean_res < 0.5:
            cur_n_new_ids = 2
        elif 0.5 < cur_mean_res < 5.0:
            cur_n_new_ids = 3
        elif 5.0 < cur_mean_res:
            cur_n_new_ids = 4
    new_record_ids.extend(__sample_record_ids(cur_n_new_ids, df_full, full_mask))
    n_new_ids[cur_label] = cur_n_new_ids

# For the worst ten also grab the nearest 2 neighbours
worst_ids = train_df.res_mean.sort_values(ascending=False).index.values[:10]
tree = KDTree(df_full.loc[:, ["X_1", "X_2"]].values)

# First column are the query points itself (with distance 0)
ind = tree.query(train_df.loc[worst_ids, ["X_1", "X_2"]].values, k=3, return_distance=False)[:, 1:]
new_record_ids.extend(list(df_full.iloc[ind.ravel()].index.values))

# Some cluster metadata
cluster_mean_res = train_df.groupby("cluster").res_mean.mean()
cluster_n_members = df_full.groupby("cluster").count().iloc[:, 0]
cluster_n_members.name = "n_members"
cluster_df = cluster_n_members.to_frame().merge(cluster_mean_res, how="left", right_index=True, left_index=True)

# Save relevant dataframes
cluster_df.to_csv(output_dir / "cluster_meta.csv", index_label="cluster_label")
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

plt.scatter(
    train_df.X_1,
    train_df.X_2,
    s=5.0,
    marker="X",
    c="k"
)
plt.scatter(
    df_full.loc[new_record_ids].X_1,
    df_full.loc[new_record_ids].X_2,
    s=5.0,
    marker="X",
    c="b"
)

# plt.show()
plt.savefig(output_dir / "clusters.png")

plt.xlim((df_full.loc[new_record_ids].X_1.min(), stats.iqr(df_full.loc[new_record_ids].X_1, rng=(5, 95))))
plt.ylim((df_full.loc[new_record_ids].X_2.min(), stats.iqr(df_full.loc[new_record_ids].X_2, rng=(5, 95))))
plt.savefig(output_dir / "clusters_zoomed.png")
plt.close()

exit()
