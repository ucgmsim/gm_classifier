"""Dimensionality reduces the feature space using an autoencoder
and then performs k-means clustering. From each cluster a x-random
samples are selected to form the testing dataset.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import gm_classifier as gm


# --- Load the data ---
feature_dir = "/Users/Clus/code/work/gm_classifier/data/records/local/training_data/all_records_features/200429"
label_dir = (
    "/Users/Clus/code/work/gm_classifier/data/records/local/training_data/labels"
)
output_dir = Path(
    "/Users/Clus/code/work/gm_classifier/results/200506_results/test_dataset"
)

feature_df = gm.utils.load_features_from_dir(feature_dir)
label_df = gm.utils.load_labels_from_dir(label_dir, f_min_100_value=10)

# --- Pre-processing ---
feature_names, _ = gm.utils.get_full_feature_config(gm.RecordCompModel.feature_config)

df_full = feature_df.loc[:, feature_names].copy()
X_full = df_full.drop(
    ["is_vertical_X", "is_vertical_Y", "is_vertical_Z"], axis=1
).values

X_full = gm.pre.standardise(X_full, X_full.mean(axis=0), X_full.std(axis=0))

# --- Dim reduction, clustering & selection ---
n_dims = 2

# Dim reduction
enc, X_trans, rec_loss = gm.dim_reduction.ae(
    X_full, n_dims=n_dims, fit_kwargs={"epochs": 200, "batch_size": 128}
)
enc.save(output_dir / "encoder.h5")

df_full["X_1"] = X_trans[:, 0]
df_full["X_2"] = X_trans[:, 1]
if n_dims == 3:
    df_full["X_3"] = X_trans[:, 2]

# K-means
print(f"Running k-means clustering")
k_means = KMeans(n_clusters=100, max_iter=300, n_jobs=1)
k_means_labels = k_means.fit_predict(X_trans)
df_full["cluster"] = k_means_labels

# Select samples per cluster
print(f"Selecting samples from clusters")
n = 2
new_record_ids = []
for cur_label in np.unique(k_means_labels):
    mask = df_full.cluster == cur_label

    cur_n = np.count_nonzero(mask) if np.count_nonzero(mask) < n else n
    new_record_ids.append(df_full.loc[mask].sample(cur_n).index.values)

new_record_ids = np.concatenate(new_record_ids).astype(str)
new_record_ids = new_record_ids[~np.isin(new_record_ids, label_df.index.values)]

with open(output_dir / "record_ids.txt", "w") as f:
    f.writelines([f"{cur_id}\n" for cur_id in new_record_ids])

# Result plot
plt.figure(figsize=(16, 10))
c_m_combs = gm.plots.get_color_marker_comb()
for cur_label, (cur_c, cur_m) in zip(np.unique(k_means_labels), c_m_combs):
    cur_mask = k_means_labels == cur_label
    plt.scatter(
        X_trans[cur_mask, 0], X_trans[cur_mask, 1], c=cur_c, marker=cur_m, s=1.0
    )

label_record_ids = label_df.index.values[np.isin(label_df.index, df_full.index)].astype(str)
plt.scatter(
    df_full.loc[label_record_ids, "X_1"],
    df_full.loc[label_record_ids, "X_2"],
    s=5.0,
    marker="X",
    c="k",
)
plt.scatter(
    df_full.loc[new_record_ids].X_1,
    df_full.loc[new_record_ids].X_2,
    s=5.0,
    marker="X",
    c="b",
)
plt.savefig(output_dir / "clusters.png")

exit()
