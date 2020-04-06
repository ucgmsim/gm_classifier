from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.neighbors import KDTree
from sklearn.model_selection import KFold
from sklearn.manifold import LocallyLinearEmbedding

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
    "/Users/Clus/code/work/gm_classifier/results/200406_active_learning_score_lle"
)

label_names = ["score_X", "score_Y", "score_Z"]

feature_names = [
    "signal_pe_ratio_max",
    "signal_ratio_max",
    "snr_min",
    "snr_max",
    "snr_average",
    "max_tail_ratio",
    "average_tail_ratio",
    "max_head_ratio",
    "snr_average_0.1_0.2",
    "snr_average_0.2_0.5",
    "snr_average_0.5_1.0",
    "snr_average_1.0_2.0",
    "snr_average_2.0_5.0",
    "snr_average_5.0_10.0",
    "fas_ratio_low",
    "fas_ratio_high",
    "pn_pga_ratio",
    # "is_vertical",
]
feature_names = list(
    np.concatenate(
        (
            np.char.add(feature_names, "_X"),
            np.char.add(feature_names, "_Y"),
            np.char.add(feature_names, "_Z"),
        )
    )
)

# Model details
model_config = {
    "units": [64, 32],
    "act_funcs": "relu",
    "n_outputs": 3,
    "output_act_func": "linear",
    "output_names": label_names,
    "dropout": 0.5,
}

# Pre-processing
feature_pre_config = {"standardise": True, "whiten": True}
# label_pre_config = {"shift": [1, 1, 1, 1, 1, 1]}
label_pre_config = None

# Training details
optimizer = "Adam"
# loss = gm.training.mape
loss = "mse"
n_epochs = 100
batch_size = 32

# ---- Run ----
output_dir = Path(output_dir)

label_df = gm.utils.load_labels_from_dir(label_dir)
feature_df = gm.utils.load_features_from_dir(features_dir)

train_df = pd.merge(feature_df, label_df, left_index=True, right_index=True)
train_df.to_csv(output_dir / "training_data.csv")

ids = train_df.index.values.astype(str)
X = train_df.loc[:, feature_names].values
y = train_df.loc[:, label_names].values

# Use K-fold to get estimated values for every labelled sample
kf = KFold(n_splits=5, shuffle=True)
r_ids, r_y_val, r_y_val_est = [], [], []
for ix, (train_ind, val_ind) in enumerate(kf.split(X)):
    cur_id = f"iter_{ix}"

    # Create the output directory
    cur_output_dir = output_dir / cur_id
    cur_output_dir.mkdir()

    # Get the current training and validation data
    X_train, y_train, ids_train = X[train_ind], y[train_ind], ids[train_ind]
    X_val, y_val, ids_val = X[val_ind], y[val_ind], ids[val_ind]

    gm.training.train(
        cur_output_dir,
        feature_pre_config,
        model_config,
        (X_train, y_train, ids_train),
        val_data=(X_val, y_val, ids_val),
        label_config=None,
        compile_kwargs={"optimizer": optimizer, "loss": loss},
        fit_kwargs={"batch_size": batch_size, "epochs": n_epochs, "verbose": 2},
    )

    cur_y_est = gm.predict.predict(cur_output_dir, X_val, feature_pre_config)

    r_ids.append(ids_val)
    r_y_val.append(y_val)
    r_y_val_est.append(cur_y_est)

r_ids = np.concatenate(r_ids)
r_y_val = np.concatenate(r_y_val)
r_y_val_est = np.concatenate(r_y_val_est)


# --- LLE + nearest neighbour selection ----
df = pd.DataFrame(
    data=np.concatenate((r_y_val, r_y_val_est), axis=1),
    index=r_ids,
    columns=np.concatenate((label_names, np.char.add(label_names, "_est"))),
)
df["res_X"] = r_y_val[:, 0] - r_y_val_est[:, 0]
df["res_Y"] = r_y_val[:, 1] - r_y_val_est[:, 1]
df["res_Z"] = r_y_val[:, 2] - r_y_val_est[:, 2]
df["res_total"] = np.abs(df.res_X) + np.abs(df.res_Y) + np.abs(df.res_Z)

df_full = feature_df.loc[:, feature_names].copy()
X_full = df_full.values

# Scale the features
X_full = gm.pre.standardise(X_full, X_full.mean(axis=0), X_full.std(axis=0))

# Apply LLE or load LLE
# print(f"Running LLE")
# lle = LocallyLinearEmbedding(n_neighbors=5, n_jobs=-1)
# X_full_trans = lle.fit_transform(X_full)
# df_full["X_1"] = X_full_trans[:, 0]
# df_full["X_2"] = X_full_trans[:, 2]


# Select the X-nearest neighbours of the datapoints with the worst residuals
X_sel = X_full

kd_tree = KDTree(X_sel)


exit()
