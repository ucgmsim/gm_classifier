from sklearn.manifold import LocallyLinearEmbedding
import numpy as np
import pandas as pd

import gm_classifier as gm

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

features_dir = "/home/cbs51/code/gm_classifier/data/records/all_records_features/200401"
output_ffp = "/home/cbs51/code/tmp/gm_classifier/lle.csv"

feature_df = gm.utils.load_features_from_dir(features_dir)

df = feature_df.loc[:, feature_names].copy()
X_full = df.values

# Scale the features
X_full = gm.pre.standardise(X_full, X_full.mean(axis=0), X_full.std(axis=0))

# Apply LLE
print(f"Running LLE")
lle = LocallyLinearEmbedding(n_neighbors=5, n_jobs=-1, neighbors_algorithm="kd_tree")
X_full_trans = lle.fit_transform(X_full)
df["X_1"] = X_full_trans[:, 0]
df["X_2"] = X_full_trans[:, 1]

df.to_save(output_ffp)

exit()