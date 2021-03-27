import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import gm_classifier as gmc

fmin_weight_mapping = {1.0: 1.0, 0.75: 1.0, 0.5:0.5, 0.25: 0.3, 0.0:0.1}

loss_fn = gmc.training.FMinLoss(fmin_weight_mapping)

fmin_true = 10.0
score_values = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
fmin_values = np.linspace(0.1, 10.0, 100)

x, y = np.meshgrid(score_values, fmin_values, indexing="xy")
pred = np.stack((np.ones_like(y.ravel()), y.ravel()), axis=1)
true = np.stack((x.ravel(), np.ones_like(x.ravel()) * fmin_true), axis=1)

z = loss_fn.compute_sample_loss(tf.constant(true, dtype=tf.float32), tf.constant(pred, dtype=tf.float32)).numpy()

plt.figure(figsize=(16, 10))

c = z.reshape(x.shape)
plt.pcolormesh(x, y, c, shading="nearest")
# plt.contourf(x, y, c)
plt.colorbar()
plt.xlabel("True score")
plt.ylabel("Fmin prediction")
plt.title(fmin_true)

plt.savefig("/home/claudy/dev/work/tmp/plot.png")

exit()