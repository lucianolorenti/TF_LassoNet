from tf_lassonet.path import LassoPath
from tf_lassonet.model import LassoNet
from typing import Optional, List
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, InputLayer
from tensorflow.keras import Model, Sequential
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from dataclasses import dataclass


model = Sequential(
    [
        InputLayer((28, 28, 1)),
        Flatten(),
        Dense(16, activation="relu", name="layer1"),
        
        Dense(2, name="layer4"),
    ]
)

lassonet = LassoNet(model)
(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


def keep_5_and_6(x, y):
    if y == 5 or y == 6:
        return True
    else:
        return False


def to_binary(x, y):
    if y== 5:
        return x, 0
    else:
        return x, 1

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label


ds_train = (
    ds_train.filter(keep_5_and_6)
    .map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .map(to_binary)
    .cache()
    .shuffle(ds_info.splits["train"].num_examples)
    .batch(len(ds_train))
    .prefetch(tf.data.experimental.AUTOTUNE)
)

ds_test = (
    ds_test.filter(keep_5_and_6)
    .map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .map(to_binary)
    .batch(len(ds_test))
    .cache()
    .prefetch(tf.data.experimental.AUTOTUNE)
)


path = LassoPath(
    model, n_iters_init=100, patience_init=3, n_iters_path=10, patience_path=2, M=30,
    path_multiplier=1.1
 
)
h = path.fit(ds_train, ds_test)




theta = lassonet.theta.weights[0].numpy().reshape(28, 28)
plt.imshow(theta)
plt.show()
img = qqq[0][3, :, :, 0] / 255.0
fig, ax = plt.subplots(1, 2)
ax[0].imshow(theta * img)
ax[1].imshow(img)
fig.show()
