import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, InputLayer
from tf_lassonet.model import LassoNet
from tf_lassonet.path import LassoPath

model = Sequential(
    [
        InputLayer((28, 28, 1)),        
        Conv2D(5, (3,3), activation="relu", name="conv"),
        Flatten(),
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
    .batch(2048)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

ds_test = (
    ds_test.filter(keep_5_and_6)
    .map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    .map(to_binary)
    .batch(2048)
    .cache()
    .prefetch(tf.data.experimental.AUTOTUNE)
)


path = LassoPath(
    model, n_iters_init=100, patience_init=3, n_iters_path=10, patience_path=2, M=30,
    path_multiplier=1.1
 
)
path.lassonet.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
h = path.fit(ds_train, ds_test, verbose=True)




