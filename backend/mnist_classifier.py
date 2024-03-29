from os.path import exists

from PIL import Image, ImageOps

import tensorflow as tf
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label


def train_model():
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=10, kernel_size=5, activation="relu",
                               padding='same', input_shape=[28, 28, 1]),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2,
                                  padding='valid'),
        tf.keras.layers.Conv2D(filters=20, kernel_size=5, activation="relu",
                               padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2,
                                  padding='valid'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(units=10, activation='softmax'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    history = model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
    )

    model.save('trained_model/mnist_predictor')

    return model, history


def plot_train_valid(history):
    history_frame = pd.DataFrame(history.history)
    print(history_frame)
    history_frame.loc[:, ['loss', 'val_loss']].plot()
    history_frame.loc[:, ['sparse_categorical_accuracy',
                          'val_sparse_categorical_accuracy']].plot()
    plt.show()


def predict(img_filename):
    if (not (exists('trained_model/mnist_predictor'))):
        model, history = train_model()
        plot_train_valid(history)
    else:
        model = tf.keras.models.load_model('trained_model/mnist_predictor')

    img = Image.open(img_filename).convert('L').resize((28, 28),
                                                       Image.ANTIALIAS)
    img = np.array(img)

    p = model.predict(img[None, :, :])
    print(p)
    pos = np.argmax(p)
    result = str(pos)
    return result


if __name__ == '__main__':
    # img_filename = input("Enter your image filename: ")
    img_filename = "image.png"
    prediction = predict(img_filename)
    print(prediction)
