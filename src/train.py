import tensorflow as tf
import pandas as pd


def load_data(train_path: str, test_path: str) -> None:
    # load train dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        image_size=(48, 48),
        batch_size=64,
        color_mode='grayscale',
        label_mode='categorical'
    )

    # load test dataset
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path,
        image_size=(48, 48),
        batch_size=64,
        color_mode='grayscale',
        label_mode='categorical'
    )

    print(train_ds.class_names)
    print(test_ds.class_names)

load_data('data/train', 'data/test') 