import glob
import sys
from random import shuffle

import tensorflow as tf
import numpy as np
import cv2

from sklearn.model_selection import train_test_split

import generators.generators

base_path = '/storage/brno12-cerit/home/xlapsa00/datasets/FaceForensics_dataset'
new_base_path = '/storage/brno12-cerit/home/xlapsa00/datasets/FaceForensics_new_dataset'
# In out case, raw 006_002_270.png has to be removed because it was corrupted (pngcheck)


def read_images(img_path, label):
    try:
        img_data = tf.io.read_file(img_path)
        img = tf.io.decode_png(img_data)
        img = tf.image.convert_image_dtype(img, tf.float32)
        if tf.strings.regex_full_match(img_path, ".*/real/.*"):
            label = tf.convert_to_tensor(1.0, dtype=tf.float32)

        return img, label
    except Exception as e:
        print(f"Corrupted input {img_path}. {e}", file=sys.stderr)
        return None


def read_images_simple(img_path, label):
    try:
        img_data = tf.io.read_file(img_path)
        img = tf.io.decode_png(img_data)
        img = tf.image.convert_image_dtype(img, tf.float32)

        return img, label
    except Exception as e:
        print(f"Corrupted input {img_path}. {e}", file=sys.stderr)
        return None


def init(fcc_shape=480, new=False, compression=None):
    if new:
        if compression is None:
            dataset_path = f"{new_base_path}_{fcc_shape}"
        elif compression == 100:
            dataset_path = f"{new_base_path}_{fcc_shape}_jpeg_{compression}"
        elif compression == 80:
            dataset_path = f"{new_base_path}_{fcc_shape}_jpeg_{compression}"
        elif compression == 60:
            dataset_path = f"{new_base_path}_{fcc_shape}_jpeg_{compression}"
        elif compression == 40:
            dataset_path = f"{new_base_path}_{fcc_shape}_jpeg_{compression}"
        else:
            exit(1)
    else:
        dataset_path = f"{base_path}_{fcc_shape}"
    # Train dataset
    real_images = glob.glob(f"{dataset_path}/train/real/raw/*.png")
    fake_images = glob.glob(f"{dataset_path}/train/fake/raw/*.png")
    if not new:
        fake_images = fake_images[:len(real_images)]
    images = real_images + fake_images
    # images = glob.glob(f"{dataset_path}/train/*/raw/*.png")
    shuffle(images)
    init_labels = [0.0] * len(images)
    dataset = tf.data.Dataset.from_tensor_slices((images, init_labels))
    dataset = dataset.map(read_images, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.shuffle(1024, reshuffle_each_iteration=True)
    dataset = dataset.batch(generators.generators.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    train_dataset_size = dataset.cardinality().numpy() * generators.generators.batch_size
    generators.generators.train_flow = dataset

    # Print the sizes of the train and test datasets:
    # print("Train dataset size:", len(list(dataset)))
    print(f"Train dataset loaded: {train_dataset_size}")
    print(f"Real: {len(real_images)}")
    print(f"Fake: {len(fake_images)}")

    # Validation dataset
    real_images = glob.glob(f"{dataset_path}/val/real/raw/*.png")
    fake_images = glob.glob(f"{dataset_path}/val/fake/raw/*.png")
    if not new:
        fake_images = fake_images[:len(real_images)]
    images = real_images + fake_images
    # images = glob.glob(f"{dataset_path}/val/*/raw/*.png")
    real_labels = [1.0] * len(real_images)
    fake_labels = [0.0] * len(fake_images)
    init_labels = real_labels + fake_labels
    dataset_val = tf.data.Dataset.from_tensor_slices((images, init_labels))
    dataset_val = dataset_val.map(read_images_simple, num_parallel_calls=tf.data.AUTOTUNE)

    dataset_val = dataset_val.batch(generators.generators.batch_size)
    dataset_val = dataset_val.prefetch(tf.data.AUTOTUNE)

    val_dataset_size = dataset_val.cardinality().numpy() * generators.generators.batch_size
    generators.generators.valid_flow = dataset_val

    generators.generators.val_labels = np.array(init_labels)

    # Print the sizes of the train and test datasets:
    print(f"Validation dataset size: {val_dataset_size}")
    print(f"Real: {len(real_images)}")
    print(f"Fake: {len(fake_images)}")
    # print("Validation dataset loaded")

    # Test dataset
    real_images = glob.glob(f"{dataset_path}/test/real/raw/*.png")
    fake_images = glob.glob(f"{dataset_path}/test/fake/raw/*.png")
    if not new:
        fake_images = fake_images[:len(real_images)]
    images = real_images + fake_images
    # images = glob.glob(f"{dataset_path}/val/*/raw/*.png")
    real_labels = [1.0] * len(real_images)
    fake_labels = [0.0] * len(fake_images)
    init_labels = real_labels + fake_labels
    dataset_test = tf.data.Dataset.from_tensor_slices((images, init_labels))
    dataset_test = dataset_test.map(read_images_simple, num_parallel_calls=tf.data.AUTOTUNE)

    dataset_test = dataset_test.batch(generators.generators.batch_size)
    dataset_test = dataset_test.prefetch(tf.data.AUTOTUNE)

    test_dataset_size = dataset_test.cardinality().numpy() * generators.generators.batch_size
    generators.generators.test_flow = dataset_test

    generators.generators.test_labels = np.array(init_labels)

    # Print the sizes of the train and test datasets:
    # print("Train dataset size:", len(list(dataset)))
    print(f"Test dataset loaded: {test_dataset_size}")
    print(f"Real: {len(real_images)}")
    print(f"Fake: {len(fake_images)}")

    # Set steps
    generators.generators.train_steps = dataset.cardinality().numpy()
    generators.generators.valid_steps = dataset_val.cardinality().numpy()

    generators.generators.is_set = True
